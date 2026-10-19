using System.Globalization;
using System.Numerics;
using System.Text;
using System.Text.Json;

namespace Laya;

/// <summary>
/// <c>json.loads</c> as CPython's C scanner implements it, producing the values Python would hold:
/// <see cref="long"/> / <see cref="BigInteger"/> for an integer literal, <see cref="double"/> for a float
/// literal (a literal that overflows is infinity), <see cref="string"/>, <see cref="bool"/>,
/// <see langword="null"/>, <c>List&lt;object?&gt;</c> and <c>OrderedDictionary&lt;string, object?&gt;</c>.
/// </summary>
/// <remarks>
/// <see cref="JsonDocument"/> is stricter than Python in exactly one place that matters here: Python also
/// accepts the bare tokens <c>NaN</c>, <c>Infinity</c> and <c>-Infinity</c> in value position. They are
/// parsed to doubles directly, so no marker text exists that a real string could collide with. Everything
/// else follows <c>json.decoder</c>: whitespace is space, tab, LF and CR only; <c>-?(0|[1-9][0-9]*)</c> with
/// an optional fraction and exponent, so <c>01</c>, <c>+1</c>, <c>.5</c> and <c>1.</c> are errors; strings
/// reject control characters but keep lone surrogates; no trailing commas, comments or single quotes; a
/// duplicate key keeps its first position and its last value. Errors are <see cref="JsonException"/>.
/// </remarks>
internal static class PythonJsonReader
{
    // Python 3.11+ refuses to turn more digits than this into an int (sys.int_info.default_max_str_digits).
    private const int MaxIntDigits = 4300;
    private const int MaxDepth = 1000;

    /// <summary>
    /// Read a JSON file the way Python's <c>json.load(open(path))</c> does. The reference opens these files
    /// without an <c>encoding</c>, so it decodes with the locale's: UTF-8 on Linux and macOS (and with
    /// Python's UTF-8 mode), the ANSI code page on a default Windows install. This reads UTF-8, strictly: invalid bytes are an
    /// error (<c>UnicodeDecodeError</c>), and a leading BOM is kept as U+FEFF so <see cref="Parse"/> rejects it as
    /// <c>json.loads</c> does. The two decodings agree on every ASCII file.
    /// </summary>
    internal static string ReadAllText(string path)
    {
        try { return new UTF8Encoding(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true).GetString(File.ReadAllBytes(path)); }
        catch (DecoderFallbackException e) { throw new JsonException($"{path} is not valid UTF-8: {e.Message}", e); }
    }

    internal static object? Parse(string text)
    {
        if (text.StartsWith('\uFEFF')) throw new JsonException("Unexpected UTF-8 BOM (decode using utf-8-sig): char 0");
        var p = new State(text);
        p.SkipWhitespace();
        var value = p.Value(0);
        p.SkipWhitespace();
        if (p.Pos != text.Length) throw p.Error("Extra data");
        return value;
    }

    /// <summary>
    /// Parse, then re-emit as a <see cref="JsonDocument"/> for code that reads through that API. NaN and
    /// the infinities, which JSON text cannot hold, become <c>null</c>.
    /// </summary>
    internal static JsonDocument ParseDocument(string text)
    {
        var value = Parse(text);
        using var stream = new MemoryStream();
        using (var writer = new Utf8JsonWriter(stream))
            Write(writer, value);
        return JsonDocument.Parse(stream.ToArray());
    }

    private static void Write(Utf8JsonWriter w, object? value)
    {
        switch (value)
        {
            case null: w.WriteNullValue(); break;
            case bool b: w.WriteBooleanValue(b); break;
            case long l: w.WriteNumberValue(l); break;
            case BigInteger big: w.WriteRawValue(big.ToString(CultureInfo.InvariantCulture)); break;
            case double d when double.IsFinite(d): w.WriteNumberValue(d); break;
            case double: w.WriteNullValue(); break;
            case string s: w.WriteStringValue(ReplaceLoneSurrogates(s)); break;
            case List<object?> list:
                w.WriteStartArray();
                foreach (var item in list) Write(w, item);
                w.WriteEndArray();
                break;
            case IEnumerable<KeyValuePair<string, object?>> map:
                w.WriteStartObject();
                foreach (var kv in map) { w.WritePropertyName(ReplaceLoneSurrogates(kv.Key)); Write(w, kv.Value); }
                w.WriteEndObject();
                break;
        }
    }

    private static string ReplaceLoneSurrogates(string s)
    {
        var sb = new StringBuilder(s.Length);
        foreach (var rune in s.EnumerateRunes()) sb.Append(rune.ToString());
        return sb.ToString();
    }

    private sealed class State(string text)
    {
        internal int Pos;

        internal JsonException Error(string message) => new($"{message}: char {Pos}");

        internal void SkipWhitespace()
        {
            while (Pos < text.Length && text[Pos] is ' ' or '\t' or '\n' or '\r') Pos++;
        }

        internal object? Value(int depth)
        {
            if (depth > MaxDepth) throw Error("maximum recursion depth exceeded");
            if (Pos >= text.Length) throw Error("Expecting value");
            switch (text[Pos])
            {
                case '"': Pos++; return String();
                case '{': Pos++; return Object(depth + 1);
                case '[': Pos++; return Array(depth + 1);
                case 'n' when Literal("null"): return null;
                case 't' when Literal("true"): return true;
                case 'f' when Literal("false"): return false;
                case 'N' when Literal("NaN"): return double.NaN;
                case 'I' when Literal("Infinity"): return double.PositiveInfinity;
                case '-' when string.CompareOrdinal(text, Pos, "-Infinity", 0, 9) == 0: Pos += 9; return double.NegativeInfinity;
                default:
                    var number = Number();
                    return number ?? throw Error("Expecting value");
            }
        }

        private bool Literal(string word)
        {
            if (string.CompareOrdinal(text, Pos, word, 0, word.Length) != 0) return false;
            Pos += word.Length;
            return true;
        }

        private static bool Digit(char c) => c is >= '0' and <= '9';

        // _match_number_unicode: -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][-+]?[0-9]+)?
        private object? Number()
        {
            var start = Pos;
            var i = Pos;
            if (i < text.Length && text[i] == '-') i++;
            if (i >= text.Length || !Digit(text[i])) return null;
            if (text[i] == '0') i++;
            else while (i < text.Length && Digit(text[i])) i++;
            var integerEnd = i;
            var isFloat = false;
            if (i + 1 < text.Length && text[i] == '.' && Digit(text[i + 1]))
            {
                i += 2;
                while (i < text.Length && Digit(text[i])) i++;
                isFloat = true;
            }
            if (i < text.Length && text[i] is 'e' or 'E')
            {
                var j = i + 1;
                if (j < text.Length && text[j] is '+' or '-') j++;
                if (j < text.Length && Digit(text[j]))
                {
                    while (j < text.Length && Digit(text[j])) j++;
                    i = j;
                    isFloat = true;
                }
            }
            var literal = text[start..i];
            Pos = i;
            if (isFloat)
                return double.Parse(literal, NumberStyles.Float, CultureInfo.InvariantCulture);   // correctly rounded; overflow is infinity
            var digits = integerEnd - start - (text[start] == '-' ? 1 : 0);
            if (digits > MaxIntDigits)
                throw new JsonException($"Exceeds the limit ({MaxIntDigits} digits) for integer string conversion: value has {digits} digits");
            return long.TryParse(literal, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out var l)
                ? (object)l   // not a bare `l`: the conditional would widen it to BigInteger
                : BigInteger.Parse(literal, CultureInfo.InvariantCulture);
        }

        // py_scanstring with strict=True: control characters below U+0020 are errors, lone surrogates stay.
        private string String()
        {
            var sb = new StringBuilder();
            while (true)
            {
                if (Pos >= text.Length) throw Error("Unterminated string starting at");
                var c = text[Pos++];
                if (c == '"') return sb.ToString();
                if (c < ' ') throw Error("Invalid control character at");
                if (c != '\\') { sb.Append(c); continue; }
                if (Pos >= text.Length) throw Error("Unterminated string starting at");
                var e = text[Pos++];
                switch (e)
                {
                    case '"': sb.Append('"'); break;
                    case '\\': sb.Append('\\'); break;
                    case '/': sb.Append('/'); break;
                    case 'b': sb.Append('\b'); break;
                    case 'f': sb.Append('\f'); break;
                    case 'n': sb.Append('\n'); break;
                    case 'r': sb.Append('\r'); break;
                    case 't': sb.Append('\t'); break;
                    case 'u': sb.Append(Hex4()); break;
                    default: throw Error("Invalid \\escape");
                }
            }
        }

        private char Hex4()
        {
            if (Pos + 4 > text.Length) throw Error("Invalid \\uXXXX escape");
            var value = 0;
            for (var k = 0; k < 4; k++)
            {
                var h = text[Pos + k];
                var digit = h is >= '0' and <= '9' ? h - '0' : h is >= 'a' and <= 'f' ? h - 'a' + 10 : h is >= 'A' and <= 'F' ? h - 'A' + 10 : -1;
                if (digit < 0) throw Error("Invalid \\uXXXX escape");
                value = value * 16 + digit;
            }
            Pos += 4;
            return (char)value;   // a high surrogate followed by \uDC00.. joins in the UTF-16 string, as Python joins the pair
        }

        private object Object(int depth)
        {
            var map = new OrderedDictionary<string, object?>(StringComparer.Ordinal);
            SkipWhitespace();
            if (Pos < text.Length && text[Pos] == '}') { Pos++; return map; }
            while (true)
            {
                if (Pos >= text.Length || text[Pos] != '"') throw Error("Expecting property name enclosed in double quotes");
                Pos++;
                var key = String();
                SkipWhitespace();
                if (Pos >= text.Length || text[Pos] != ':') throw Error("Expecting ':' delimiter");
                Pos++;
                SkipWhitespace();
                map[key] = Value(depth);
                SkipWhitespace();
                if (Pos < text.Length && text[Pos] == '}') { Pos++; return map; }
                if (Pos >= text.Length || text[Pos] != ',') throw Error("Expecting ',' delimiter");
                Pos++;
                SkipWhitespace();
            }
        }

        private object Array(int depth)
        {
            var list = new List<object?>();
            SkipWhitespace();
            if (Pos < text.Length && text[Pos] == ']') { Pos++; return list; }
            while (true)
            {
                list.Add(Value(depth));
                SkipWhitespace();
                if (Pos < text.Length && text[Pos] == ']') { Pos++; return list; }
                if (Pos >= text.Length || text[Pos] != ',') throw Error("Expecting ',' delimiter");
                Pos++;
                SkipWhitespace();
            }
        }
    }
}

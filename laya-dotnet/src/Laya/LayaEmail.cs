using System.Text;
using System.Text.RegularExpressions;

namespace Laya;

/// <summary>Email utilities for cleaning and structuring email inputs. Ports <c>laya/email.py</c>.</summary>
public static partial class LayaEmail
{
    // Python's \b is a boundary between \w and \W with \w = str.isalnum() or "_"; .NET's \w also takes
    // marks and connector punctuation and leaves out Nl/No. [\p{L}\p{N}_] is Python's \w.
    private const string Wb = @"(?:(?<=[\p{L}\p{N}_])(?![\p{L}\p{N}_])|(?<![\p{L}\p{N}_])(?=[\p{L}\p{N}_]))";

    // re.match anchors at the start only; every pattern below carries its own leading `^`, so
    // .NET's Regex.IsMatch (which is not implicitly start-anchored) reproduces that exactly.
    // `re.I` -> RegexOptions.IgnoreCase.
    [GeneratedRegex(@"^[\s\x1c-\x1f]*On .{0,300}wrote:[\s\x1c-\x1f]*$", RegexOptions.IgnoreCase)]
    private static partial Regex QuoteHeaderOnWrote();

    [GeneratedRegex(@"^[\s\x1c-\x1f]*(Em (?=.*\d).{0,300}escreveu:|El (?=.*\d).{0,300}escr[iIıİ]b[iIıİ][óo]:|Le (?=.*\d).{0,300}a [eé]cr[iIıİ]t[\s\x1c-\x1f]*:)[\s\x1c-\x1f]*$", RegexOptions.IgnoreCase)]
    private static partial Regex QuoteHeaderTranslated();

    [GeneratedRegex(@"^[\s\x1c-\x1f]*-{2,}[\s\x1c-\x1f]*(Or[iIıİ]g[iIıİ]nal|Forwarded) Message[\s\x1c-\x1f]*-{2,}", RegexOptions.IgnoreCase)]
    private static partial Regex QuoteHeaderOriginalOrForwarded();

    [GeneratedRegex(@"^[\s\x1c-\x1f]*-{2,}[\s\x1c-\x1f]*(Mensagem (or[iIıİ]g[iIıİ]nal|encam[iIıİ]nhada)|Mensaje (or[iIıİ]g[iIıİ]nal|reenv[iIıİ]ado)|Message d'or[iIıİ]g[iIıİ]ne)[\s\x1c-\x1f]*-{2,}", RegexOptions.IgnoreCase)]
    private static partial Regex QuoteHeaderTranslatedMessage();

    [GeneratedRegex(@"^[\s\x1c-\x1f]*_{8,}[\s\x1c-\x1f]*$")]
    private static partial Regex QuoteHeaderUnderscores();

    [GeneratedRegex(@"^[\s\x1c-\x1f]*(From:|De[\s\x1c-\x1f]*:)[\s\x1c-\x1f].*[@<]", RegexOptions.IgnoreCase)]
    private static partial Regex QuoteHeaderFromLine();

    [GeneratedRegex(@"^[\s\x1c-\x1f]*(De|From)[\s\x1c-\x1f]*:[\s\x1c-\x1f]+[^\s\x1c-\x1f]", RegexOptions.IgnoreCase)]
    private static partial Regex HeaderFromName();

    [GeneratedRegex(@"^[\s\x1c-\x1f]*(Env[iIıİ]ad[oa]( em| el)?:[\s\x1c-\x1f]|Envoy[ée]( le)?[\s\x1c-\x1f]*:[\s\x1c-\x1f]|Sent:[\s\x1c-\x1f]|(Data|Fecha|Date):[\s\x1c-\x1f].*\d{4})", RegexOptions.IgnoreCase)]
    private static partial Regex HeaderNext();

    [GeneratedRegex(@"^.{0,120}[^\s\x1c-\x1f]@[^\s\x1c-\x1f]+[\s\x1c-\x1f]+(wrote|escreveu|escr[iIıİ]b[iIıİ][óo]|a [eé]cr[iIıİ]t)[\s\x1c-\x1f]*:[\s\x1c-\x1f]*$", RegexOptions.IgnoreCase)]
    private static partial Regex AttributionTail();

    [GeneratedRegex(@"^[\s\x1c-\x1f]*(On|Em|El|Le) (?=.*\d)", RegexOptions.IgnoreCase)]
    private static partial Regex AttributionHead();

    [GeneratedRegex(@"^[\s\x1c-\x1f]*--[\s\x1c-\x1f]*$")]
    private static partial Regex SignatureDashDash();

    [GeneratedRegex(@"^[\s\x1c-\x1f]*(?i:best|k[iIıİ]nd|warmest|warm|many thanks|thanks|thank you|regards|cheers|s[iIıİ]ncerely)"
        + @"(?i:[\s\x1c-\x1f]+(?:and|&)[\s\x1c-\x1f]+regards|[\s\x1c-\x1f]+(?:regards|w[iIıİ]shes|aga[iIıİ]n|[iIıİ]n advance|a lot|so much|very much))?")]
    private static partial Regex SignatureClosingHead();

    [GeneratedRegex(@"^[\s\x1c-\x1f]*(atenc[iIıİ]osamente|att|abraços?|abs|um abraço|cord[iIıİ]almente|grat[oa]|(mu[iIıİ]to )?obr[iIıİ]gad[oa]s?"
        + @"( desde já| pela atenção)?|(com os melhores )?cumpr[iIıİ]mentos|saudações|"
        + @"(un )?saludos?( cord[iIıİ]ales)?|atentamente|(muchas )?grac[iIıİ]as( de antemano)?|"
        + @"(b[iIıİ]en )?cord[iIıİ]alement|salutat[iIıİ]ons( d[iIıİ]st[iIıİ]nguées)?|b[iIıİ]en à vous|merc[iIıİ]( d'avance)?|bonne journée)[\s\x1c-\x1f,!.]*$",
        RegexOptions.IgnoreCase)]
    private static partial Regex SignatureTranslated();

    private const string Device = @"[iIıİ]phone|[iIıİ]pad|andro[iIıİ]d|[iIıİ]os|mob[iIıİ]le|celular|telemóvel|móv[iIıİ]l|galaxy|smartphone|samsung|tablet|outlook|yahoo|ma[iIıİ]l|e-?ma[iIıİ]l|gma[iIıİ]l|w[iIıİ]ndows";

    [GeneratedRegex(@"^[\s\x1c-\x1f]*((env[iIıİ]ad[oa] (do|pelo|pela|v[iIıİ]a|desde|a part[iIıİ]r do)( meu| m[iIıİ]nha| m[iIıİ])?|sent from( my)?|"
        + @"envoy[ée] (depu[iIıİ]s|de)( mon| ma| mes)?)"
        + @" (" + Device + @")( (" + Device + @"|para|for|no|na|\d+|phone|dev[iIıİ]ce|pro|max|m[iIıİ]n[iIıİ]|plus|us[iIıİ]ng [a-zıİ][a-z0-9_.+ıİ-]*))*"
        + @"|(obter o|get) outlook (para|for) ([iIıİ]os|andro[iIıİ]d))[\s\x1c-\x1f.!]*$", RegexOptions.IgnoreCase)]
    private static partial Regex SignatureSentFromMobile();

    // .search() semantics: no anchors, matches anywhere in the text.
    [GeneratedRegex(
        @"(" + Wb + @"(e-?ma[iIıİ]l|message|[iIıİ]nformat[iIıİ]on|commun[iIıİ]cat[iIıİ]on|transm[iIıİ]ss[iIıİ]on|contents?)" + Wb + @"[^.]{0,60}"
        + @"" + Wb + @"conf[iIıİ]dent[iIıİ]al" + Wb + @"[^.]{0,60}" + Wb + @"([iIıİ]ntended|solely|addressee|rec[iIıİ]p[iIıİ]ent|pr[iIıİ]v[iIıİ]leged|d[iIıİ]sclos|unauthor[iIıİ][sz]ed)|"
        + @"" + Wb + @"conf[iIıİ]dent[iIıİ]al" + Wb + @"[^.]{0,60}" + Wb + @"(and (may|[iIıİ]s) (also )?pr[iIıİ]v[iIıİ]leged)|"
        + @"[iIıİ]f you (have )?rece[iIıİ]ved th[iIıİ]s (e-?ma[iIıİ]l|message) [iIıİ]n error|"
        + @"" + Wb + @"(esta|este) (mensagem|e-?ma[iIıİ]l|mensaje|correo)" + Wb + @"[^.]{0,80}(conf[iIıİ]denc[iIıİ]a|s[iIıİ]g[iIıİ]los|pr[iIıİ]v[iIıİ]leg[iIıİ]ad)|"
        + @"" + Wb + @"(uso exclus[iIıİ]vo|exclus[iIıİ]vamente|ún[iIıİ]camente|un[iIıİ]camente)" + Wb + @"[^.]{0,30}(dest[iIıİ]natár[iIıİ]|dest[iIıİ]natar[iIıİ]|pessoa|persona|ent[iIıİ]dade|ent[iIıİ]dad)|"
        + @"" + Wb + @"(recebeu|receb[iIıİ]do|receber) (esta|este) (mensagem|e-?ma[iIıİ]l)" + Wb + @"[^.]{0,20} por (engano|erro)|"
        + @"" + Wb + @"(ha rec[iIıİ]b[iIıİ]do|rec[iIıİ]b[iIıİ]ó|rec[iIıİ]be) (este|esta) (mensaje|correo)" + Wb + @"[^.]{0,20} por error|"
        + @"" + Wb + @"antes de [iIıİ]mpr[iIıİ]m[iIıİ]r" + Wb + @"[^.]{0,100}(me[iIıİ]o amb[iIıİ]ente|med[iIıİ]o amb[iIıİ]ente|natureza|planeta|realmente necess)|"
        + @"" + Wb + @"(me[iIıİ]o|med[iIıİ]o) amb[iIıİ]ente" + Wb + @"[^.]{0,30}antes de [iIıİ]mpr[iIıİ]m[iIıİ]r|"
        + @"" + Wb + @"(ce|cet|cette) (message|e-?ma[iIıİ]l|ma[iIıİ]l|courr[iIıİ]el)" + Wb + @"[^.]{0,80}(conf[iIıİ]dent[iIıİ]el|pr[iIıİ]v[iIıİ]l[eé]g[iIıİ])|"
        + @"" + Wb + @"avez re[çc]u (ce|cet|cette) (message|e-?ma[iIıİ]l|ma[iIıİ]l)" + Wb + @"[^.]{0,20} par erreur|"
        + @"" + Wb + @"(usage exclus[iIıİ]f|exclus[iIıİ]vement|un[iIıİ]quement)" + Wb + @"[^.]{0,30}dest[iIıİ]nata[iIıİ]re)",
        RegexOptions.IgnoreCase)]
    private static partial Regex Disclaimer();

    // Sentence boundary: split just after a `.`/`!`/`?` and the whitespace that follows it. .NET
    // Regex supports the lookbehind natively, so this needs no hand-written splitter.
    [GeneratedRegex(@"(?<=[.!?])[\s\x1c-\x1f]+")]
    private static partial Regex SentenceBoundary();

    [GeneratedRegex(@"\n[\s\x1c-\x1f]*\n")]
    private static partial Regex ParagraphBreak();

    [GeneratedRegex(@"[ \t]+")]
    private static partial Regex SpaceTabRun();

    // Python's `re` works on code points; .NET's on UTF-16 units, where a character outside the BMP is
    // two units that `.`, `{0,n}` counts and `\b` would treat as two non-word characters. Every
    // pattern here is only asked "does it match", so each such character is first replaced by one BMP
    // stand-in of the same word class (a letter or number by U+4E00, anything else by U+2603), which
    // no pattern mentions.
    private static string CodePoints(string s)
    {
        if (!s.Any(char.IsSurrogate)) return s;
        var sb = new StringBuilder(s.Length);
        foreach (var rune in s.EnumerateRunes())
            sb.Append(rune.IsBmp ? rune.ToString() : Rune.IsLetter(rune) || Rune.IsNumber(rune) ? "一" : "☃");
        return sb.ToString();
    }

    private static bool Hit(this Regex regex, string text) => regex.IsMatch(CodePoints(text));

    private static readonly Regex[] QuoteHeaders =
    [
        QuoteHeaderOnWrote(), QuoteHeaderTranslated(), QuoteHeaderOriginalOrForwarded(),
        QuoteHeaderTranslatedMessage(), QuoteHeaderUnderscores(), QuoteHeaderFromLine(),
    ];

    private static readonly Regex[] SignatureMarkers =
    [
        SignatureDashDash(), SignatureTranslated(),
    ];

    /// <summary>Remove quoted email history, signatures and disclaimers to keep input focused.</summary>
    /// <remarks>
    /// <paramref name="maxChars"/> is a slice budget, not a floor, applied twice exactly as in Python:
    /// as the <c>max_chars * 4</c> pre-cut that bounds the regex work and as the final
    /// <c>text[:max_chars]</c>. <c>0</c> answers with the empty string and a negative value trims from
    /// the end (<c>5 * |max_chars|</c> characters off the body in all), as Python's slicing does.
    /// </remarks>
    public static string CleanBody(string? body, int maxChars = 3000)
    {
        var text = (body ?? string.Empty).Replace("\r\n", "\n").Replace("\r", "\n").Replace("\\n", "\n");
        if (CodePointLength(text) > (long)maxChars * 4)
            text = SliceTo(text, (long)maxChars * 4);

        var lines = new List<string>();
        var source = text.Split('\n');
        for (var i = 0; i < source.Length; i++)
        {
            var line = source[i];
            if (lines.Count > 0 && Array.Exists(QuoteHeaders, p => p.Hit(line))) break;
            if (lines.Count > 0 && HeaderFromName().Hit(line) && i + 1 < source.Length
                && HeaderNext().Hit(source[i + 1])) break;
            if (lines.Count > 0 && AttributionTail().Hit(line))
            {
                if (AttributionHead().Hit(lines[^1])) lines.RemoveAt(lines.Count - 1);
                break;
            }
            if (PythonStr.LStrip(line).StartsWith('>')) continue;
            lines.Add(PythonStr.RStrip(line));
        }

        var cut = lines.Count;
        var start = Math.Max(1, Math.Min((int)(lines.Count * 0.6), lines.Count - 8));
        for (var i = start; i < lines.Count; i++)
        {
            var stripped = PythonStr.Strip(lines[i]);
            var length = CodePointLength(stripped);
            if ((length <= 40 && (IsEnglishSignoff(lines[i]) || Array.Exists(SignatureMarkers, p => p.Hit(lines[i]))))
                || (length <= 60 && SignatureSentFromMobile().Hit(lines[i])))
            {
                cut = i;
                break;
            }
        }
        lines = lines.GetRange(0, cut);

        var paragraphs = ParagraphBreak().Split(string.Join('\n', lines)).Select(StripDisclaimer);
        var collapsed = string.Join("\n\n", paragraphs.Select(PythonStr.Strip).Where(p => p.Length > 0));
        text = SpaceTabRun().Replace(collapsed, " ");
        return SliceTo(text, maxChars);
    }

    private static bool IsEnglishSignoff(string line)
    {
        var head = SignatureClosingHead().Match(line);
        if (!head.Success) return false;
        var tail = new List<Rune>();
        foreach (var rune in line[head.Length..].EnumerateRunes())
        {
            if (Rune.GetUnicodeCategory(rune) is System.Globalization.UnicodeCategory.NonSpacingMark
                or System.Globalization.UnicodeCategory.SpacingCombiningMark
                or System.Globalization.UnicodeCategory.EnclosingMark
                && tail.Count > 0 && !PythonStr.IsSpace(tail[^1].Value)) continue;
            tail.Add(rune);
        }
        var i = 0;
        while (i < tail.Count && (PythonStr.IsSpace(tail[i].Value) || tail[i].Value is ',' or ';' or ':' or '!' or '.')) i++;
        var tokens = 0;
        while (i < tail.Count)
        {
            if (++tokens > 3 || Rune.GetUnicodeCategory(tail[i]) is not
                (System.Globalization.UnicodeCategory.UppercaseLetter
                or System.Globalization.UnicodeCategory.TitlecaseLetter
                or System.Globalization.UnicodeCategory.OtherLetter)) return false;
            i++;
            while (i < tail.Count && (Rune.IsLetterOrDigit(tail[i])
                || Rune.GetUnicodeCategory(tail[i]) is System.Globalization.UnicodeCategory.LetterNumber
                    or System.Globalization.UnicodeCategory.OtherNumber
                || tail[i].Value is '_' or '\'' or '-')) i++;
            while (i < tail.Count && (PythonStr.IsSpace(tail[i].Value) || tail[i].Value is ',' or '.')) i++;
        }
        return true;
    }

    /// <summary>
    /// Drop boilerplate disclaimer text from one paragraph.
    /// </summary>
    /// <remarks>
    /// A paragraph is dropped whole only when *every* sentence in it is boilerplate; otherwise
    /// only the boilerplate sentences go. A footer that runs on without a blank line would
    /// otherwise take the sender's actual request with it, which is worse than leaving one
    /// boilerplate line behind.
    /// </remarks>
    private static string StripDisclaimer(string paragraph)
    {
        if (!Disclaimer().Hit(paragraph)) return paragraph; // nothing to do: keep the original line structure
        var pieces = new List<string>();
        foreach (var part in SentenceBoundary().Split(paragraph).Select(PythonStr.Strip).Where(p => p.Length > 0))
        {
            if (!Disclaimer().Hit(part)) { pieces.Add(part); continue; }
            var buffer = "";
            foreach (var line in part.Split('\n').Select(PythonStr.Strip).Where(l => l.Length > 0))
            {
                var firstLetter = line.EnumerateRunes().FirstOrDefault(Rune.IsLetter);
                if (buffer.Length > 0 && PythonStr.IsUpper(firstLetter))
                {
                    pieces.Add(buffer);
                    buffer = line;
                }
                else buffer = buffer.Length > 0 ? buffer + " " + line : line;
            }
            if (buffer.Length > 0) pieces.Add(buffer);
        }
        return string.Join(' ', pieces.Where(p => !Disclaimer().Hit(p)));
    }

    /// <summary>Construct a clean state dictionary for email classification.</summary>
    /// <param name="subject">The email subject. Whitespace-trimmed regardless of <paramref name="clean"/>.</param>
    /// <param name="body">The email body.</param>
    /// <param name="sender">Optional sender, stored under the key <c>"from"</c> when non-empty.</param>
    /// <param name="clean">
    /// Whether <paramref name="body"/> is run through <see cref="CleanBody"/>. When
    /// <see langword="false"/>, the raw body is kept as-is.
    /// </param>
    /// <param name="extra">
    /// Additional fields to merge in, in the given order; a <see langword="null"/> value is
    /// dropped entirely rather than stored, and a key that collides with <c>subject</c>/<c>body</c>/
    /// <c>from</c> overwrites it in place, matching Python's <c>**extra</c> plus <c>dict.update</c>.
    /// </param>
    public static IReadOnlyDictionary<string, object?> State(
        string? subject, string? body, string? sender = null, bool clean = true,
        IEnumerable<KeyValuePair<string, object?>>? extra = null)
        => State(subject, body, sender, clean, extra, 3000);

    /// <summary>
    /// <see cref="State(string?, string?, string?, bool, IEnumerable{KeyValuePair{string, object?}}?)"/>
    /// with Python's <c>max_chars</c> budget for <see cref="CleanBody"/> (<c>email_state(..., max_chars=...)</c>).
    /// </summary>
    /// <param name="subject">The email subject.</param>
    /// <param name="body">The email body.</param>
    /// <param name="sender">Optional sender, stored under the key <c>"from"</c> when non-empty.</param>
    /// <param name="clean">Whether <paramref name="body"/> is cleaned; when <see langword="false"/> it is kept whole.</param>
    /// <param name="extra">Additional fields; see the shorter overload.</param>
    /// <param name="maxChars">
    /// The slice budget <see cref="CleanBody"/> cuts the body to. It is consumed here, never added to
    /// the state as a field, and ignored when <paramref name="clean"/> is <see langword="false"/>.
    /// Zero gives an empty body and a negative value cuts from the end; pass a budget larger than the
    /// body, never <c>0</c> or a negative, to mean "no limit".
    /// </param>
    public static IReadOnlyDictionary<string, object?> State(
        string? subject, string? body, string? sender, bool clean,
        IEnumerable<KeyValuePair<string, object?>>? extra, int maxChars)
    {
        var state = new OrderedDictionary<string, object?>(StringComparer.Ordinal)
        {
            ["subject"] = PythonStr.Strip(subject ?? string.Empty),
            ["body"] = clean ? CleanBody(body, maxChars) : body ?? string.Empty,
        };
        if (!string.IsNullOrEmpty(sender)) state["from"] = sender;
        if (extra is not null)
            foreach (var (key, value) in extra)
                if (value is not null) state[key] = value;
        return state;
    }

    /// <summary>Python's <c>s[:n]</c> for any integer <c>n</c>, measured in code points: a negative <c>n</c> drops <c>|n|</c> from the end.</summary>
    private static string SliceTo(string s, long n)
    {
        if (n >= 0) return LanguageDetection.SliceCodePoints(s, (int)Math.Min(n, int.MaxValue));
        var keep = CodePointLength(s) + n;
        return keep <= 0 ? string.Empty : LanguageDetection.SliceCodePoints(s, (int)keep);
    }

    private static int CodePointLength(string s)
    {
        var count = 0;
        foreach (var _ in s.EnumerateRunes()) count++;
        return count;
    }
}

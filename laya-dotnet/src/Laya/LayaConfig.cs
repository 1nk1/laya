namespace Laya;

/// <summary>
/// The contents of a checkpoint's <c>rl_agent_config.json</c>.
/// </summary>
/// <remarks>
/// Defaults match Python's <c>cfg.get(...)</c> fallbacks in <c>Agent.system_one</c>, not the
/// shipped multilingual values, so a config missing a key behaves the same in both runtimes.
/// </remarks>
public sealed class LayaConfig
{
    private LayaConfig(string? encoder, int maxLen, int headMaxLen,
                       IReadOnlyList<object?> temperatureValues,
                       IReadOnlyDictionary<string, object?> temperatureByOptionsValues,
                       DeferredLengthError? deferredLength)
    {
        DeferredLength = deferredLength;
        TemperatureRawValues = temperatureValues;
        TemperatureByOptionsRawValues = temperatureByOptionsValues;
        var temperatureRaw = temperatureValues.Select(AsDouble).ToList();
        var temperatureByOptionsRaw = temperatureByOptionsValues
            .ToDictionary(kv => kv.Key, kv => AsDouble(kv.Value), StringComparer.Ordinal);
        Encoder = encoder;
        MaxLen = maxLen;
        HeadMaxLen = headMaxLen;
        TemperatureRaw = temperatureRaw;
        TemperatureByOptionsRaw = temperatureByOptionsRaw;

        // clamp_temperature on every entry, with the JSON value as Python holds it (onnx_agent.py:249-252).
        Temperature = temperatureValues.Select(Calibration.ClampTemperature).ToList();
        TemperatureByOptions = temperatureByOptionsValues
            .ToDictionary(kv => kv.Key, kv => Calibration.ClampTemperature(kv.Value), StringComparer.Ordinal);

        // onnx_agent.py:254-262: "<name>=<repr of the raw value> -> <applied, %g>", by-options entries first.
        var rejected = new List<string>();
        foreach (var kv in temperatureByOptionsValues)
            if (Calibration.TemperatureRejected(kv.Value, TemperatureByOptions[kv.Key]))
                rejected.Add($"{kv.Key}={PythonFloat.Repr(kv.Value)} -> {PythonFloat.FormatG(TemperatureByOptions[kv.Key])}");
        for (var i = 0; i < temperatureValues.Count; i++)
            if (Calibration.TemperatureRejected(temperatureValues[i], Temperature[i]))
                rejected.Add($"temperature[{i}]={PythonFloat.Repr(temperatureValues[i])} -> {PythonFloat.FormatG(Temperature[i])}");
        ClampedTemperatures = rejected;

        // Mirrors Python's RuntimeWarning, one warning listing every rejected entry, shown once per process
        // for the same text (WarnOnce); it goes through System.Diagnostics.Trace so a caller can suppress
        // it by removing trace listeners.
        if (rejected.Count > 0)
            WarnOnce.Warn(
                $"laya ONNX: this checkpoint ships temperatures outside [{PythonFloat.FormatG(Calibration.TempMin)}, "
                + $"{PythonFloat.FormatG(Calibration.TempMax)}] which would distort "
                + $"confidence; clamping {string.Join(", ", rejected)}. Treat confidence from the affected buckets as uncalibrated.");
    }

    // The numeric view of a raw JSON value kept for TemperatureRaw: a number as itself, a numeric string
    // as float() reads it, anything else NaN.
    private static double AsDouble(object? value) => PythonFloat.TryConvert(value, out var d) && value is not bool ? d : double.NaN;

    /// <summary>The encoder the checkpoint was trained on, for information only.</summary>
    public string? Encoder { get; }

    /// <summary>Maximum total sequence length.</summary>
    public int MaxLen { get; }

    /// <summary>Token budget for the question head: instructions plus all option fragments.</summary>
    public int HeadMaxLen { get; }

    /// <summary>
    /// Per-question-type temperature, clamped to the usable range: the entries of the config's <c>temperature</c>, three
    /// for a shipped checkpoint. A shorter list is kept as it is, and a prediction that needs the missing entry fails as
    /// Python's does (<c>IndexError: list index out of range</c>).
    /// </summary>
    public IReadOnlyList<double> Temperature { get; }

    /// <summary>Per <c>(type, option-count)</c> bucket temperature, clamped. Keys as <see cref="Calibration.TempBucket"/> builds them.</summary>
    public IReadOnlyDictionary<string, double> TemperatureByOptions { get; }

    /// <summary>
    /// What the checkpoint shipped, before clamping, as numbers: an entry that is not a number
    /// (and not a numeric string <c>float()</c> accepts) reads <see cref="double.NaN"/>. See
    /// <see cref="TemperatureRawValues"/> for the values themselves.
    /// </summary>
    public IReadOnlyList<double> TemperatureRaw { get; }

    /// <summary>
    /// The <c>temperature</c> entries exactly as the JSON held them: a <see cref="long"/> or
    /// <see cref="System.Numerics.BigInteger"/> for an integer, a <see cref="double"/>, a
    /// <see cref="string"/>, a <see cref="bool"/>, <see langword="null"/>, or a list / key-value list.
    /// </summary>
    public IReadOnlyList<object?> TemperatureRawValues { get; }

    /// <summary>The <c>temperature_by_options</c> entries exactly as the JSON held them; see <see cref="TemperatureRawValues"/>.</summary>
    public IReadOnlyDictionary<string, object?> TemperatureByOptionsRawValues { get; }

    /// <summary>What the checkpoint shipped, before clamping.</summary>
    public IReadOnlyDictionary<string, double> TemperatureByOptionsRaw { get; }

    /// <summary>
    /// Descriptions of every temperature that had to be clamped, empty when none were. A
    /// non-empty list means confidence from those buckets is uncalibrated and worth surfacing.
    /// Each entry also triggers a <c>System.Diagnostics.Trace.TraceWarning</c> at load
    /// time, mirroring Python's <c>RuntimeWarning</c> for the same condition.
    /// </summary>
    public IReadOnlyList<string> ClampedTemperatures { get; }

    /// <summary>Read a config from a <c>rl_agent_config.json</c> file.</summary>
    public static LayaConfig Load(string path) => Parse(PythonJsonReader.ReadAllText(path));

    /// <summary>Parse a config from JSON text, read as Python's <c>json.loads</c> reads it.</summary>
    /// <exception cref="InvalidDataException">
    /// Where Python's loader raises: a config that is not an object, an unsupported <c>option_layout</c>, a
    /// <c>temperature</c> that is not iterable, a <c>temperature_by_options</c> that is not a mapping, or an
    /// integer temperature too large for a float. The message carries Python's exception type and text.
    /// </exception>
    public static LayaConfig Parse(string json)
    {
        var parsed = PythonJsonReader.Parse(json);
        if (parsed is not IReadOnlyDictionary<string, object?> root)
            throw new InvalidDataException($"AttributeError: '{PythonFloat.TypeName(parsed)}' object has no attribute 'get'");

        // onnx_agent.py:193: only the sequential layout runs on the exported graph.
        if (root.TryGetValue("option_layout", out var layout) && layout is not "sequential")
            throw new InvalidDataException(
                $"ValueError: option_layout={PythonFloat.Repr(layout)} is not supported by the ONNX runtime; load this checkpoint with laya.Agent");

        // `cfg.get("temperature", [1.0, 1.0, 1.0])`, then `[clamp_temperature(t) for t in temperature_raw]`: whatever
        // the key holds is iterated, so a string yields its characters and a dict its keys, while a number, bool or
        // null raises TypeError (onnx_agent.py:247-250).
        var temperature = new List<object?>();
        if (root.TryGetValue("temperature", out var temp)) temperature.AddRange(Iterate(temp));
        else temperature.AddRange([1.0, 1.0, 1.0]);
        // A list shorter than three is kept as it is: Python indexes self.temperature[qtype] when a question is
        // answered and raises IndexError there, which Calibration.ResolveTemperature reproduces.
        ThrowIfOverflow(temperature);

        // `{k: clamp_temperature(v) for k, v in temperature_by_options_raw.items()}`: a mapping or AttributeError.
        var byOptions = new Dictionary<string, object?>(StringComparer.Ordinal);
        if (root.TryGetValue("temperature_by_options", out var tbo))
        {
            if (tbo is not IEnumerable<KeyValuePair<string, object?>> map)
                throw new InvalidDataException($"AttributeError: '{PythonFloat.TypeName(tbo)}' object has no attribute 'items'");
            foreach (var (key, value) in map)
                byOptions[key] = value;
        }
        ThrowIfOverflow(byOptions.Values);

        return new LayaConfig(
            root.TryGetValue("encoder", out var enc) && enc is string encoder ? encoder : null,
            Length(root, "max_len", 512),
            Length(root, "head_max_len", 192),
            temperature,
            byOptions,
            DeferredLengthError.For(root.TryGetValue("max_len", out var maxRaw) ? maxRaw : 512L,
                                    root.TryGetValue("head_max_len", out var headRaw) ? headRaw : 192L));
    }

    // `for t in raw`: a list's items, a string's characters (code points), a dict's keys.
    private static IEnumerable<object?> Iterate(object? raw)
    {
        switch (raw)
        {
            case List<object?> list: return list;
            case string s: return CodePointStrings(s);
            case IEnumerable<KeyValuePair<string, object?>> map: return map.Select(kv => (object?)kv.Key).ToList();
            default: throw new InvalidDataException($"TypeError: '{PythonFloat.TypeName(raw)}' object is not iterable");
        }
    }

    private static List<object?> CodePointStrings(string s)
    {
        var items = new List<object?>();
        for (var i = 0; i < s.Length; i++)
        {
            if (char.IsHighSurrogate(s[i]) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1])) { items.Add(s.Substring(i, 2)); i++; }
            else items.Add(s[i].ToString());
        }
        return items;
    }

    // clamp_temperature(<int too large for a float>) raises OverflowError, which nothing catches, so Python fails
    // to load the checkpoint (onnx_agent.py:249-252); a float literal reads as inf instead.
    private static void ThrowIfOverflow(IEnumerable<object?> values)
    {
        if (values.Any(v => v is System.Numerics.BigInteger big && double.IsInfinity((double)big)))
            throw new InvalidDataException("OverflowError: int too large to convert to float");
    }

    // `cfg.get(key, fallback)` for max_len / head_max_len. Python keeps whatever the JSON held and only fails when a
    // prediction does arithmetic or slicing with it, so a wrong type is recorded (kind) and raised then, not here;
    // `true` / `false` are the ints 1 / 0. A negative integer slices from the end in Python, which this port does not
    // support, so it is refused at load.
    private static int Length(IReadOnlyDictionary<string, object?> root, string key, int fallback)
    {
        if (!root.TryGetValue(key, out var v)) return fallback;
        switch (v)
        {
            case bool b: return b ? 1 : 0;
            case long l when l < 0: throw NegativeLength(key, l.ToString(System.Globalization.CultureInfo.InvariantCulture));
            case long l: return (int)Math.Min(l, int.MaxValue);
            case System.Numerics.BigInteger big when big.Sign < 0: throw NegativeLength(key, big.ToString(System.Globalization.CultureInfo.InvariantCulture));
            case System.Numerics.BigInteger: return int.MaxValue;
            default: return fallback;   // float, str, null, list, dict: DeferredLengthError raises at prediction
        }
    }

    private static InvalidDataException NegativeLength(string key, string literal) =>
        new($"{key}={literal} is negative: Python slices from the end for it, which this port does not support");

    /// <summary>
    /// A <c>max_len</c> / <c>head_max_len</c> of the wrong JSON type: loading succeeds and the first prediction raises
    /// Python's <c>TypeError</c>.
    /// </summary>
    internal sealed record DeferredLengthError(string WithQuestions, string? NoQuestions, double? NoQuestionsWindow)
    {
        internal static DeferredLengthError? For(object? maxLen, object? headMaxLen)
        {
            var left = PythonFloat.TypeName(maxLen);
            var right = PythonFloat.TypeName(headMaxLen);
            var leftBad = left is not ("int" or "bool");
            var rightBad = right is not ("int" or "bool");
            if (!leftBad && !rightBad) return null;
            // build_sequence meets head_max_len first (build_head), then max_len. A non-numeric head fails the subtraction
            // at once; a float head normally fails the slice of the head tokens (which a very small float survives: not
            // ported); then a max_len that is not an int fails its own subtraction or, for a float, the final slice.
            const string SliceError = "TypeError: slice indices must be integers or None or have an __index__ method";
            string Operand(string kind) => $"TypeError: unsupported operand type(s) for -: '{kind}' and 'int'";
            var withQuestions = rightBad && right != "float" ? Operand(right)
                : leftBad ? (left == "float" ? SliceError : Operand(left))
                : SliceError;
            // Without questions window_budget starts with `max_len - head_max_len`; a float on its own survives that.
            static bool Numeric(string k) => k is "int" or "bool" or "float";
            if (!(Numeric(left) && Numeric(right)))
                return new DeferredLengthError(withQuestions, $"TypeError: unsupported operand type(s) for -: '{left}' and '{right}'", null);
            // Only floats are wrong: window_budget runs, with `max(64, max_len - head_max_len - 8)` as a float window,
            // and a state longer than that window fails later with the slice TypeError.
            static double Number(object? v) => v switch { bool b => b ? 1 : 0, long l => l, System.Numerics.BigInteger big => (double)big, double d => d, _ => 0 };
            return new DeferredLengthError(withQuestions, null, Math.Max(64.0, Number(maxLen) - Number(headMaxLen) - 8));
        }
    }

    /// <summary>The wrong-typed length setting's <c>TypeError</c>, raised by a prediction as Python does.</summary>
    internal void ThrowIfLengthDeferred(bool withQuestions)
    {
        if (DeferredLength is not { } deferred) return;
        var message = withQuestions ? deferred.WithQuestions : deferred.NoQuestions;
        if (message is not null) throw new InvalidOperationException(message);
    }

    /// <summary>
    /// For a prediction without questions under float-only wrong lengths: a state of more tokens than the float window
    /// Python scans with fails the slice <c>TypeError</c>, a shorter one reads in one pass.
    /// </summary>
    internal void ThrowIfStateExceedsDeferredWindow(int stateTokens)
    {
        if (DeferredLength is { NoQuestionsWindow: { } window } deferred && stateTokens > window)
            throw new InvalidOperationException(deferred.WithQuestions);
    }

    internal DeferredLengthError? DeferredLength { get; }
}

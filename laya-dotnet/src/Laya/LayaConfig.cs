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
                       IReadOnlyDictionary<string, object?> temperatureByOptionsValues)
    {
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

    /// <summary>Per-question-type temperature, clamped to the usable range. Always 3 entries.</summary>
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
    public static LayaConfig Parse(string json)
    {
        if (PythonJsonReader.Parse(json) is not IReadOnlyDictionary<string, object?> root)
            throw new InvalidDataException("the checkpoint config must be a JSON object");

        // `cfg.get("temperature", [1.0, 1.0, 1.0])`: the list as written, entries as json.loads typed them.
        var temperature = new List<object?>();
        if (root.TryGetValue("temperature", out var temp) && temp is List<object?> list)
            temperature.AddRange(list);
        // Python indexes self.temperature[qtype] for qtype 0..2, so a short list would be an
        // IndexError there; pad to 3 with the neutral value rather than failing later per-question.
        while (temperature.Count < 3) temperature.Add(1.0);

        var byOptions = new Dictionary<string, object?>(StringComparer.Ordinal);
        if (root.TryGetValue("temperature_by_options", out var tbo) && tbo is IEnumerable<KeyValuePair<string, object?>> map)
            foreach (var (key, value) in map)
                byOptions[key] = value;

        // clamp_temperature(<int too large for a float>) raises OverflowError, which nothing catches, so
        // Python fails to load the checkpoint (onnx_agent.py:249-252); a float literal reads as inf instead.
        if (temperature.Concat(byOptions.Values).Any(v => v is System.Numerics.BigInteger big && double.IsInfinity((double)big)))
            throw new InvalidDataException("OverflowError: int too large to convert to float");

        return new LayaConfig(
            root.TryGetValue("encoder", out var enc) && enc is string encoder ? encoder : null,
            IntOr(root, "max_len", 512),
            IntOr(root, "head_max_len", 192),
            temperature,
            byOptions);
    }

    // `cfg.get(key, fallback)` for an integer setting: a JSON number is taken, anything else is the fallback.
    private static int IntOr(IReadOnlyDictionary<string, object?> root, string key, int fallback)
    {
        if (!root.TryGetValue(key, out var v)) return fallback;
        return v switch
        {
            long l when l is >= int.MinValue and <= int.MaxValue => (int)l,
            long or System.Numerics.BigInteger or double => throw new FormatException($"{key} is not a 32-bit integer"),
            _ => fallback,
        };
    }
}

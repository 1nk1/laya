using System.Diagnostics;
using Laya.Tokenization;

namespace Laya;

internal static class LongPrediction
{
    internal static (int Window, int Stride, int Room) Budget(
        ILayaTokenizer tokenizer, QuestionSet questions, int maxLen, int headMaxLen,
        int? window, int? stride)
    {
        var requested = window is > 0 ? window.Value : Math.Max(64, maxLen - headMaxLen - 8);
        var room = questions.Count == 0 ? requested : questions.Ids.Min(
            id => SequenceBuilder.StateRoom(tokenizer, questions[id], maxLen, headMaxLen));
        if (room <= 0)
            throw new ArgumentException(
                $"predict_long: the questions' options fill the whole sequence (max_len={maxLen},"
                + $" head_max_len={headMaxLen}), leaving no room for the state; no window can carry any of it."
                + " A label set this large is what laya.shortlist.predict_shortlist is for", nameof(questions));
        var size = Math.Min(requested, room);
        if (size < requested && (window is > 0 || requested >= room * 2L))
            Trace.TraceWarning(
                $"laya: predict_long: scan window capped {requested} -> {size} to fit these questions"
                + $" inside max_len={maxLen}; fewer or shorter options or shortlisting reduce scan cost");
        var step = stride is > 0 ? stride.Value : Math.Max(1, size / 2);
        if (step > size)
        {
            if (size < requested && stride <= requested)
            {
                step = Math.Max(1, size / 2);
                Trace.TraceWarning($"laya: predict_long: stride={stride} reduced to {step} after window clamp");
            }
            else
                throw new ArgumentException(
                    $"predict_long: stride={step} steps past the {size}-token window"
                    + (size < requested ? " these questions leave room for" : "")
                    + $", so {step - size} tokens between every pair of windows would be read by no window at all;"
                    + $" pass stride <= {size}", nameof(stride));
        }
        return (size, step, room);
    }

    internal static IReadOnlyList<WindowInfo> Spans(int tokens, int window, int stride)
    {
        var starts = new List<int>();
        for (var i = 0; i < tokens;)
        {
            starts.Add(i);
            if ((long)i + window >= tokens) break;
            i += stride;
        }
        return starts.Select((start, index) =>
            new WindowInfo(index, start, (int)Math.Min((long)start + window, tokens), starts.Count)).ToArray();
    }

    internal static int? BatchCap(int count, int window, int configBudget, int? batchSize)
    {
        if (batchSize is > 0) return batchSize;
        if (window >= configBudget) return null;
        var step = Math.Max(1, window / 2);
        var configStep = Math.Max(1, configBudget / 2);
        var unclamped = Math.Max(1, (int)(((long)count * step + configStep - 1) / configStep));
        return count <= unclamped * 2L ? null : unclamped;
    }

    internal static LayaResult Aggregate(
        QuestionSet questions, IReadOnlyList<LayaResult> results, IReadOnlyList<WindowInfo> spans)
    {
        var answers = new Dictionary<string, Answer>(StringComparer.Ordinal);
        foreach (var id in questions.Ids)
        {
            var best = 0;
            double Evidence(int index) => questions[id].Type == QuestionType.Noul
                ? results[index][id].AsNoul().Probability : results[index][id].AnswerConfidence;
            for (var i = 1; i < results.Count; i++)
                if (Evidence(i) > Evidence(best)) best = i;
            answers[id] = results[best][id].WithWindow(spans[best]);
        }
        Dictionary<string, OptionUsage>? options = null;
        foreach (var result in results)
            if (result.Usage.Options is { } records)
            {
                options ??= new(StringComparer.Ordinal);
                foreach (var (id, stats) in records) options[id] = stats;
            }
        var usage = new Usage(results.Sum(r => r.Usage.InputTokens), 0)
        {
            Windows = results.Count,
            StateTokens = results.Any(r => r.Usage.StateTokens.HasValue)
                ? results.Sum(r => r.Usage.StateTokens ?? 0) : null,
            StateTokensDropped = results.Any(r => r.Usage.StateTokensDropped.HasValue)
                ? results.Sum(r => r.Usage.StateTokensDropped ?? 0) : null,
            Truncated = results.Any(r => r.Usage.Truncated.HasValue)
                ? results.Sum(r => r.Usage.Truncated ?? 0) : null,
            TruncatedQuestions = results.Count == 0 ? null : results[^1].Usage.TruncatedQuestions,
            Options = options,
        };
        return new LayaResult("laya-rl-agent", [.. questions.Ids], answers, usage);
    }
}

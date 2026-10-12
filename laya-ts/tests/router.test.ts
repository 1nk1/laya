import { describe, expect, it, vi } from "vitest";
import { Router, normaliseName } from "../src/router.js";
describe("router", () => {
  it("normalises aliases", () => { expect(normaliseName("en")).toBe("english"); });
  it("rejects unknown model", () => { expect(() => normaliseName("jev-1")).toThrow(); });
  it.each(["refund", "12345 ???", { body: "refund" }])("uses multilingual for unidentified state %j", (state) => {
    const r = new Router();
    const decision = r.route(state, {});
    expect(r.default).toBe("multilingual");
    expect(decision.detection?.languageUndecided).toBe(true);
    expect(decision.model).toBe("multilingual");
    expect(decision.reason).toContain("using default (multilingual)");
    expect(r.loaded).toEqual([]);
  });
  it.each(["english", "multilingual"])("applies default %s only to unidentified text", (fallback) => {
    const r = new Router({ default: fallback });
    expect(r.route("refund", {}).model).toBe(fallback);
    expect(r.route("Please refund the duplicate charge on the invoice", {}).model).toBe("english");
    expect(r.route("मुझसे दो बार शुल्क लिया गया", {}).model).toBe("multilingual");
    expect(r.route("refund", {}, { lang: "en" }).model).toBe("english");
    expect(r.route("refund", {}, { model: "typed-decisions" }).model).toBe("typed-decisions");
  });
  it("uses the default in a mixed routing batch", () => {
    const r = new Router();
    const decisions = r.routeBatch([
      { state: "refund", questions: {} },
      { state: "Please refund the duplicate charge on the invoice", questions: {} },
    ]);
    expect(decisions.map((d) => d.model)).toEqual(["multilingual", "english"]);
    expect(r.loaded).toEqual([]);
  });
  it("loads the multilingual checkpoint when predicting unidentified text", async () => {
    const loader = vi.fn(async () => ({
      async systemOne() {
        return { model: "laya-rl-agent", answers: {}, usage: { input_tokens: 5, output_tokens: 0 } };
      },
    }));
    const r = new Router({ loader });
    const result = await r.predict("refund", { q: { type: "noul", instructions: "?" } });
    expect(loader).toHaveBeenCalledWith("multilingual", r.models.multilingual);
    expect(result.routing.model).toBe("multilingual");
    expect(r.loaded).toEqual(["multilingual"]);
  });
  it("routes hindi script to multilingual without loading", () => {
    const r = new Router();
    expect(r.route({ body: "मुझसे दो बार शुल्क लिया गया" }, {}).model).toBe("multilingual");
    expect(r.loaded).toEqual([]);
  });
  it.each([
    ["Please send me the café menu today please", "english"],
    ["Send the invoice to José before Friday", "english"],
    ["Grüße aus Köln, wir melden uns wegen der Rechnung", "multilingual"],
    ["sluk lyset i soveværelset", "multilingual"],
  ])("routes %s to %s", (text, model) => {
    expect(new Router().route(text, {}).model).toBe(model);
  });
  it("explicit model wins", () => {
    expect(new Router().route("hi", {}, { model: "typed-decisions" }).model).toBe("typed-decisions");
  });
  it("lang_guess callable routes", () => {
    const r = new Router();
    expect(r.route("text", {}, { langGuess: () => "ro" }).model).toBe("multilingual");
  });
  it("shares concurrent loads and emits lifecycle hooks once per cached model", async () => {
    const events: string[] = [];
    const loader = vi.fn(async (name: string) => ({ name }));
    const r = new Router({
      maxLoaded: 1,
      loader,
      hooks: [{
        onLoad: (ctx) => events.push(`load:${ctx.model}`),
        onEvict: (ctx) => events.push(`evict:${ctx.model}`),
      }],
    });

    const [first, second] = await Promise.all([r.load("en"), r.load("english")]);
    expect(first).toBe(second);
    expect(loader).toHaveBeenCalledTimes(1);
    expect(r.loaded).toEqual(["english"]);
    expect(r._agents.size).toBe(1);
    await r.load("multilingual");
    expect(events).toEqual(["load:english", "evict:english", "load:multilingual"]);
    expect(r.loaded).toEqual(["multilingual"]);
  });
  it("clears an unsuccessful in-flight load so the model can be retried", async () => {
    const agent = { name: "english" };
    const loader = vi.fn()
      .mockRejectedValueOnce(new Error("load failed"))
      .mockResolvedValue(agent);
    const r = new Router({ loader });

    const results = await Promise.allSettled([r.load("en"), r.load("english")]);
    expect(results.map((result) => result.status)).toEqual(["rejected", "rejected"]);
    expect(loader).toHaveBeenCalledTimes(1);
    expect(await r.load("english")).toBe(agent);
    expect(loader).toHaveBeenCalledTimes(2);
    expect(r.loaded).toEqual(["english"]);
  });
});

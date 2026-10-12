import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { cleanEmailBody, emailState } from "../src/email.js";

// `laya-java/fixtures/email.json` is the Python reference recording what its own cleaner does: 95
// bodies, 13 budget cuts and 7 states, each with the exact result. The Java port is tested against
// it; the TypeScript port re-implements the same markers and had no test that read it, so the two
// places where JS's primitives differ from Python's -- `\s`/`\d`/`\w`/`\b` are ASCII-ish where
// Python's are Unicode-wide, and `.length`/`.slice` count UTF-16 units where Python counts code
// points -- were invisible here. This file runs the recorded oracle as-is: no expectation is
// restated in TypeScript, so a future divergence fails by case name.
const fixture = JSON.parse(
  readFileSync(new URL("../../laya-java/fixtures/email.json", import.meta.url), "utf8"),
) as {
  cleaned: { name: string; body: string; result: string }[];
  budgets: { name: string; body: string; max_chars: number; result: string }[];
  states: {
    name: string;
    subject: string;
    body: string;
    sender: string | null;
    clean: boolean;
    max_chars: number;
    extra: Record<string, unknown>;
    result: Record<string, unknown>;
  }[];
};

describe("email.json cleaned cases", () => {
  it.each(fixture.cleaned.map((c) => [c.name, c.body, c.result] as const))(
    "%s",
    (_name, body, want) => {
      expect(cleanEmailBody(body)).toBe(want);
    },
  );
});

describe("email.json budget cases", () => {
  it.each(fixture.budgets.map((c) => [c.name, c.body, c.max_chars, c.result] as const))(
    "%s",
    (_name, body, maxChars, want) => {
      expect(cleanEmailBody(body, maxChars)).toBe(want);
    },
  );
});

describe("email.json state cases", () => {
  it.each(fixture.states.map((c) => [c.name, c] as const))("%s", (_name, c) => {
    expect(emailState(c.subject, c.body, c.sender, c.clean, c.extra ?? {}, c.max_chars)).toEqual(
      c.result,
    );
  });
});

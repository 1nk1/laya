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

// Two of the primitives in `src/email.ts` are invisible to the corpus above: no recorded body or
// subject holds U+FEFF, and no recorded character sits next to an `@` where the difference between
// the two whitespace classes decides a match. Python's `\s` and `str.strip()` are its 29 whitespace
// code points, which hold U+001C-U+001F and U+0085 and not U+FEFF, while JS's hold U+FEFF and not
// U+001C-U+001F. So each answer below is what `laya/email.py` printed for the same input
// (`clean_email_body`, and `email_state(...)["subject"]` for the two subjects), written out here
// instead of added to `email.json` because that corpus is shared with the Java port, whose
// `EmailTest` asserts its exact case counts.
describe("Python's whitespace set, where JS's differs", () => {
  const BOM_CONST = "\uFEFF";
  const FILE_SEP = "\u001C";
  const UNIT_SEP = "\u001F";

  // A paragraph's trailing whitespace is removed by `trimPy`, which reads Python's class, and
  // U+FEFF is not in it. `String.prototype.trim` does read U+FEFF as whitespace, and would delete a
  // character from the text handed to the model.
  it("keeps a U+FEFF that ends a paragraph", () => {
    const body = "I cannot log in" + BOM_CONST + "\n\nPlease reset my password.";
    expect(cleanEmailBody(body)).toBe(body);
  });

  it("keeps a U+FEFF that ends a subject", () => {
    expect(emailState("Refund" + BOM_CONST, "body").subject).toBe("Refund" + BOM_CONST);
  });

  // The other direction of the same difference: JS's `trim` leaves U+001C in the text.
  it("strips a U+001C that ends a line", () => {
    expect(cleanEmailBody("Trace the refund." + FILE_SEP + "\n\nPlease reset it.")).toBe(
      "Trace the refund.\n\nPlease reset it.",
    );
  });

  // U+001F is whitespace to Python, so the `\S` before the `@` in the attribution tail does not
  // match it and the line stays ordinary text; JS's `\S` would match, cut the body there, and drop
  // the sender's request. The next case is that same line with an ordinary letter, where the tail
  // does fire.
  it("does not read a U+001F before an address as an attribution tail", () => {
    const body = "Trace the refund.\n\nBruno " + UNIT_SEP + "@acme.com wrote:";
    expect(cleanEmailBody(body)).toBe(body);
  });

  it("reads an ordinary attribution tail and cuts it", () => {
    expect(cleanEmailBody("Trace the refund.\n\nBruno Xavier@acme.com wrote:")).toBe(
      "Trace the refund.",
    );
  });
});

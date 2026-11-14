import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import {
  SYMBOL_PATH,
  SYMBOL_UNIT_PATH,
  WORDMARK_LEG_PATH,
  WORDMARK_PATH,
  WORDMARK_UNIT_PATH,
} from "@/components/logo";

const brand = path.resolve(__dirname, "../../brand");
const read = (p: string) => readFileSync(p, "utf8");

describe("brand single source of truth", () => {
  it("web tokens are identical to brand/tokens.css", () => {
    expect(read(path.resolve(__dirname, "../src/styles/tokens.css"))).toBe(read(path.join(brand, "tokens.css")));
  });

  it.each(["white", "black", "mono-white", "mono-black"])("wordmark-%s geometry matches the site", (v) => {
    const svg = read(path.join(brand, `logo/thor-wordmark-${v}.svg`));
    expect(svg).toContain(`d="${WORDMARK_PATH}"`);
    expect(svg).toContain(`d="${WORDMARK_LEG_PATH}"`);
    expect(svg).toContain(`d="${WORDMARK_UNIT_PATH}"`);
  });

  it.each(["thor-symbol-white.svg", "thor-symbol-black.svg", "thor-app-icon.svg", "../../web/src/app/icon.svg"])(
    "%s symbol geometry matches the site",
    (f) => {
      const svg = read(path.join(brand, "logo", f));
      expect(svg).toContain(`d="${SYMBOL_PATH}"`);
      expect(svg).toContain(`d="${SYMBOL_UNIT_PATH}"`);
    },
  );
});

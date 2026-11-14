import { execFileSync } from "node:child_process";
import { readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { models } from "@/content/models";
import { pipeline, products } from "@/content/products";
import { research } from "@/content/research";
import { STATUSES } from "@/content/status";

const repoRoot = path.resolve(__dirname, "../..");

describe("model specifications", () => {
  it("match models/thor/config.py exactly", () => {
    // Load config.py directly (it has no third-party imports), so this needs only Python, not PyTorch.
    const script = `
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("cfg", sys.argv[1])
cfg = importlib.util.module_from_spec(spec); sys.modules["cfg"] = cfg; spec.loader.exec_module(cfg)
print(json.dumps({n: dict(c.to_dict(), params=c.num_params(), non_embedding=c.num_params(include_embedding=False))
                  for n, c in cfg.PRESETS.items()}))`;
    const presets = JSON.parse(
      execFileSync("python3", ["-c", script, path.join(repoRoot, "models/thor/config.py")], { encoding: "utf8" }),
    );
    const specified = models.filter((m) => m.spec);
    expect(specified.length).toBeGreaterThan(0);
    for (const { name, spec } of specified) {
      const p = presets[spec!.preset];
      expect(p, `${name}: unknown preset ${spec!.preset}`).toBeDefined();
      expect(spec).toEqual({
        preset: p.name,
        params: p.params,
        nonEmbeddingParams: p.non_embedding,
        layers: p.n_layer,
        dModel: p.d_model,
        heads: p.n_head,
        ffnHidden: p.ffn_hidden,
        context: p.max_seq_len,
        vocab: p.vocab_size,
      });
    }
  });
});

describe("content integrity", () => {
  it("uses only known statuses and unique slugs", () => {
    for (const list of [models, products, research]) {
      const slugs = list.map((x) => x.slug);
      expect(new Set(slugs).size).toBe(slugs.length);
      for (const item of list) expect(STATUSES).toContain(item.status);
    }
  });

  it("never marks a model trained or released without published evaluation", () => {
    for (const m of models) {
      if (m.status === "trained-internal" || m.status === "released") {
        expect(m.evaluation, m.name).not.toMatch(/no results/i);
        expect(research.some((r) => r.date && r.title.includes(m.name)), `${m.name} needs a published report`).toBe(true);
      }
    }
  });

  it("publishes research only with a date, and only planned work without one", () => {
    for (const r of research) {
      if (r.date) expect(r.date).toMatch(/^\d{4}-\d{2}-\d{2}$/);
      else expect(r.status).toBe("planned");
    }
  });

  it("reports pipeline stages as built only when the model package has tests", () => {
    const tests = readdirSync(path.join(repoRoot, "models/tests"));
    expect(tests.length).toBeGreaterThan(0);
    expect(pipeline.filter((s) => s.status === "working").length).toBeGreaterThan(0);
  });
});

describe("claims guard", () => {
  // Phrases THOR's brand rules forbid: superlatives, frontier comparisons, invented traction.
  const banned = [
    /state[- ]of[- ]the[- ]art/i,
    /\bbest[- ]in[- ]class\b/i,
    /\boutperform/i,
    /\bworld[- ]class\b/i,
    /\brevolutionary\b/i,
    /\bGPT-\d/i,
    /\bClaude\b/,
    /\bGemini\b/,
    /\bLlama\b/,
    /\btrusted by\b/i,
    /\bour (customers|partners|investors)\b/i,
    /\bbacked by\b/i,
    /\bAGI\b/,
  ];

  function files(dir: string): string[] {
    return readdirSync(dir).flatMap((f) => {
      const p = path.join(dir, f);
      return statSync(p).isDirectory() ? files(p) : /\.(tsx?|css)$/.test(f) ? [p] : [];
    });
  }

  it("site source contains no banned claims", () => {
    for (const file of files(path.resolve(__dirname, "../src"))) {
      const text = readFileSync(file, "utf8");
      for (const re of banned) expect(text, `${path.relative(repoRoot, file)} matches ${re}`).not.toMatch(re);
    }
  });
});

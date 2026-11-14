import type { Metadata } from "next";
import { StatusBadge } from "@/components/status-badge";
import { PageHeader, Section } from "@/components/ui";

export const metadata: Metadata = {
  title: "Developers",
  description: "THOR for developers: the THOR toolchain today, and the planned API.",
};

const toolchain = `# Inspect model specifications
python -m thor.specs

# Prepare a corpus: trains the THOR tokenizer and tokenizes the text
python -m thor.prepare --input corpus.txt --out data/corpus --vocab-size 16384

# Train (re-run the same command to resume from the last checkpoint)
python -m thor.train --preset thor-30m --data-dir data/corpus --out-dir runs/thor-30m

# Evaluate and sample
python -m thor.evaluate --ckpt runs/thor-30m/model_best.pt --data-dir data/corpus
python -m thor.sample --ckpt runs/thor-30m/model_best.pt \\
  --tokenizer data/corpus/tokenizer.json --prompt "Once upon a time"`;

const planned = [
  { name: "THOR API", detail: "HTTP API for text generation with THOR models, served by THOR Cloud." },
  { name: "Model downloads", detail: "Weights for released THOR models, with model cards and licenses." },
  { name: "Integration guides", detail: "Examples for using THOR models from Python and from the API." },
];

export default function DevelopersPage() {
  return (
    <>
      <PageHeader
        eyebrow="Developers"
        title="Build with THOR"
        lede={
          <p>
            There is no public THOR API or downloadable model yet. This page will host the API reference and
            integration guides once a THOR model is released.
          </p>
        }
      />
      <Section
        label="Today"
        title="The THOR toolchain"
        intro="The Python package THOR uses internally to build its models. Source access is currently private."
      >
        <pre className="overflow-x-auto rounded-lg border border-graphite-600 bg-graphite-900 p-5 font-mono text-[13px] leading-relaxed text-silver">
          <code>{toolchain}</code>
        </pre>
      </Section>
      <Section label="Roadmap" title="Coming for developers">
        <ul className="divide-y divide-graphite-600 border-y border-graphite-600">
          {planned.map((p) => (
            <li key={p.name} className="flex flex-col gap-3 py-5 sm:flex-row sm:items-center sm:justify-between sm:gap-8">
              <div>
                <h3 className="font-medium text-white">{p.name}</h3>
                <p className="mt-1 text-[15px] text-silver">{p.detail}</p>
              </div>
              <StatusBadge status="planned" />
            </li>
          ))}
        </ul>
      </Section>
    </>
  );
}

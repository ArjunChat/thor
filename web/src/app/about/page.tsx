import type { Metadata } from "next";
import { PageHeader, Section } from "@/components/ui";

export const metadata: Metadata = {
  title: "About",
  description: "THOR's mission, philosophy and long-term direction.",
};

const roadmap = [
  { phase: "01", title: "Foundations", detail: "Tokenizer, architecture, training and evaluation pipeline.", state: "Built & tested" },
  { phase: "02", title: "First model", detail: "Train and evaluate THOR-30M; publish the training report.", state: "Next" },
  { phase: "03", title: "Main model", detail: "Train THOR-100M on broader text with documented evaluation.", state: "Planned" },
  { phase: "04", title: "Instruction tuning and Chat", detail: "THOR-Instruct, and THOR Chat running on it.", state: "Planned" },
  { phase: "05", title: "Platform", detail: "THOR Cloud inference, the API and THOR Studio.", state: "Planned" },
];

export default function AboutPage() {
  return (
    <>
      <PageHeader
        eyebrow="About THOR"
        title="An independent AI company"
        lede={
          <p>
            THOR exists to engineer intelligent systems from first principles: to understand every layer of a language
            model by building it, and to turn that into AI products people can use.
          </p>
        }
      />
      <Section label="Mission" title="Why THOR builds its own models">
        <div className="max-w-2xl space-y-5 text-[16px] leading-[1.75] text-silver">
          <p>
            Most AI products today resell access to someone else&apos;s model. THOR takes the longer road: its own
            tokenizer, its own architecture, its own training pipeline and its own weights.
          </p>
          <p>
            Owning the stack means THOR can explain how its models work, measure them honestly, and decide for itself
            how they are built and deployed.
          </p>
        </div>
      </Section>
      <Section label="Philosophy" title="How we work">
        <ul className="grid gap-x-10 gap-y-8 sm:grid-cols-2">
          {[
            ["Precision over hype", "We publish specifications and measured results, not adjectives."],
            ["Honest status", "Nothing is called trained, available or released until it is."],
            ["No invented facts", "No made-up partners, customers, funding, benchmarks or statistics. Ever."],
            ["Responsible engineering", "Open-source licenses respected, data sources recorded, credentials never exposed."],
          ].map(([t, b]) => (
            <li key={t} className="border-t border-graphite-600 pt-5">
              <h3 className="font-medium text-white">{t}</h3>
              <p className="mt-2 text-[15px] leading-relaxed text-silver">{b}</p>
            </li>
          ))}
        </ul>
      </Section>
      <Section label="Direction" title="The long-term path" intro="In order. No dates are promised until they can be kept.">
        <ol className="divide-y divide-graphite-600 border-y border-graphite-600">
          {roadmap.map((r) => (
            <li key={r.phase} className="grid gap-2 py-5 sm:grid-cols-[3rem_1fr_9rem] sm:items-baseline sm:gap-6">
              <span className="font-mono text-sm text-steel">{r.phase}</span>
              <div>
                <h3 className="font-medium text-white">{r.title}</h3>
                <p className="mt-1 text-[15px] text-silver">{r.detail}</p>
              </div>
              <span className={`font-mono text-xs tracking-[0.06em] uppercase ${r.state === "Planned" ? "text-steel" : "text-signal"}`}>
                {r.state}
              </span>
            </li>
          ))}
        </ol>
        <p className="mt-8 max-w-2xl text-[15px] leading-relaxed text-silver">
          Beyond language models, THOR intends to expand into AI agents, research tools, developer infrastructure and
          enterprise products, all built on the same foundation.
        </p>
      </Section>
    </>
  );
}

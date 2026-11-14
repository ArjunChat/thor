import Link from "next/link";
import { ModelCard } from "@/components/model-card";
import { ComponentBadge, StatusBadge } from "@/components/status-badge";
import { TokenField } from "@/components/token-field";
import { ButtonLink, Section } from "@/components/ui";
import { formatParams, getModel, models } from "@/content/models";
import { pipeline, products } from "@/content/products";
import { formatDate, site } from "@/content/site";

const principles = [
  {
    title: "Our own models",
    body: "Every THOR model is trained by THOR, from the tokenizer to the weights. External model output is never presented as THOR's.",
  },
  {
    title: "Measured, not claimed",
    body: "Specifications come from the code. Results come from recorded runs. No benchmark is published until it has been run.",
  },
  {
    title: "Honest status",
    body: "Everything carries a status, from Planned to Released, and moves forward only when it has actually happened.",
  },
  {
    title: "Built to last",
    body: "Reproducible data, deterministic evaluation, and training that survives interruption without wasting compute.",
  },
];

export default function Home() {
  const flagship = getModel("thor-30m")!;
  const spec = flagship.spec!;

  return (
    <>
      {/* Hero */}
      <section className="relative overflow-hidden">
        <div className="container-thor grid items-center gap-12 pt-16 pb-20 sm:pt-24 lg:grid-cols-[1.1fr_0.9fr] lg:pb-28">
          <div>
            <p className="label">THOR · Independent AI research</p>
            <h1 className="display mt-6 text-5xl sm:text-7xl lg:text-[5.5rem]">{site.tagline}</h1>
            <p className="lede mt-7 max-w-xl sm:text-xl">
              THOR is an independent AI company building its own language models, from the tokenizer to the
              trained weights, and the products that run on them.
            </p>
            <div className="mt-10 flex flex-wrap gap-3">
              <ButtonLink href="/models">Explore THOR Models</ButtonLink>
              <ButtonLink href="/research" variant="ghost">
                Read the engineering notes
              </ButtonLink>
            </div>
          </div>

          <div className="relative">
            <TokenField className="w-full" />
            <div className="absolute right-0 bottom-0 left-0 mx-auto w-full max-w-sm rounded-lg border border-graphite-600 bg-black/85 p-5 backdrop-blur sm:left-auto sm:mx-0">
              <div className="flex items-center justify-between gap-3">
                <span className="font-mono text-sm text-white">{flagship.name}</span>
                <StatusBadge status={flagship.status} />
              </div>
              <dl className="mt-4 grid grid-cols-3 gap-3 font-mono">
                {[
                  ["Params", formatParams(spec.params)],
                  ["Layers", String(spec.layers)],
                  ["Context", spec.context.toLocaleString("en-US")],
                ].map(([k, v]) => (
                  <div key={k}>
                    <dt className="text-[11px] tracking-[0.06em] text-steel uppercase">{k}</dt>
                    <dd className="mt-1 text-lg text-white tabular-nums">{v}</dd>
                  </div>
                ))}
              </dl>
            </div>
          </div>
        </div>
      </section>

      {/* Where things stand */}
      <section className="border-y border-graphite-600 bg-graphite-900">
        <div className="container-thor flex flex-col gap-4 py-6 sm:flex-row sm:items-center sm:justify-between">
          <p className="text-[15px] leading-relaxed text-silver">
            <span className="text-white">Where things stand:</span> the THOR training pipeline is built and tested.
            No THOR model has been trained yet. First target: {flagship.name}.
          </p>
          <p className="label shrink-0">Updated {formatDate(site.statusUpdated)}</p>
        </div>
      </section>

      <Section
        label="THOR Models"
        title="A model family, built in-house"
        intro={
          <>
            Parameter counts are specifications of size, not claims about capability.{" "}
            <Link href="/models" className="text-white underline decoration-graphite-600 underline-offset-4 hover:decoration-white">
              All models
            </Link>
          </>
        }
      >
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {models.map((m) => (
            <ModelCard key={m.slug} model={m} />
          ))}
        </div>
      </Section>

      <Section
        label="Engineering"
        title="Built from first principles"
        intro="Each stage is THOR's own code. Built stages are covered by automated tests."
      >
        <ol className="divide-y divide-graphite-600 border-y border-graphite-600">
          {pipeline.map((stage, i) => (
            <li key={stage.name} className="grid gap-3 py-5 sm:grid-cols-[3rem_12rem_1fr_auto] sm:items-center sm:gap-6">
              <span className="font-mono text-sm text-steel tabular-nums">{String(i + 1).padStart(2, "0")}</span>
              <span className="font-medium text-white">{stage.name}</span>
              <span className="text-[15px] text-silver">{stage.detail}</span>
              <span>
                <ComponentBadge status={stage.status} />
              </span>
            </li>
          ))}
        </ol>
      </Section>

      <Section label="Ecosystem" title="One company, a growing stack" intro="THOR is the parent brand. Products build on THOR's own models.">
        <ul className="grid gap-px overflow-hidden rounded-lg border border-graphite-600 bg-graphite-600 sm:grid-cols-2">
          {products.map((p) => (
            <li key={p.slug} className="flex flex-col gap-3 bg-black p-6">
              <div className="flex items-center justify-between gap-4">
                <h3 className="font-medium text-white">{p.name}</h3>
                <StatusBadge status={p.status} />
              </div>
              <p className="text-[15px] text-silver">{p.summary}</p>
            </li>
          ))}
          {products.length % 2 ? <li aria-hidden className="hidden bg-black sm:block" /> : null}
        </ul>
      </Section>

      <Section label="Principles" title="How THOR works">
        <div className="grid gap-x-10 gap-y-10 sm:grid-cols-2">
          {principles.map((p) => (
            <div key={p.title} className="border-t border-graphite-600 pt-5">
              <h3 className="font-medium text-white">{p.title}</h3>
              <p className="mt-2 text-[15px] leading-relaxed text-silver">{p.body}</p>
            </div>
          ))}
        </div>
      </Section>
    </>
  );
}

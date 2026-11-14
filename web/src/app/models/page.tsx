import type { Metadata } from "next";
import { ModelCard } from "@/components/model-card";
import { PageHeader, Section } from "@/components/ui";
import { architectureFamily, formatParams, models } from "@/content/models";
import { STATUS_INFO, STATUSES } from "@/content/status";

export const metadata: Metadata = {
  title: "Models",
  description: "THOR's proprietary language models: specifications, architecture, training status and evaluation.",
};

export default function ModelsPage() {
  const specified = models.filter((m) => m.spec);
  return (
    <>
      <PageHeader
        eyebrow="THOR Models"
        title="Language models, trained from scratch"
        lede={
          <p>
            THOR designs, trains and evaluates its own models. Specifications below come directly from the model code.
            No THOR model has been trained yet.
          </p>
        }
      />

      <Section label="Family" title="All models">
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {models.map((m) => (
            <ModelCard key={m.slug} model={m} />
          ))}
        </div>
      </Section>

      <Section label="Specifications" title="Side by side" intro={architectureFamily}>
        <div className="overflow-x-auto rounded-lg border border-graphite-600">
          <table className="w-full min-w-[40rem] text-left font-mono text-sm">
            <thead className="bg-graphite-900 text-[11px] tracking-[0.06em] text-silver uppercase">
              <tr>
                {["Model", "Params", "Non-embedding", "Layers", "d_model", "Heads", "FFN", "Context", "Vocab"].map((h) => (
                  <th key={h} scope="col" className="px-4 py-3 font-normal">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-graphite-600 tabular-nums">
              {specified.map(({ name, spec }) => (
                <tr key={name}>
                  <th scope="row" className="px-4 py-3 font-normal text-white">{name}</th>
                  <td className="px-4 py-3">{formatParams(spec!.params)}</td>
                  <td className="px-4 py-3 text-silver">{formatParams(spec!.nonEmbeddingParams)}</td>
                  <td className="px-4 py-3">{spec!.layers}</td>
                  <td className="px-4 py-3">{spec!.dModel}</td>
                  <td className="px-4 py-3">{spec!.heads}</td>
                  <td className="px-4 py-3">{spec!.ffnHidden.toLocaleString("en-US")}</td>
                  <td className="px-4 py-3">{spec!.context.toLocaleString("en-US")}</td>
                  <td className="px-4 py-3">{spec!.vocab.toLocaleString("en-US")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section label="Status" title="What each status means" intro="Every model moves forward only when that has actually happened.">
        <dl className="divide-y divide-graphite-600 border-y border-graphite-600">
          {STATUSES.map((s) => (
            <div key={s} className="grid gap-2 py-4 sm:grid-cols-[14rem_1fr]">
              <dt className="font-mono text-sm text-white">{STATUS_INFO[s].label}</dt>
              <dd className="text-[15px] text-silver">{STATUS_INFO[s].description}</dd>
            </div>
          ))}
        </dl>
      </Section>
    </>
  );
}

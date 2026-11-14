import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { StatusBadge } from "@/components/status-badge";
import { Section, SpecList } from "@/components/ui";
import { architectureFamily, formatParams, getModel, models } from "@/content/models";
import { STATUS_INFO } from "@/content/status";

export const dynamicParams = false;

export function generateStaticParams() {
  return models.map((m) => ({ slug: m.slug }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const model = getModel((await params).slug);
  if (!model) return {};
  return { title: model.name, description: `${model.name}: ${model.role}. Status: ${STATUS_INFO[model.status].label}.` };
}

export default async function ModelPage({ params }: { params: Promise<{ slug: string }> }) {
  const model = getModel((await params).slug);
  if (!model) notFound();
  const { spec } = model;

  return (
    <>
      <header className="container-thor pt-12 pb-12 sm:pt-16 sm:pb-16">
        <Link href="/models" className="label hover:text-white">
          ← THOR Models
        </Link>
        <div className="mt-8 flex flex-wrap items-center gap-4">
          <h1 className="display font-mono text-4xl sm:text-6xl">{model.name}</h1>
          <StatusBadge status={model.status} />
        </div>
        <p className="mt-3 text-silver">{model.role}</p>
        <p className="lede mt-6 max-w-2xl">{model.summary}</p>
      </header>

      <Section label="Specification" title={spec ? formatParams(spec.params) + " parameters" : "Pending"}>
        {spec ? (
          <SpecList
            rows={[
              ["Parameters", spec.params.toLocaleString("en-US")],
              ["Non-embedding", spec.nonEmbeddingParams.toLocaleString("en-US")],
              ["Layers", spec.layers],
              ["d_model", spec.dModel],
              ["Attention heads", `${spec.heads} × ${spec.dModel / spec.heads}`],
              ["FFN hidden", spec.ffnHidden.toLocaleString("en-US")],
              ["Context length", `${spec.context.toLocaleString("en-US")} tokens`],
              ["Vocabulary", `${spec.vocab.toLocaleString("en-US")} (THOR BPE)`],
              ["Architecture", architectureFamily],
            ]}
          />
        ) : (
          <p className="text-silver">The specification will be published when this model enters development.</p>
        )}
      </Section>

      <Section label="Training" title="Training status">
        <ul className="space-y-3 text-[15px] text-silver">
          {model.training.map((t) => (
            <li key={t} className="flex gap-3">
              <span aria-hidden className="mt-2 size-1.5 shrink-0 rounded-[2px] bg-steel" />
              {t}
            </li>
          ))}
        </ul>
      </Section>

      <Section label="Evaluation" title="Results">
        <p className="text-[15px] text-silver">{model.evaluation}</p>
      </Section>

      <Section label="Availability" title="Downloads">
        <p className="text-[15px] text-silver">
          No weights are available. Current status: {STATUS_INFO[model.status].label}.{" "}
          {STATUS_INFO[model.status].description}
        </p>
      </Section>
    </>
  );
}

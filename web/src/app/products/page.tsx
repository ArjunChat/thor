import type { Metadata } from "next";
import { StatusBadge } from "@/components/status-badge";
import { PageHeader, Section } from "@/components/ui";
import { products } from "@/content/products";

export const metadata: Metadata = {
  title: "Products",
  description: "THOR Chat, THOR Studio, THOR Cloud and the rest of the THOR ecosystem, with honest status for each.",
};

export default function ProductsPage() {
  return (
    <>
      <PageHeader
        eyebrow="Products"
        title="Products built on THOR models"
        lede={
          <p>
            THOR&apos;s products will run on THOR&apos;s own models. None are publicly available yet; the status of each
            is shown below.
          </p>
        }
      />
      <Section label="Ecosystem" title="Everything THOR is building">
        <ul className="space-y-4">
          {products.map((p) => (
            <li key={p.slug} id={p.slug} className="scroll-mt-24 rounded-lg border border-graphite-600 bg-graphite-900 p-6 sm:p-8">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <h2 className="text-xl font-semibold tracking-tight text-white">{p.name}</h2>
                <StatusBadge status={p.status} />
              </div>
              <p className="mt-3 text-white/90">{p.summary}</p>
              <p className="mt-2 text-[15px] leading-relaxed text-silver">{p.detail}</p>
            </li>
          ))}
        </ul>
      </Section>
      <Section label="Policy" title="External models">
        <p className="max-w-2xl text-[15px] leading-relaxed text-silver">
          THOR may offer external AI providers as clearly identified, optional integrations. Output from an external
          model is always labelled with its source and is never presented as a THOR model.
        </p>
      </Section>
    </>
  );
}

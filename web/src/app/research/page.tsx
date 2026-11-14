import type { Metadata } from "next";
import Link from "next/link";
import { StatusBadge } from "@/components/status-badge";
import { Arrow, PageHeader, Section } from "@/components/ui";
import { research } from "@/content/research";
import { formatDate } from "@/content/site";

export const metadata: Metadata = {
  title: "Research",
  description: "THOR Research: experiments, engineering notes and technical reports.",
};

export default function ResearchPage() {
  const published = research.filter((r) => r.date);
  const upcoming = research.filter((r) => !r.date);
  return (
    <>
      <PageHeader
        eyebrow="THOR Research"
        title="Experiments, notes and reports"
        lede={
          <p>
            How THOR&apos;s models are built, and what the experiments actually measured. Results are published from
            recorded runs only.
          </p>
        }
      />
      <Section label="Published" title={`${published.length} entries`}>
        <ul className="divide-y divide-graphite-600 border-y border-graphite-600">
          {published.map((r) => (
            <li key={r.slug}>
              <Link href={`/research/${r.slug}`} className="group grid gap-2 py-6 sm:grid-cols-[10rem_1fr_auto] sm:gap-8">
                <div className="font-mono text-xs text-steel">
                  <p className="uppercase tracking-[0.06em]">{r.kind}</p>
                  <p className="mt-1">{formatDate(r.date!)}</p>
                </div>
                <div>
                  <h3 className="text-lg font-medium text-white">{r.title}</h3>
                  <p className="mt-2 text-[15px] leading-relaxed text-silver">{r.summary}</p>
                </div>
                <Arrow className="hidden self-center text-silver group-hover:text-white sm:block" />
              </Link>
            </li>
          ))}
        </ul>
      </Section>
      {upcoming.length ? (
        <Section label="Upcoming" title="Not yet published">
          <ul className="space-y-4">
            {upcoming.map((r) => (
              <li key={r.slug} className="flex flex-col gap-3 rounded-lg border border-graphite-600 p-6 sm:flex-row sm:items-start sm:justify-between">
                <div>
                  <p className="font-mono text-xs tracking-[0.06em] text-steel uppercase">{r.kind}</p>
                  <h3 className="mt-2 font-medium text-white">{r.title}</h3>
                  <p className="mt-2 text-[15px] text-silver">{r.summary}</p>
                </div>
                <StatusBadge status={r.status} />
              </li>
            ))}
          </ul>
        </Section>
      ) : null}
    </>
  );
}

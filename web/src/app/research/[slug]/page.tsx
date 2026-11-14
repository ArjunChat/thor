import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { getResearch, research } from "@/content/research";
import { formatDate, site } from "@/content/site";

export const dynamicParams = false;

// Planned entries have no page until they are published.
export function generateStaticParams() {
  return research.filter((r) => r.date).map((r) => ({ slug: r.slug }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const entry = getResearch((await params).slug);
  return entry ? { title: entry.title, description: entry.summary } : {};
}

export default async function ResearchEntryPage({ params }: { params: Promise<{ slug: string }> }) {
  const entry = getResearch((await params).slug);
  if (!entry || !entry.date) notFound();

  return (
    <article className="container-thor pt-12 sm:pt-16">
      <Link href="/research" className="label hover:text-white">
        ← THOR Research
      </Link>
      <header className="mt-8 max-w-3xl">
        <p className="font-mono text-xs tracking-[0.06em] text-steel uppercase">
          {entry.kind} · {formatDate(entry.date)}
        </p>
        <h1 className="display mt-4 text-4xl sm:text-5xl">{entry.title}</h1>
        <p className="lede mt-6">{entry.summary}</p>
      </header>

      <div className="mt-14 max-w-3xl space-y-14 border-t border-graphite-600 pt-14">
        {entry.sections.map((s) => (
          <section key={s.heading}>
            <h2 className="text-xl font-semibold tracking-tight text-white">{s.heading}</h2>
            {s.paragraphs?.map((p) => (
              <p key={p} className="mt-4 text-[16px] leading-[1.75] text-silver">
                {p}
              </p>
            ))}
            {s.list ? (
              <ul className="mt-4 space-y-3">
                {s.list.map((item) => (
                  <li key={item} className="flex gap-3 text-[16px] leading-[1.7] text-silver">
                    <span aria-hidden className="mt-[0.7em] size-1.5 shrink-0 rounded-[2px] bg-steel" />
                    {item}
                  </li>
                ))}
              </ul>
            ) : null}
            {s.table ? (
              <div className="mt-6 overflow-x-auto rounded-lg border border-graphite-600">
                <table className="w-full text-left text-sm">
                  {s.table.head.some(Boolean) ? (
                    <thead className="bg-graphite-900 font-mono text-[11px] tracking-[0.06em] text-silver uppercase">
                      <tr>
                        {s.table.head.map((h, i) => (
                          <th key={i} scope="col" className="px-4 py-3 font-normal">
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                  ) : null}
                  <tbody className="divide-y divide-graphite-600">
                    {s.table.rows.map((row) => (
                      <tr key={row[0]}>
                        <th scope="row" className="w-[34%] px-4 py-3 align-top font-mono text-[13px] font-normal text-white">
                          {row[0]}
                        </th>
                        {row.slice(1).map((cell, i) => (
                          <td key={i} className="px-4 py-3 align-top text-silver tabular-nums">
                            {cell}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
          </section>
        ))}
        {entry.source ? (
          <p className="font-mono text-xs text-steel">
            Source:{" "}
            {site.repo.public ? (
              <a href={`${site.repo.url}/blob/main/${entry.source}`} className="text-silver underline underline-offset-4 hover:text-white">
                {entry.source}
              </a>
            ) : (
              entry.source
            )}
          </p>
        ) : null}
      </div>
    </article>
  );
}

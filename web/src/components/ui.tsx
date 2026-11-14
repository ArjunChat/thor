import Link from "next/link";
import type { ReactNode } from "react";

export function ButtonLink({
  href,
  children,
  variant = "primary",
}: {
  href: string;
  children: ReactNode;
  variant?: "primary" | "ghost";
}) {
  const base =
    "group inline-flex h-11 items-center gap-2 rounded-md px-5 text-[15px] font-medium transition-colors";
  const styles =
    variant === "primary"
      ? "bg-white text-black hover:bg-silver"
      : "border border-graphite-600 text-white hover:border-steel hover:bg-graphite-900";
  return (
    <Link href={href} className={`${base} ${styles}`}>
      {children}
      <Arrow />
    </Link>
  );
}

export function Arrow({ className = "" }: { className?: string }) {
  return (
    <svg
      aria-hidden
      viewBox="0 0 16 16"
      className={`size-3.5 transition-transform group-hover:translate-x-0.5 ${className}`}
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
    >
      <path d="M3 8h9M8.5 4.5 12 8l-3.5 3.5" strokeLinecap="square" />
    </svg>
  );
}

export function PageHeader({ eyebrow, title, lede }: { eyebrow: string; title: string; lede?: ReactNode }) {
  return (
    <header className="container-thor pt-16 pb-12 sm:pt-24 sm:pb-16">
      <p className="label">{eyebrow}</p>
      <h1 className="display mt-5 max-w-4xl text-4xl sm:text-6xl">{title}</h1>
      {lede ? <div className="lede mt-6 max-w-2xl">{lede}</div> : null}
    </header>
  );
}

export function Section({
  label,
  title,
  intro,
  children,
  id,
}: {
  label: string;
  title?: string;
  intro?: ReactNode;
  children: ReactNode;
  id?: string;
}) {
  return (
    <section id={id} className="container-thor border-t border-graphite-600 py-16 sm:py-24">
      <div className="grid gap-10 lg:grid-cols-[16rem_1fr] lg:gap-16">
        <div>
          <p className="label">{label}</p>
          {title ? <h2 className="display mt-4 text-2xl sm:text-3xl">{title}</h2> : null}
          {intro ? <div className="mt-4 text-[15px] leading-relaxed text-silver">{intro}</div> : null}
        </div>
        <div className="min-w-0">{children}</div>
      </div>
    </section>
  );
}

export function SpecList({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <dl className="divide-y divide-graphite-600 border-y border-graphite-600 font-mono text-sm">
      {rows.map(([k, v]) => (
        <div key={k} className="grid grid-cols-[minmax(7rem,40%)_1fr] gap-4 py-3">
          <dt className="text-steel">{k}</dt>
          <dd className="text-white tabular-nums">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

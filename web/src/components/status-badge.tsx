import { COMPONENT_STATUS_INFO, STATUS_INFO, type ComponentStatus, type Status } from "@/content/status";

const STYLES: Record<Status, { box: string; dot: string }> = {
  planned: { box: "border-graphite-600 text-silver", dot: "border border-steel bg-transparent" },
  "in-development": { box: "border-signal/40 text-white", dot: "bg-signal" },
  training: { box: "border-signal/60 text-white", dot: "bg-signal animate-pulse" },
  "trained-internal": { box: "border-silver/50 text-white", dot: "bg-silver" },
  released: { box: "border-white bg-white text-black", dot: "bg-black" },
};

function Badge({ label, box, dot, title }: { label: string; box: string; dot: string; title?: string }) {
  return (
    <span
      title={title}
      className={`inline-flex shrink-0 items-center gap-2 rounded-full border px-2.5 py-1 font-mono text-[11px] leading-none tracking-[0.06em] uppercase ${box}`}
    >
      <span aria-hidden className={`size-1.5 rounded-[2px] ${dot}`} />
      {label}
    </span>
  );
}

export function StatusBadge({ status }: { status: Status }) {
  const info = STATUS_INFO[status];
  return <Badge label={info.label} title={info.description} {...STYLES[status]} />;
}

export function ComponentBadge({ status }: { status: ComponentStatus }) {
  return (
    <Badge
      label={COMPONENT_STATUS_INFO[status].label}
      {...(status === "working" ? STYLES["in-development"] : STYLES.planned)}
    />
  );
}

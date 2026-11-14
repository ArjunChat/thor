import Link from "next/link";
import { formatParams, type Model } from "@/content/models";
import { StatusBadge } from "./status-badge";
import { Arrow } from "./ui";

export function ModelCard({ model }: { model: Model }) {
  return (
    <Link
      href={`/models/${model.slug}`}
      className="group flex h-full flex-col rounded-lg border border-graphite-600 bg-graphite-900 p-6 transition-colors hover:border-steel"
    >
      <div>
        <StatusBadge status={model.status} />
      </div>
      <h3 className="mt-5 font-mono text-lg font-medium whitespace-nowrap text-white">{model.name}</h3>
      <p className="mt-1 text-sm text-silver">{model.role}</p>
      <div className="mt-8 flex items-end justify-between gap-4">
        {model.spec ? (
          <p className="font-mono text-sm text-silver">
            <span className="text-2xl text-white tabular-nums">{formatParams(model.spec.params)}</span>
            <span className="ml-2">params</span>
          </p>
        ) : (
          <p className="font-mono text-sm text-silver">Specification pending</p>
        )}
        <Arrow className="text-silver group-hover:text-white" />
      </div>
    </Link>
  );
}

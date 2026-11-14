import Link from "next/link";
import { formatDate, site } from "@/content/site";
import { LogoSymbol } from "./logo";

export function SiteFooter() {
  return (
    <footer className="mt-24 border-t border-graphite-600">
      <div className="container-thor grid gap-12 py-14 sm:grid-cols-[1fr_auto]">
        <div className="max-w-sm">
          <div className="flex items-center gap-3 text-white">
            <LogoSymbol className="size-6" />
            <span className="font-medium">{site.tagline}</span>
          </div>
          <p className="mt-4 text-sm leading-relaxed text-silver">
            An independent AI company building its own language models.
          </p>
        </div>
        <nav aria-label="Footer">
          <ul className="grid grid-cols-2 gap-x-12 gap-y-3 text-sm text-silver sm:grid-cols-3">
            {site.nav.map((item) => (
              <li key={item.href}>
                <Link href={item.href} className="hover:text-white">
                  {item.label}
                </Link>
              </li>
            ))}
            {site.repo.public ? (
              <li>
                <a href={site.repo.url} className="hover:text-white">
                  Source
                </a>
              </li>
            ) : null}
          </ul>
        </nav>
      </div>
      <div className="container-thor flex flex-col gap-2 border-t border-graphite-600 py-6 text-xs text-steel sm:flex-row sm:justify-between">
        <p>© {new Date(`${site.statusUpdated}T00:00:00Z`).getUTCFullYear()} THOR</p>
        <p>Status as of {formatDate(site.statusUpdated)}</p>
      </div>
    </footer>
  );
}

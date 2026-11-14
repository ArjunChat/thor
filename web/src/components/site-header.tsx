import Link from "next/link";
import { site } from "@/content/site";
import { Wordmark } from "./logo";

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-40 border-b border-graphite-600/70 bg-black/80 backdrop-blur-md">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:top-3 focus:left-3 focus:rounded-sm focus:bg-white focus:px-3 focus:py-2 focus:text-black"
      >
        Skip to content
      </a>
      <div className="container-thor flex h-16 items-center justify-between gap-6">
        <Link href="/" aria-label="THOR home" className="text-white">
          <Wordmark decorative className="h-[15px] w-auto" />
        </Link>

        <nav aria-label="Primary" className="hidden md:block">
          <ul className="flex items-center gap-8 text-[14px] text-silver">
            {site.nav.map((item) => (
              <li key={item.href}>
                <Link href={item.href} className="transition-colors hover:text-white">
                  {item.label}
                </Link>
              </li>
            ))}
          </ul>
        </nav>

        {/* Mobile menu: native <details>, no JavaScript needed */}
        <details className="group relative md:hidden">
          <summary className="flex size-10 cursor-pointer list-none items-center justify-center rounded-md border border-graphite-600 text-white [&::-webkit-details-marker]:hidden">
            <span className="sr-only">Menu</span>
            <svg aria-hidden viewBox="0 0 16 16" className="size-4" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path className="group-open:hidden" d="M2 5h12M2 11h12" />
              <path className="hidden group-open:block" d="M3.5 3.5l9 9M12.5 3.5l-9 9" />
            </svg>
          </summary>
          <nav
            aria-label="Mobile"
            className="absolute top-12 right-0 w-56 rounded-lg border border-graphite-600 bg-graphite-900 p-2 shadow-2xl"
          >
            <ul>
              {site.nav.map((item) => (
                <li key={item.href}>
                  <Link
                    href={item.href}
                    className="block rounded-md px-3 py-2.5 text-[15px] text-silver hover:bg-graphite-800 hover:text-white"
                  >
                    {item.label}
                  </Link>
                </li>
              ))}
            </ul>
          </nav>
        </details>
      </div>
    </header>
  );
}

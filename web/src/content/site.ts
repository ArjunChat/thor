export const site = {
  name: "THOR",
  tagline: "Intelligence, engineered.",
  description:
    "THOR is an independent AI company engineering its own language models, from the tokenizer to the trained weights, and building accessible AI products on top of them.",
  /**
   * Canonical origin for metadata, the sitemap and social images. An explicit
   * NEXT_PUBLIC_SITE_URL wins; on Vercel, the project's production domain is
   * provided at build time (and follows a custom domain once one is added).
   */
  url:
    process.env.NEXT_PUBLIC_SITE_URL ??
    (process.env.VERCEL_PROJECT_PRODUCTION_URL
      ? `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}`
      : "https://thor-alpha-three.vercel.app"),
  /** Date of the last factual status update shown on the site. */
  statusUpdated: "2026-11-14",
  repo: {
    url: "https://github.com/ArjunChat/thor",
    /** Links to source are shown only when the repository is public. */
    public: false,
  },
  nav: [
    { href: "/models", label: "Models" },
    { href: "/research", label: "Research" },
    { href: "/products", label: "Products" },
    { href: "/developers", label: "Developers" },
    { href: "/about", label: "About" },
  ],
} as const;

export function formatDate(iso: string): string {
  return new Date(`${iso}T00:00:00Z`).toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
    timeZone: "UTC",
  });
}

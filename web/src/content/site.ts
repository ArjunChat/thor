export const site = {
  name: "THOR",
  tagline: "Intelligence, engineered.",
  description:
    "THOR is an independent AI company engineering its own language models, from the tokenizer to the trained weights, and building accessible AI products on top of them.",
  url: process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000",
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

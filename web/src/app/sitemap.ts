import type { MetadataRoute } from "next";
import { models } from "@/content/models";
import { research } from "@/content/research";
import { site } from "@/content/site";

export default function sitemap(): MetadataRoute.Sitemap {
  const paths = [
    "",
    ...site.nav.map((n) => n.href),
    ...models.map((m) => `/models/${m.slug}`),
    ...research.filter((r) => r.date).map((r) => `/research/${r.slug}`),
  ];
  return paths.map((p) => ({ url: `${site.url}${p}`, lastModified: site.statusUpdated }));
}

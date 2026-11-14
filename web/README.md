# THOR website

The public site for THOR: Next.js 16 (App Router), React 19 and Tailwind CSS 4. Every page is prerendered as static HTML; there is no database, no client-side data fetching, no analytics and no third-party requests (fonts are self-hosted).

```bash
npm install
npm run dev        # http://localhost:3000
npm run check      # lint + typecheck + tests + production build
```

## Where things live

| Path | What |
|---|---|
| `src/content/` | **All factual content**: models, research, products, statuses. Edit these, not the pages. |
| `src/content/status.ts` | The only allowed status vocabulary: Planned → In development → Training → Trained (internal) → Released |
| `src/components/` | Logo, status badges, layout primitives |
| `src/styles/tokens.css` | Copy of `brand/tokens.css` |
| `src/app/` | Routes: `/`, `/models`, `/models/[slug]`, `/research`, `/research/[slug]`, `/products`, `/developers`, `/about` |

## Guardrails (tests)

- Model specifications on the site must exactly match `models/thor/config.py`.
- A model can't be marked trained or released without published evaluation and a report.
- Design tokens and logo geometry must be identical to `brand/`.
- A claims guard rejects superlatives, frontier-model comparisons and invented traction ("state-of-the-art", "outperforms", "trusted by", …).

## Updating status

When something actually happens (for example, THOR-30M finishes training), update `src/content/models.ts` and `research.ts` with the measured numbers, bump `site.statusUpdated`, and run `npm run check`.

Set `site.repo.public` to `true` in `src/content/site.ts` if the repository becomes public; source links appear only then. The canonical URL comes from `NEXT_PUBLIC_SITE_URL` if set, otherwise Vercel's production domain (`VERCEL_PROJECT_PRODUCTION_URL`), otherwise https://thor-alpha-three.vercel.app.

## Deploying (Vercel)

The repo-root `vercel.json` defines a single Vercel service, `web` (root `web/`, Next.js), which serves all public traffic. Live at https://thor-alpha-three.vercel.app. Import `ArjunChat/thor` in Vercel and leave the project's Root Directory at the repository root; `vercel.json` points the service at `web/`. `models/` is not deployed: it is an offline training toolkit, not a web service.

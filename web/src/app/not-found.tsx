import { ButtonLink } from "@/components/ui";

export default function NotFound() {
  return (
    <section className="container-thor py-32">
      <p className="label">404</p>
      <h1 className="display mt-5 text-4xl sm:text-6xl">This page doesn&apos;t exist.</h1>
      <p className="lede mt-6 max-w-xl">It may not have been published yet, or the address may be wrong.</p>
      <div className="mt-10">
        <ButtonLink href="/">Back to THOR</ButtonLink>
      </div>
    </section>
  );
}

/**
 * THOR marks as inline SVG. Geometry is copied from brand/logo/*.svg;
 * tests/brand.test.ts fails if the two ever differ.
 */

export const WORDMARK_PATH =
  "M0 0H84V16H50V100H34V16H0Z M102 0H118V42H166V0H182V100H166V58H118V100H102Z M236 0H260A32 32 0 0 1 292 32V68A32 32 0 0 1 260 100H236A32 32 0 0 1 204 68V32A32 32 0 0 1 236 0ZM236 16A16 16 0 0 0 220 32V68A16 16 0 0 0 236 84H260A16 16 0 0 0 276 68V32A16 16 0 0 0 260 16Z M314 0H365A29 29 0 0 1 365 58H330V100H314ZM330 16V42H365A13 13 0 0 0 365 16Z";
export const WORDMARK_LEG_PATH = "M338.6 42H358.6L400 100H380Z";
export const WORDMARK_UNIT_PATH = "M240 42H256V58H240Z";
export const SYMBOL_PATH =
  "M34 0H66A34 34 0 0 1 100 34V66A34 34 0 0 1 66 100H34A34 34 0 0 1 0 66V34A34 34 0 0 1 34 0ZM36 18A18 18 0 0 0 18 36V64A18 18 0 0 0 36 82H64A18 18 0 0 0 82 64V36A18 18 0 0 0 64 18Z";
export const SYMBOL_UNIT_PATH = "M39 39H61V61H39Z";

interface MarkProps {
  className?: string;
  /** Decorative marks are hidden from assistive tech; labelled ones announce "THOR". */
  decorative?: boolean;
}

export function Wordmark({ className, decorative = false }: MarkProps) {
  return (
    <svg
      viewBox="0 0 400 100"
      className={className}
      role={decorative ? undefined : "img"}
      aria-label={decorative ? undefined : "THOR"}
      aria-hidden={decorative || undefined}
    >
      <path fillRule="evenodd" fill="currentColor" d={WORDMARK_PATH} />
      <path fill="currentColor" d={WORDMARK_LEG_PATH} />
      <path fill="var(--thor-signal)" d={WORDMARK_UNIT_PATH} />
    </svg>
  );
}

export function LogoSymbol({ className, decorative = true }: MarkProps) {
  return (
    <svg
      viewBox="0 0 100 100"
      className={className}
      role={decorative ? undefined : "img"}
      aria-label={decorative ? undefined : "THOR"}
      aria-hidden={decorative || undefined}
    >
      <path fillRule="evenodd" fill="currentColor" d={SYMBOL_PATH} />
      <path fill="var(--thor-signal)" d={SYMBOL_UNIT_PATH} />
    </svg>
  );
}

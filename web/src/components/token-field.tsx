/**
 * Hero visual: a field of square "units" (the symbol's unit, repeated) whose
 * density falls off from a focal point, with a few lit in Signal Blue.
 * Deterministic: the same SVG is rendered every build, with no client JavaScript.
 */

function rand(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

const COLS = 28;
const ROWS = 18;
const CELL = 20;

export function TokenField({ className = "" }: { className?: string }) {
  const r = rand(30_075_200);
  const cells: { x: number; y: number; size: number; lit: boolean; o: number }[] = [];
  const fx = COLS * 0.62;
  const fy = ROWS * 0.46;
  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLS; col++) {
      const d = Math.hypot((col - fx) / COLS, (row - fy) / ROWS);
      const p = Math.max(0, 1 - d * 2.1);
      const v = r();
      if (v > p * 0.95 + 0.05) continue;
      const size = p > 0.62 && v < 0.25 ? 10 : p > 0.35 ? 6 : 4;
      cells.push({
        x: col * CELL + (CELL - size) / 2,
        y: row * CELL + (CELL - size) / 2,
        size,
        lit: p > 0.55 && r() < 0.07,
        o: 0.18 + p * 0.82,
      });
    }
  }
  return (
    <svg
      viewBox={`0 0 ${COLS * CELL} ${ROWS * CELL}`}
      className={className}
      aria-hidden
      preserveAspectRatio="xMidYMid meet"
    >
      {cells.map((c, i) => (
        <rect
          key={i}
          x={c.x}
          y={c.y}
          width={c.size}
          height={c.size}
          rx={c.size > 6 ? 2 : 1}
          fill={c.lit ? "var(--thor-signal)" : "var(--thor-white)"}
          opacity={c.lit ? 1 : Number(c.o.toFixed(2))}
        />
      ))}
    </svg>
  );
}

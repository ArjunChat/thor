import { ImageResponse } from "next/og";
import { WORDMARK_LEG_PATH, WORDMARK_PATH, WORDMARK_UNIT_PATH } from "@/components/logo";
import { site } from "@/content/site";

export const alt = `${site.name}: ${site.tagline}`;
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default async function Image() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          padding: 80,
          background: "#0a0a0b",
          color: "#f5f6f7",
        }}
      >
        <svg width="320" height="80" viewBox="0 0 400 100">
          <path fillRule="evenodd" fill="#f5f6f7" d={WORDMARK_PATH} />
          <path fill="#f5f6f7" d={WORDMARK_LEG_PATH} />
          <path fill="#3d7bff" d={WORDMARK_UNIT_PATH} />
        </svg>
        <div style={{ display: "flex", flexDirection: "column" }}>
          <div style={{ fontSize: 84, fontWeight: 600, letterSpacing: -3 }}>{site.tagline}</div>
          <div style={{ marginTop: 24, fontSize: 30, color: "#a9adb6" }}>
            An independent AI company building its own language models.
          </div>
        </div>
      </div>
    ),
    size,
  );
}

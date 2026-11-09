import { useEffect, useRef, type ReactNode, type RefObject } from "react";

// The welcome video has product cards drawn into it. A real, tappable card is laid exactly over one of them.
// Card boxes in the video's own pixels (2160x3840) per playback time, measured by tracking each card frame by frame.
// Linear between keyframes.
const VIDEO_W = 2160;
const VIDEO_H = 3840;
type Keyframe = [number, number, number, number, number]; // [time s, x, y, w, h]

// "Footwear": rests, swells ~12% between 3.0 s and 3.6 s, holds, and settles back by 5.2 s
export const FOOTWEAR_KEYFRAMES: Keyframe[] = [
  [0.0, 100, 1472, 492, 588],
  [2.95, 100, 1472, 492, 588],
  [3.6, 73, 1434, 552, 659],
  [4.7, 73, 1434, 552, 659],
  [5.2, 100, 1472, 492, 588],
];

// "Try on" (right side): enlarges ~12% around 1.5-2.4 s, eases back by 3.5 s, and pulses again at 6.5 s and 7.75 s
export const TRY_ON_KEYFRAMES: Keyframe[] = [
  [0.0, 1578, 1807, 489, 509],
  [0.5, 1575, 1805, 495, 513],
  [1.35, 1580, 1810, 484, 503],
  [1.5, 1578, 1807, 488, 507],
  [2.0, 1546, 1718, 551, 619],
  [2.4, 1539, 1712, 562, 629],
  [2.9, 1542, 1713, 560, 629],
  [3.5, 1552, 1722, 539, 609],
  [6.0, 1551, 1722, 541, 609],
  [6.5, 1541, 1711, 559, 629],
  [7.0, 1553, 1723, 539, 607],
  [7.25, 1554, 1725, 535, 602],
  [7.75, 1546, 1716, 557, 621],
  [8.25, 1545, 1716, 553, 625],
];
// Drawn slightly larger than the video card so none of its edge or shadow peeks out
const COVER_PAD = 0.03;

function boxAt(keyframes: Keyframe[], t: number): [number, number, number, number] {
  const last = keyframes[keyframes.length - 1];
  if (t >= last[0]) return [last[1], last[2], last[3], last[4]];
  for (let i = 1; i < keyframes.length; i++) {
    const [t1] = keyframes[i];
    if (t <= t1) {
      const a = keyframes[i - 1];
      const b = keyframes[i];
      const f = (t - a[0]) / (t1 - a[0] || 1);
      return [1, 2, 3, 4].map((k) => a[k] + (b[k] - a[k]) * f) as [number, number, number, number];
    }
  }
  return [last[1], last[2], last[3], last[4]];
}

type Props = {
  videoRef: RefObject<HTMLVideoElement | null>;
  keyframes: Keyframe[];
  label: ReactNode;
  ariaLabel: string;
  icon: ReactNode;
  onClick: () => void;
  sheenDelay?: string;
};

export default function VideoEntryCard({ videoRef, keyframes, label, ariaLabel, icon, onClick, sheenDelay = "0s" }: Props) {
  const cardRef = useRef<HTMLButtonElement>(null);

  // Follow the video card every frame. Styles are written directly (no React re-render at 60 fps).
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const video = videoRef.current;
      const card = cardRef.current;
      const parent = card?.offsetParent as HTMLElement | null;
      if (video && card && parent) {
        const v = video.getBoundingClientRect();
        const p = parent.getBoundingClientRect();
        // object-fit: cover
        const k = Math.max(v.width / VIDEO_W, v.height / VIDEO_H);
        const offX = v.left - p.left + (v.width - VIDEO_W * k) / 2;
        const offY = v.top - p.top + (v.height - VIDEO_H * k) / 2;
        const [x, y, w, h] = boxAt(keyframes, video.currentTime || 0);
        const padX = w * COVER_PAD;
        const padY = h * COVER_PAD;
        const width = (w + 2 * padX) * k;
        card.style.left = `${offX + (x - padX) * k}px`;
        card.style.top = `${offY + (y - padY) * k}px`;
        card.style.width = `${width}px`;
        card.style.height = `${(h + 2 * padY) * k}px`;
        card.style.fontSize = `${width / 10}px`;
        card.style.visibility = "visible";
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [videoRef, keyframes]);

  return (
    <button
      ref={cardRef}
      onClick={(e) => {
        e.stopPropagation();
        onClick();
      }}
      aria-label={ariaLabel}
      style={{
        position: "absolute",
        visibility: "hidden", // until the first position is computed
        zIndex: 11,
        padding: 0,
        border: "none",
        borderRadius: "1.0em",
        background: "linear-gradient(160deg, #EDEBE8 0%, #DCD8D3 100%)",
        boxShadow: "0 0.35em 0.9em rgba(60, 45, 35, 0.22)",
        cursor: "pointer",
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
        alignItems: "flex-start",
        textAlign: "left",
        outline: "none",
        WebkitTapHighlightColor: "transparent",
      }}
    >
      <style>{`
        @keyframes videoCardSheen { 0%, 55% { transform: translateX(-130%) skewX(-20deg); } 85%, 100% { transform: translateX(260%) skewX(-20deg); } }
        @keyframes videoCardRim { 0%, 100% { opacity: 0.35; } 50% { opacity: 1; } }
      `}</style>
      {/* Rim light and diagonal sheen, like the satin glow on the video's cards */}
      <div aria-hidden style={{ position: "absolute", inset: 0, borderRadius: "inherit", pointerEvents: "none", boxShadow: "inset 0 0 0 0.08em rgba(255,255,255,0.9), inset 0.15em -0.15em 0.5em rgba(255,255,255,0.75)", animation: "videoCardRim 3.2s ease-in-out infinite", animationDelay: sheenDelay }} />
      <div aria-hidden style={{ position: "absolute", top: 0, bottom: 0, left: 0, width: "45%", pointerEvents: "none", background: "linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.55) 50%, rgba(255,255,255,0) 100%)", animation: "videoCardSheen 4.5s ease-in-out infinite", animationDelay: sheenDelay }} />
      <div style={{ flex: 1, width: "100%", display: "flex", alignItems: "center", justifyContent: "center", paddingTop: "0.6em" }}>
        {icon}
      </div>
      {/* Label and page dots placed like the video's own cards */}
      <div style={{ padding: "0 0.75em 0.9em 0.75em", width: "100%", boxSizing: "border-box" }}>
        <div style={{ fontFamily: "'Poppins', sans-serif", fontWeight: 500, fontSize: "0.78em", lineHeight: 1, color: "#1B1B1B", letterSpacing: "0.01em" }}>
          {label}
        </div>
        <div style={{ display: "flex", gap: "0.22em", marginTop: "0.55em" }}>
          {["#1B1B1B", "#B9B5B0", "#B9B5B0"].map((c, i) => (
            <span key={i} style={{ width: "0.3em", height: "0.3em", borderRadius: "50%", background: c }} />
          ))}
        </div>
      </div>
    </button>
  );
}

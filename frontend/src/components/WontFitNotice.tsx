import type { FitCheck } from "../context/TrialRoomContext";

const RED = "#A03C3C";
const BROWN = "#5C4A3D";

const DIMENSION_NAMES: Record<string, string> = {
  chest: "Chest",
  waist: "Waist",
  shoulder: "Shoulders",
};

// Vertical position (in the 100x140 diagram) where each strain zone is drawn
const ZONE_Y: Record<string, number[]> = {
  "shoulders and upper back": [34],
  "shoulders and chest": [34, 52],
  "chest and upper arms": [52],
  "chest and stomach": [52, 82],
  "stomach and waist": [82, 96],
  stomach: [84],
};

type Props = {
  message: string;
  recommendedSize: string | null;
  fitCheck: FitCheck | null;
  // True when no render could depict it believably: the strain diagram replaces the photo
  illustrationOnly: boolean;
  onTryRecommended: () => void;
};

/** Honest diagram of a garment that is too small: the body outline, a garment that stops short of it, strain marks. */
function StrainIllustration({ fitCheck }: { fitCheck: FitCheck | null }) {
  const zones = fitCheck?.strain_zones?.length ? fitCheck.strain_zones : ["chest and stomach"];
  const ys = Array.from(new Set(zones.flatMap((z) => ZONE_Y[z] || [])));
  return (
    <svg viewBox="0 0 100 140" width="11vh" height="15.4vh" role="img" aria-label="Illustration: garment too small for the body">
      {/* Body outline (unchanged, real size) */}
      <path
        d="M50 8 a9 9 0 1 1 0.1 0 M36 30 Q50 25 64 30 L78 36 L82 70 M22 70 L18 36 L36 30 M30 34 Q27 70 30 96 Q32 120 36 136 M70 34 Q73 70 70 96 Q68 120 64 136"
        fill="none"
        stroke={BROWN}
        strokeOpacity="0.35"
        strokeWidth="1.6"
        strokeDasharray="3 2"
      />
      {/* Garment: narrower than the body and riding up, fronts not meeting */}
      <path d="M38 31 L47 33 L46 84 L34 88 Q33 60 34 38 Z" fill={BROWN} fillOpacity="0.55" />
      <path d="M62 31 L53 33 L54 84 L66 88 Q67 60 66 38 Z" fill={BROWN} fillOpacity="0.55" />
      <path d="M38 31 L28 36 L27 52 L33 51 Z M62 31 L72 36 L73 52 L67 51 Z" fill={BROWN} fillOpacity="0.45" />
      {/* Strain marks where this body is widest */}
      {ys.map((y) => (
        <g key={y} stroke={RED} strokeWidth="1.8" strokeLinecap="round">
          <line x1="40" y1={y} x2="46" y2={y - 2} />
          <line x1="54" y1={y - 2} x2="60" y2={y} />
          <line x1="40" y1={y + 4} x2="46" y2={y + 2} />
          <line x1="54" y1={y + 2} x2="60" y2={y + 4} />
        </g>
      ))}
    </svg>
  );
}

export default function WontFitNotice({ message, recommendedSize, fitCheck, illustrationOnly, onTryRecommended }: Props) {
  const shortfalls = Object.entries(fitCheck?.body_check || {})
    .filter(([, steps]) => steps > 0)
    .sort((a, b) => b[1] - a[1]);

  const button = recommendedSize ? (
    <button
      type="button"
      onClick={onTryRecommended}
      style={{
        fontFamily: "'Poppins', sans-serif",
        fontWeight: 600,
        fontSize: "1.1vh",
        color: "#FFFFFF",
        background: "linear-gradient(135deg, #B79579 0%, #5C4A3D 100%)",
        border: "none",
        borderRadius: "999vh",
        padding: "0.6vh 2vh",
        cursor: "pointer",
        outline: "none",
        textTransform: "uppercase",
        letterSpacing: "0.08vh",
        boxShadow: "0px 0.3vh 1vh rgba(92, 74, 61, 0.25)",
      }}
    >
      Try {recommendedSize}
    </button>
  ) : null;

  const text = (
    <span
      style={{
        fontFamily: "'Poppins', sans-serif",
        fontWeight: 700,
        fontSize: "1.35vh",
        color: RED,
        textAlign: "center",
        lineHeight: 1.25,
      }}
    >
      {message}
    </span>
  );

  if (!illustrationOnly) {
    // The render itself shows the strain: a banner over the bottom of the photo, the photo stays visible
    return (
      <div
        data-testid="wont-fit-banner"
        style={{
          position: "absolute",
          left: "1vh",
          right: "1vh",
          bottom: "1vh",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: "0.6vh",
          padding: "1vh 1.2vh",
          background: "rgba(255, 255, 255, 0.9)",
          border: "0.15vh solid rgba(192, 82, 82, 0.45)",
          borderRadius: "1.2vh",
          boxShadow: "0px 0.4vh 1.4vh rgba(160, 60, 60, 0.18)",
          zIndex: 16,
        }}
      >
        {text}
        {button}
      </div>
    );
  }

  return (
    <div
      data-testid="wont-fit-illustration"
      style={{
        position: "absolute",
        inset: 0,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: "1vh",
        padding: "0 2vh",
        background: "rgba(255, 255, 255, 0.88)",
        backdropFilter: "blur(0.6vh)",
        WebkitBackdropFilter: "blur(0.6vh)",
        zIndex: 16,
      }}
    >
      {text}
      <StrainIllustration fitCheck={fitCheck} />
      <span style={{ fontFamily: "'Poppins', sans-serif", fontSize: "0.85vh", color: "rgba(92, 74, 61, 0.6)" }}>
        Illustration — this size would strain and not close on your body
      </span>
      {shortfalls.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "0.2vh" }}>
          {shortfalls.map(([dim, steps]) => (
            <span key={dim} style={{ fontFamily: "'Poppins', sans-serif", fontSize: "1.0vh", color: BROWN }}>
              {`${DIMENSION_NAMES[dim] || dim}: about ${steps.toFixed(1)} size${steps >= 1.05 ? "s" : ""} too small`}
            </span>
          ))}
        </div>
      )}
      {button}
    </div>
  );
}

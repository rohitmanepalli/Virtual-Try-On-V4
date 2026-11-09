import { useEffect, useState } from "react";

type ProgressCircleProps = {
  onComplete?: () => void;
  durationMs?: number;
  apiDone?: boolean;
};

export default function ProgressCircle({
  onComplete,
  durationMs = 4000,
  apiDone = false,
}: ProgressCircleProps) {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (apiDone) {
      const delayTimer = setTimeout(() => setProgress(100), 0);
      return () => clearTimeout(delayTimer);
    }

    const stepTime = durationMs / 90;
    const timer = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 90) {
          clearInterval(timer);
          return 90;
        }
        return prev + 1;
      });
    }, stepTime);

    return () => clearInterval(timer);
  }, [apiDone, durationMs]);

  useEffect(() => {
    if (progress === 100 && onComplete) {
      const timeout = setTimeout(onComplete, 400);
      return () => clearTimeout(timeout);
    }
  }, [progress, onComplete]);

  // Internal SVG coordinate system (unitless — used for viewBox calculations only)
  const viewBoxSize = 300;
  const strokeWidth = 14;
  const radius = (viewBoxSize - strokeWidth * 2) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (progress / 100) * circumference;

  // Outer decorative ring
  const outerRadius = radius + 20;

  return (
    <div
      style={{
        position: "relative",
        width: "19vh",
        height: "19vh",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      {/* Outer faint decorative ring (Ellipse from CSS: opacity 0.05, border 23px solid white) */}
      <div
        style={{
          position: "absolute",
          width: "100%",
          height: "100%",
          borderRadius: "50%",
          border: "0.1vh solid rgba(255, 255, 255, 0.05)",
          boxSizing: "border-box",
        }}
      />

      {/* SVG Progress Ring */}
      <svg
        width="100%"
        height="100%"
        viewBox={`0 0 ${viewBoxSize} ${viewBoxSize}`}
        style={{ transform: "rotate(-90deg)", position: "absolute", inset: 0 }}
      >
        {/* Outer faint decorative circle */}
        <circle
          cx={viewBoxSize / 2}
          cy={viewBoxSize / 2}
          r={outerRadius}
          fill="transparent"
          stroke="rgba(255, 255, 255, 0.03)"
          strokeWidth={1.5}
        />

        {/* Background track ring */}
        <circle
          cx={viewBoxSize / 2}
          cy={viewBoxSize / 2}
          r={radius}
          fill="transparent"
          stroke="rgba(255, 255, 255, 0.08)"
          strokeWidth={strokeWidth}
        />

        {/* Active progress arc — Gradient from cyan to purple to pink */}
        <defs>
          <linearGradient id="progressGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#00C2FF" />
            <stop offset="50%" stopColor="#8A2BFF" />
            <stop offset="100%" stopColor="#FF2BD6" />
          </linearGradient>
        </defs>
        <circle
          cx={viewBoxSize / 2}
          cy={viewBoxSize / 2}
          r={radius}
          fill="none"
          stroke="url(#progressGradient)"
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          style={{
            transition: progress === 100
              ? "stroke-dashoffset 0.6s cubic-bezier(0.1, 0.8, 0.25, 1)"
              : "stroke-dashoffset 0.15s linear"
          }}
        />
      </svg>

      {/* Center Text — PROCESSING */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        <span
          style={{
            fontSize: "1.0vh",
            fontWeight: 600,
            color: "#FF2BD6",
            letterSpacing: "0.25vh",
            textTransform: "uppercase",
            fontFamily: "'Montserrat', sans-serif",
          }}
        >
          PROCESSING
        </span>
        {/* Small animated dots */}
        <span
          style={{
            fontSize: "1.26vh",
            color: "#FF2BD6",
            marginTop: "0.13vh",
            letterSpacing: "0.25vh",
          }}
        >
          ...
        </span>
      </div>
    </div>
  );
}

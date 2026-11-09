import React from "react";
import { type FitState } from "../context/TrialRoomContext";

type FitOption = {
  key: FitState;
  label: string;
  sublabel: string;
};

const OPTIONS: FitOption[] = [
  { key: "fitted", label: "Fitted", sublabel: "Tailored Cut" },
  { key: "standard", label: "Standard", sublabel: "Classic Retail" },
  { key: "relaxed", label: "Relaxed", sublabel: "Casual Drape" },
];

type FitSliderProps = {
  value: FitState;
  onChange: (fit: FitState) => void;
  disabled?: boolean;
  compact?: boolean;
  style?: React.CSSProperties;
};

export default function FitSlider({
  value,
  onChange,
  disabled = false,
  compact = false,
  style,
}: FitSliderProps) {
  const activeIndex = OPTIONS.findIndex((o) => o.key === value);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: compact ? "0.6vh" : "1.0vh",
        userSelect: "none",
        ...style,
      }}
    >
      {/* Title / Label */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "1.0vh",
          opacity: 0.9,
        }}
      >
        <div style={{ width: "1.8vh", height: "0.15vh", background: "#5C4A3D" }} />
        <span
          style={{
            fontFamily: "'Montserrat', sans-serif",
            fontWeight: 700,
            fontSize: compact ? "1.2vh" : "1.4vh",
            letterSpacing: "0.12vh",
            textTransform: "uppercase",
            color: "#5C4A3D",
          }}
        >
          FITFINDER STYLE
        </span>
        <div style={{ width: "1.8vh", height: "0.15vh", background: "#5C4A3D" }} />
      </div>

      {/* 3-Position Slider Pill Track */}
      <div
        style={{
          position: "relative",
          display: "flex",
          flexDirection: "row",
          alignItems: "center",
          width: compact ? "32vh" : "38vh",
          height: compact ? "5.0vh" : "6.2vh",
          background: "rgba(255, 255, 255, 0.45)",
          backdropFilter: "blur(1.2vh)",
          WebkitBackdropFilter: "blur(1.2vh)",
          border: "0.15vh solid rgba(183, 149, 121, 0.45)",
          borderRadius: "999vh",
          padding: "0.4vh",
          boxSizing: "border-box",
          boxShadow: "0px 0.8vh 2.4vh rgba(92, 74, 61, 0.08), inset 0px 0.1vh 0.2vh rgba(255, 255, 255, 0.6)",
          opacity: disabled ? 0.6 : 1,
          pointerEvents: disabled ? "none" : "auto",
        }}
      >
        {/* Animated Sliding Highlight Pill */}
        <div
          style={{
            position: "absolute",
            top: "0.4vh",
            bottom: "0.4vh",
            left: `calc(${activeIndex} * (100% - 0.8vh) / 3 + 0.4vh)`,
            width: "calc((100% - 0.8vh) / 3)",
            background: "linear-gradient(135deg, #B79579 0%, #90755E 60%, #5C4A3D 100%)",
            borderRadius: "999vh",
            boxShadow:
              "0px 0.4vh 1.2vh rgba(92, 74, 61, 0.25), 0px 0.1vh 0.4vh rgba(219, 184, 148, 0.4)",
            transition: "left 0.3s cubic-bezier(0.25, 1, 0.5, 1)",
            zIndex: 1,
          }}
        />

        {/* 3 Segment Buttons */}
        {OPTIONS.map((opt) => {
          const isSelected = opt.key === value;
          return (
            <button
              key={opt.key}
              type="button"
              onClick={() => onChange(opt.key)}
              style={{
                position: "relative",
                flex: 1,
                height: "100%",
                background: "transparent",
                border: "none",
                outline: "none",
                cursor: "pointer",
                zIndex: 2,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                padding: 0,
                transition: "color 0.25s ease",
              }}
            >
              <span
                style={{
                  fontFamily: "'Poppins', sans-serif",
                  fontWeight: isSelected ? 700 : 500,
                  fontSize: compact ? "1.2vh" : "1.4vh",
                  letterSpacing: "0.04vh",
                  color: isSelected ? "#FFFFFF" : "#5C4A3D",
                  transition: "color 0.2s ease, transform 0.2s ease",
                  transform: isSelected ? "scale(1.02)" : "scale(1)",
                  textTransform: "uppercase",
                  lineHeight: 1.1,
                }}
              >
                {opt.label}
              </span>
              {!compact && (
                <span
                  style={{
                    fontFamily: "'Poppins', sans-serif",
                    fontWeight: 400,
                    fontSize: "0.95vh",
                    color: isSelected ? "rgba(255, 255, 255, 0.85)" : "rgba(92, 74, 61, 0.55)",
                    marginTop: "0.2vh",
                    lineHeight: 1.0,
                  }}
                >
                  {opt.sublabel}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}

import React from "react";
import type { SizeFit } from "../context/TrialRoomContext";
import { FIT_LABELS, getSizeQuality } from "../utils/sizeFit";

const DEFAULT_SIZES = ["XS", "S", "M", "L", "XL", "XXL"];

type SizeSelectorProps = {
  availableSizes?: string[];
  selectedSize: string | null;
  recommendedSize: string | null;
  sizeFitQuality?: string | null;
  sizeConfidence?: number | null;
  sizeFitNotes?: string | null;
  // Backend fit per size (body-checked: a smaller size can be "won't fit" for one visitor and "tight" for another)
  sizeFits?: Record<string, SizeFit>;
  onChange: (size: string) => void;
  disabled?: boolean;
  compact?: boolean;
  style?: React.CSSProperties;
};

const PILL_LABELS: Record<string, string> = {
  oversized: "Oversized",
  loose: "Loose",
  relaxed: "Relaxed",
  snug: "Snug",
  tight: "Tight",
  does_not_fit: "Won't Fit",
};


/**
 * Returns the pill background, text, and border styling based on fit quality and selection state.
 */
function getQualityColors(quality: string, isSelected: boolean) {
  if (isSelected) {
    switch (quality) {
      case "perfect":
        return {
          bg: "linear-gradient(135deg, #B79579 0%, #90755E 60%, #5C4A3D 100%)",
          text: "#FFFFFF",
          border: "rgba(183, 149, 121, 0.6)",
          shadow: "0px 0.4vh 1.4vh rgba(92, 74, 61, 0.35)",
        };
      case "snug":
        return {
          bg: "linear-gradient(135deg, #D4956A 0%, #B87A4F 100%)",
          text: "#FFFFFF",
          border: "rgba(212, 149, 106, 0.6)",
          shadow: "0px 0.4vh 1.4vh rgba(184, 122, 79, 0.35)",
        };
      case "tight":
        return {
          bg: "linear-gradient(135deg, #C97B4E 0%, #A55F36 100%)",
          text: "#FFFFFF",
          border: "rgba(201, 123, 78, 0.6)",
          shadow: "0px 0.4vh 1.4vh rgba(165, 95, 54, 0.35)",
        };
      case "relaxed":
        return {
          bg: "linear-gradient(135deg, #9BAABD 0%, #7A8DA3 100%)",
          text: "#FFFFFF",
          border: "rgba(155, 170, 189, 0.6)",
          shadow: "0px 0.4vh 1.4vh rgba(122, 141, 163, 0.35)",
        };
      case "oversized":
        return {
          bg: "linear-gradient(135deg, #6B7F96 0%, #4A5D72 100%)",
          text: "#FFFFFF",
          border: "rgba(107, 127, 150, 0.6)",
          shadow: "0px 0.4vh 1.4vh rgba(74, 93, 114, 0.35)",
        };
      case "does_not_fit":
        return {
          bg: "linear-gradient(135deg, #C05252 0%, #A03C3C 100%)",
          text: "#FFFFFF",
          border: "rgba(192, 82, 82, 0.6)",
          shadow: "0px 0.4vh 1.4vh rgba(160, 60, 60, 0.35)",
        };
      case "loose":
        return {
          bg: "linear-gradient(135deg, #8B9DB3 0%, #6B7F96 100%)",
          text: "#FFFFFF",
          border: "rgba(139, 157, 179, 0.6)",
          shadow: "0px 0.4vh 1.4vh rgba(107, 127, 150, 0.35)",
        };
      default:
        return {
          bg: "linear-gradient(135deg, #B79579 0%, #5C4A3D 100%)",
          text: "#FFFFFF",
          border: "rgba(183, 149, 121, 0.6)",
          shadow: "0px 0.4vh 1.4vh rgba(92, 74, 61, 0.3)",
        };
    }
  }

  // Unselected states
  switch (quality) {
    case "does_not_fit":
      return {
        bg: "rgba(192, 82, 82, 0.08)",
        text: "rgba(192, 82, 82, 0.7)",
        border: "rgba(192, 82, 82, 0.2)",
        shadow: "none",
      };
    default:
      return {
        bg: "transparent",
        text: "#5C4A3D",
        border: "transparent",
        shadow: "none",
      };
  }
}

export default function SizeSelector({
  availableSizes,
  selectedSize,
  recommendedSize,
  sizeFitQuality,
  sizeConfidence,
  sizeFitNotes,
  sizeFits,
  onChange,
  disabled = false,
  compact = false,
  style,
}: SizeSelectorProps) {
  const sizes = availableSizes && availableSizes.length > 0 ? availableSizes : DEFAULT_SIZES;
  const selectedQuality = selectedSize ? getSizeQuality(selectedSize, recommendedSize, sizes, sizeFits) : "unknown";
  const selectedFitLabel = selectedSize
    ? sizeFits?.[selectedSize]?.fit_label || FIT_LABELS[selectedQuality] || null
    : null;
  const selectedColors = getQualityColors(selectedQuality, true);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: compact ? "0.6vh" : "0.9vh",
        userSelect: "none",
        ...style,
      }}
    >
      {/* Title / Label Header matching FitSlider styling */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "1.0vh",
          opacity: 0.92,
        }}
      >
        <div style={{ width: "1.8vh", height: "0.15vh", background: "#5C4A3D" }} />
        <span
          style={{
            fontFamily: "'Montserrat', sans-serif",
            fontWeight: 700,
            fontSize: compact ? "1.2vh" : "1.35vh",
            letterSpacing: "0.12vh",
            textTransform: "uppercase",
            color: "#5C4A3D",
          }}
        >
          SELECT SIZE
        </span>
        <div style={{ width: "1.8vh", height: "0.15vh", background: "#5C4A3D" }} />
      </div>

      {/* 6-Position Pill Track matching FitSlider style */}
      <div
        style={{
          position: "relative",
          display: "flex",
          flexDirection: "row",
          alignItems: "center",
          justifyContent: "space-between",
          width: compact ? "40vh" : "46vh",
          height: compact ? "5.4vh" : "6.2vh",
          background: "rgba(255, 255, 255, 0.45)",
          backdropFilter: "blur(1.2vh)",
          WebkitBackdropFilter: "blur(1.2vh)",
          border: "0.15vh solid rgba(183, 149, 121, 0.45)",
          borderRadius: "999vh",
          padding: "0.4vh",
          boxSizing: "border-box",
          boxShadow:
            "0px 0.8vh 2.4vh rgba(92, 74, 61, 0.08), inset 0px 0.1vh 0.2vh rgba(255, 255, 255, 0.6)",
          opacity: disabled ? 0.6 : 1,
          pointerEvents: disabled ? "none" : "auto",
        }}
      >
        {sizes.map((size) => {
          const isSelected = size === selectedSize;
          const isRecommended = size === recommendedSize;
          const quality = getSizeQuality(size, recommendedSize, sizes, sizeFits);
          const colors = getQualityColors(quality, isSelected);

          return (
            <div
              key={size}
              style={{
                position: "relative",
                flex: 1,
                height: "100%",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              {/* Visually mark the auto-detected size as "Recommended" badge above the pill */}
              {isRecommended && (
                <div
                  style={{
                    position: "absolute",
                    top: "-1.85vh",
                    background: "linear-gradient(135deg, #B79579 0%, #5C4A3D 100%)",
                    color: "#FFFFFF",
                    fontFamily: "'Montserrat', sans-serif",
                    fontWeight: 800,
                    fontSize: "0.72vh",
                    letterSpacing: "0.06vh",
                    padding: "0.2vh 0.65vh",
                    borderRadius: "999vh",
                    whiteSpace: "nowrap",
                    boxShadow: "0px 0.2vh 0.6vh rgba(92, 74, 61, 0.25)",
                    pointerEvents: "none",
                    textTransform: "uppercase",
                    zIndex: 5,
                  }}
                >
                  RECOMMENDED
                </div>
              )}

              {/* Pill Button */}
              <button
                type="button"
                onClick={() => onChange(size)}
                style={{
                  position: "relative",
                  width: "100%",
                  height: "100%",
                  background: colors.bg,
                  border: isSelected
                    ? `0.15vh solid ${colors.border}`
                    : `0.1vh solid ${colors.border}`,
                  borderRadius: "999vh",
                  cursor: "pointer",
                  outline: "none",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  padding: "0.2vh 0",
                  transition: "all 0.25s cubic-bezier(0.25, 1, 0.5, 1)",
                  transform: isSelected ? "scale(1.04)" : "scale(1)",
                  boxShadow: colors.shadow,
                  zIndex: 2,
                }}
              >
                <span
                  style={{
                    fontFamily: "'Poppins', sans-serif",
                    fontWeight: isSelected ? 800 : 600,
                    fontSize: compact ? "1.2vh" : "1.4vh",
                    letterSpacing: "0.03vh",
                    color: colors.text,
                    lineHeight: 1.1,
                    transition: "color 0.2s ease, transform 0.2s ease",
                  }}
                >
                  {size}
                </span>

                {/* Micro Sublabel on selected pill */}
                {isSelected && (
                  <span
                    style={{
                      fontFamily: "'Poppins', sans-serif",
                      fontWeight: 500,
                      fontSize: "0.68vh",
                      color: "rgba(255, 255, 255, 0.9)",
                      lineHeight: 1.0,
                      marginTop: "0.1vh",
                      textTransform: "capitalize",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {isRecommended ? "✓ Your Size" : PILL_LABELS[quality] || ""}
                  </span>
                )}
              </button>
            </div>
          );
        })}
      </div>

      {/* Fit label for the selected size */}
      {selectedFitLabel && (
        <span
          data-testid="size-fit-label"
          style={{
            fontFamily: "'Poppins', sans-serif",
            fontWeight: 600,
            fontSize: compact ? "1.05vh" : "1.2vh",
            letterSpacing: "0.04vh",
            color: "#FFFFFF",
            background: selectedColors.bg,
            borderRadius: "999vh",
            padding: "0.25vh 1.2vh",
            marginTop: "0.2vh",
            boxShadow: selectedColors.shadow,
            whiteSpace: "nowrap",
          }}
        >
          {`Size ${selectedSize}: ${selectedFitLabel}${selectedSize === recommendedSize ? " — your size" : ""}`}
        </span>
      )}

      {/* Helpful Fit Notes below pill track */}
      {sizeFitNotes && (
        <span
          style={{
            fontFamily: "'Poppins', sans-serif",
            fontWeight: 400,
            fontSize: "0.85vh",
            color: "rgba(92, 74, 61, 0.65)",
            textAlign: "center",
            maxWidth: "42vh",
            lineHeight: 1.3,
            marginTop: "0.1vh",
          }}
        >
          {sizeFitNotes}
        </span>
      )}
    </div>
  );
}

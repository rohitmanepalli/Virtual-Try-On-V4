import React from "react";

type OutfitCardProps = {
  imageSrc: string;
  label: string;
  category?: string;
  selected?: boolean;
  onClick?: () => void;
  style?: React.CSSProperties;
};

/**
 * Premium glassmorphic outfit thumbnail card with a gold/espresso
 * highlight border when selected. Used in OutfitSelectionPage.
 */
export default function OutfitCard({
  imageSrc,
  label,
  selected = false,
  onClick,
  style,
}: OutfitCardProps) {
  return (
    <button
      onClick={onClick}
      aria-label={label}
      style={{
        width: "8.2vh",
        height: "9.6vh",
        background: selected
          ? "linear-gradient(145deg, rgba(255, 255, 255, 0.5), rgba(243, 229, 216, 0.4))"
          : "rgba(255, 255, 255, 0.18)",
        border: selected
          ? "0.22vh solid #5C4A3D"
          : "0.12vh solid rgba(92, 74, 61, 0.2)",
        boxShadow: selected
          ? "0px 0.6vh 1.8vh rgba(92, 74, 61, 0.28), 0 0 1vh rgba(219, 184, 148, 0.4)"
          : "0px 0.3vh 0.8vh rgba(0, 0, 0, 0.07)",
        backdropFilter: "blur(0.8vh)",
        WebkitBackdropFilter: "blur(0.8vh)",
        borderRadius: "1.6vh",
        cursor: "pointer",
        padding: "0.5vh 0.4vh",
        display: "flex",
        flexDirection: "column",
        justifyContent: "space-between",
        alignItems: "center",
        outline: "none",
        transition: "all 0.25s cubic-bezier(0.4, 0, 0.2, 1)",
        flexShrink: 0,
        transform: selected ? "scale(1.05)" : "scale(1)",
        position: "relative",
        ...style,
      }}
      onMouseDown={(e) => {
        e.currentTarget.style.transform = selected ? "scale(1.02)" : "scale(0.96)";
      }}
      onMouseUp={(e) => {
        e.currentTarget.style.transform = selected ? "scale(1.05)" : "scale(1)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = selected ? "scale(1.05)" : "scale(1)";
      }}
    >
      {/* Selected Indicator Dot */}
      {selected && (
        <span
          style={{
            position: "absolute",
            top: "0.5vh",
            right: "0.5vh",
            width: "0.7vh",
            height: "0.7vh",
            borderRadius: "50%",
            backgroundColor: "#5C4A3D",
            boxShadow: "0 0 0.4vh #DBB894",
          }}
        />
      )}

      {/* Garment Image */}
      <div
        style={{
          width: "100%",
          height: "6.4vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <img
          src={imageSrc}
          alt={label}
          style={{
            maxWidth: "100%",
            maxHeight: "100%",
            objectFit: "contain",
            filter: selected
              ? "drop-shadow(0 0.3vh 0.6vh rgba(92, 74, 61, 0.25))"
              : "drop-shadow(0 0.2vh 0.4vh rgba(0, 0, 0, 0.15))",
            transition: "transform 0.25s ease",
          }}
        />
      </div>

      {/* Label */}
      <span
        style={{
          fontFamily: "'Montserrat', sans-serif",
          fontSize: "0.8vh",
          fontWeight: selected ? 800 : 600,
          lineHeight: "1.1vh",
          color: selected ? "#5C4A3D" : "rgba(92, 74, 61, 0.75)",
          textAlign: "center",
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis",
          maxWidth: "7.4vh",
          letterSpacing: "0.03vh",
          textTransform: "uppercase",
        }}
      >
        {label}
      </span>
    </button>
  );
}

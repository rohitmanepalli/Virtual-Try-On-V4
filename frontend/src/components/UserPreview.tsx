import React from "react";

type UserPreviewProps = {
  imageSrc: string;
  alt?: string;
  style?: React.CSSProperties;
};

/**
 * Reusable white-bordered rounded card that displays
 * the captured user image. Used across Confirmation,
 * OutfitSelection, and Result screens.
 */
export default function UserPreview({
  imageSrc,
  alt = "User Preview",
  style,
}: UserPreviewProps) {
  return (
    <div
      style={{
        width: "82%",
        background: "#FFFFFF",
        border: "0.19vh solid #FFFFFF",
        borderRadius: "1.0vh",
        overflow: "hidden",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        boxShadow: "0 0.25vh 1.26vh rgba(0, 0, 0, 0.08)",
        position: "relative",
        ...style,
      }}
    >
      <img
        src={imageSrc}
        alt={alt}
        style={{
          width: "100%",
          height: "100%",
          objectFit: "contain",
        }}
      />
    </div>
  );
}

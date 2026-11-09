import React from "react";

type HeaderProps = {
  text?: string;
  statusText?: string;
  showStatusIndicator?: boolean;
  style?: React.CSSProperties;
};

export default function Header({
  text,
  statusText,
  showStatusIndicator = false,
  style,
}: HeaderProps) {
  const containerStyle: React.CSSProperties = {
    width: "100%",
    padding: "1.26vh 1.0vh",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    pointerEvents: "none",
    ...style,
  };

  if (showStatusIndicator && statusText) {
    return (
      <div style={containerStyle}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.5vh",
            justifyContent: "center",
          }}
        >
          {/* Green Status LED Dot */}
          <div
            style={{
              width: "0.76vh",
              height: "0.76vh",
              borderRadius: "50%",
              backgroundColor: "#2ECC71", // Neon Green
              boxShadow: "0 0 0.25vh rgba(46, 204, 113, 0.4)",
            }}
          />
          <span
            style={{
              fontSize: "0.82vh",
              fontWeight: 500,
              color: "#666666",
              letterSpacing: "0.03vh",
            }}
          >
            {statusText}
          </span>
        </div>
      </div>
    );
  }

  if (text) {
    return (
      <div style={containerStyle}>
        <h2
          style={{
            fontSize: "1.0vh",
            fontWeight: 800,
            color: "#1E2022",
            textAlign: "center",
            textTransform: "uppercase",
            letterSpacing: "0.05vh",
            lineHeight: 1.4,
            margin: 0,
            width: "90%",
          }}
        >
          {text}
        </h2>
      </div>
    );
  }

  return null;
}

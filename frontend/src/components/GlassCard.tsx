import React from "react";

type CardProps = {
  children: React.ReactNode;
  style?: React.CSSProperties;
  className?: string;
};

export default function GlassCard({
  children,
  style,
  className = "",
}: CardProps) {
  const cardStyle: React.CSSProperties = {
    boxSizing: "border-box",
    width: "32.95vh",
    height: "21.05vh",
    background: "rgba(247, 242, 234, 0.9)",
    border: "0.14vh solid rgba(255, 255, 255, 0.15)",
    boxShadow: "0px 0.25vh 3.15vh rgba(0, 0, 0, 0.25), 0px 0.18vh 0.9vh rgba(0, 0, 0, 0.25), 0px 0.18vh 1.09vh rgba(0, 0, 0, 0.25)",
    borderRadius: "0.72vh",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    ...style,
  };

  return (
    <div style={cardStyle} className={`figma-card ${className}`}>
      {children}
    </div>
  );
}

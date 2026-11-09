import React from "react";

type ButtonProps = {
  children: React.ReactNode;
  onClick?: React.MouseEventHandler<HTMLButtonElement>;
  variant?: "primary" | "secondary" | "welcome";
  style?: React.CSSProperties;
  className?: string;
  disabled?: boolean;
};

export default function PrimaryButton({
  children,
  onClick,
  variant = "primary",
  style,
  className = "",
  disabled = false,
}: ButtonProps) {
  const getBackgroundColor = () => {
    if (disabled) return "#d1d5db";
    if (variant === "welcome") return "linear-gradient(90deg, #B79579 0%, #90755E 48.08%, #CFB499 100%)";
    if (variant === "primary") return "linear-gradient(90deg, #00C2FF 0%, #8A2BFF 50%, #FF2BD6 100%)";
    return "rgba(30, 30, 30, 0.9)"; // secondary variant
  };

  const getBoxShadow = () => {
    if (disabled) return "none";
    if (variant === "welcome") return "0px 0.25vh 0.25vh rgba(0, 0, 0, 0.25), 0px 2.27vh 4.54vh -1.09vh rgba(0, 0, 0, 0.25)";
    if (variant === "primary") return "0px 0px 1.5vh rgba(255, 43, 214, 0.55)";
    return "0px 0px 1.5vh rgba(255, 43, 214, 0.3)";
  };

  const getBorder = () => {
    if (variant === "welcome") return "0.09vh solid #DBB894";
    return "none";
  };

  const getBorderRadius = () => {
    if (variant === "welcome") return "9999px";
    return "1.0vh";
  };

  const getWidth = () => {
    if (variant === "welcome") return "28.85vh"; // ~457.92px scaled to vh
    return "29.17vh";
  };

  const getHeight = () => {
    if (variant === "welcome") return "5.8vh"; // ~92.16px scaled to vh
    return "3.97vh";
  };

  const buttonStyle: React.CSSProperties = {
    background: getBackgroundColor(),
    color: "#F5F7FF",
    width: getWidth(),
    height: getHeight(),
    borderRadius: getBorderRadius(),
    border: getBorder(),
    boxShadow: getBoxShadow(),
    cursor: disabled ? "not-allowed" : "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontFamily: "'Poppins', sans-serif",
    fontWeight: variant === "welcome" ? 700 : 600,
    fontSize: variant === "welcome" ? "1.45vh" : "1.63vh",
    lineHeight: variant === "welcome" ? "1.45vh" : "2.45vh",
    letterSpacing: variant === "welcome" ? "0.22vh" : "0.24vh",
    textTransform: "uppercase",
    transition: "transform 0.1s ease",
    outline: "none",
    gap: variant === "welcome" ? "1.45vh" : undefined,
    padding: variant === "welcome" ? "0px" : undefined,
    ...style,
  };

  const renderChildren = () => {
    if (variant === "welcome") {
      return (
        <>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              padding: "0px",
              width: "auto",
              height: "auto",
              flex: "none",
              order: 1,
              flexGrow: 0,
              zIndex: 1,
            }}
          >
            <span
              style={{
                fontFamily: "'Poppins', sans-serif",
                fontStyle: "normal",
                fontWeight: 700,
                fontSize: "1.45vh", // ~23.04px
                lineHeight: "1.45vh", // ~23px
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                textAlign: "center",
                letterSpacing: "0.22vh", // ~3.456px
                textTransform: "uppercase",
                color: "#FFFFFF",
                whiteSpace: "nowrap",
                flex: "none",
                order: 0,
                flexGrow: 0,
              }}
            >
              {children}
            </span>
          </div>
          <svg
            viewBox="0 0 36 36"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            style={{
              width: "2.27vh", // ~36px
              height: "2.27vh", // ~36px
              flex: "none",
              order: 2,
              flexGrow: 0,
              zIndex: 2,
            }}
          >
            <path
              d="M4.5 18H31.5M31.5 18L20.25 6.75M31.5 18L20.25 29.25"
              stroke="#FFFFFF"
              strokeWidth="3.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </>
      );
    }
    return children;
  };

  return (
    <button
      onClick={!disabled ? onClick : undefined}
      style={buttonStyle}
      disabled={disabled}
      className={`primary-btn ${className}`}
      onMouseDown={(e) => {
        if (!disabled) {
          e.currentTarget.style.transform = "scale(0.97)";
        }
      }}
      onMouseUp={(e) => {
        if (!disabled) {
          e.currentTarget.style.transform = "scale(1)";
        }
      }}
      onMouseLeave={(e) => {
        if (!disabled) {
          e.currentTarget.style.transform = "scale(1)";
        }
      }}
    >
      {renderChildren()}
    </button>
  );
}


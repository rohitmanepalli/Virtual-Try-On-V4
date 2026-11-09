import React from "react";

type NavProps = {
  type: "back" | "home";
  onClick?: () => void;
  style?: React.CSSProperties;
  className?: string;
  disabled?: boolean;
};

export default function NavigationButton({
  type,
  onClick,
  style,
  className = "",
  disabled = false,
}: NavProps) {
  const isBack = type === "back";

  // For home button pass through style as-is (caller controls size)
  const cleanedStyle = style;

  const buttonStyle: React.CSSProperties = isBack
    ? {
        width: "4.4vh",
        height: "4.6vh",
        background: "#05060B",
        border: "0.063vh solid rgba(0, 194, 255, 0.4)",
        borderRadius: "1.0vh",
        boxShadow: "0px 0px 1.0vh rgba(0, 194, 255, 0.25)",
        cursor: disabled ? "not-allowed" : "pointer",
        display: "inline-flex",
        justifyContent: "center",
        alignItems: "center",
        color: "#00C2FF",
        filter: "drop-shadow(0px 0px 0.5vh rgba(0, 194, 255, 0.45))",
        transition: "background-color 0.2s ease, transform 0.1s ease, box-shadow 0.2s ease",
        outline: "none",
        ...cleanedStyle,
      }
    : {
        width: "5.6vh",
        height: "5.6vh",
        background: "transparent",
        border: "none",
        borderRadius: "50%",
        boxShadow: "none",
        cursor: disabled ? "not-allowed" : "pointer",
        display: "inline-flex",
        justifyContent: "center",
        alignItems: "center",
        outline: "none",
        transition: "transform 0.1s ease",
        ...cleanedStyle,
      };

  return (
    <button
      onClick={!disabled ? onClick : undefined}
      style={buttonStyle}
      disabled={disabled}
      className={`nav-btn-${type} ${className}`}
      onMouseDown={(e) => {
        if (!disabled) {
          e.currentTarget.style.transform = "scale(0.95)";
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
      {isBack ? (
        // Clean Back Arrow
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          style={{
            width: "50%",
            height: "50%",
          }}
        >
          <line x1="19" y1="12" x2="5" y2="12"></line>
          <polyline points="12 19 5 12 12 5"></polyline>
        </svg>
      ) : (
        // New Home Icon SVG
        <svg
          viewBox="20 -10 152 132"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          style={{
            width: "100%",
            height: "100%",
          }}
        >
          <rect x="40" width="112" height="112" rx="56" fill="#5C4A3D" />
          <g filter="url(#filter0_dd_155_536)">
            <rect
              x="40"
              width="112"
              height="112"
              rx="56"
              fill="white"
              fillOpacity="0.01"
              shapeRendering="crispEdges"
            />
          </g>
          <rect x="40" width="112" height="112" rx="56" fill="#5C4A3D" />
          <g filter="url(#filter1_ddd_155_536)">
            <rect
              x="40"
              width="112"
              height="112"
              rx="56"
              fill="white"
              fillOpacity="0.01"
              shapeRendering="crispEdges"
            />
            <rect
              x="40.5"
              y="0.5"
              width="111"
              height="111"
              rx="55.5"
              stroke="#CFB49C"
              shapeRendering="crispEdges"
            />
          </g>
          <path
            d="M114.387 51.673L98.7617 36.048C98.0292 35.3157 97.0358 34.9043 96 34.9043C94.9642 34.9043 93.9708 35.3157 93.2383 36.048L77.6133 51.673C77.2491 52.035 76.9604 52.4657 76.7639 52.9401C76.5674 53.4145 76.4671 53.9232 76.4688 54.4367V73.1867C76.4688 73.8083 76.7157 74.4044 77.1552 74.8439C77.5948 75.2835 78.1909 75.5304 78.8125 75.5304H91.3125C91.9341 75.5304 92.5303 75.2835 92.9698 74.8439C93.4093 74.4044 93.6563 73.8083 93.6563 73.1867V63.0304H98.3438V73.1867C98.3438 73.8083 98.5907 74.4044 99.0302 74.8439C99.4698 75.2835 100.066 75.5304 100.688 75.5304H113.188C113.809 75.5304 114.405 75.2835 114.845 74.8439C115.284 74.4044 115.531 73.8083 115.531 73.1867V54.4367C115.533 53.9232 115.433 53.4145 115.236 52.9401C115.04 52.4657 114.751 52.035 114.387 51.673ZM110.844 70.8429H103.031V60.6867C103.031 60.0651 102.784 59.4689 102.345 59.0294C101.905 58.5898 101.309 58.3429 100.688 58.3429H91.3125C90.6909 58.3429 90.0948 58.5898 89.6552 59.0294C89.2157 59.4689 88.9688 60.0651 88.9688 60.6867V70.8429H81.1563V54.7589L96 39.9152L110.844 54.7589V70.8429Z"
            fill="white"
          />
          <defs>
            <filter
              id="filter0_dd_155_536"
              x="0"
              y="0"
              width="192"
              height="192"
              filterUnits="userSpaceOnUse"
              colorInterpolationFilters="sRGB"
            >
              <feFlood floodOpacity="0" result="BackgroundImageFix" />
              <feColorMatrix
                in="SourceAlpha"
                type="matrix"
                values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0"
                result="hardAlpha"
              />
              <feMorphology
                radius="12"
                operator="erode"
                in="SourceAlpha"
                result="effect1_dropShadow_155_536"
              />
              <feOffset dy="16" />
              <feGaussianBlur stdDeviation="10" />
              <feComposite in2="hardAlpha" operator="out" />
              <feColorMatrix
                type="matrix"
                values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.1 0"
              />
              <feBlend
                mode="normal"
                in2="BackgroundImageFix"
                result="effect1_dropShadow_155_536"
              />
              <feColorMatrix
                in="SourceAlpha"
                type="matrix"
                values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0"
                result="hardAlpha"
              />
              <feMorphology
                radius="10"
                operator="erode"
                in="SourceAlpha"
                result="effect2_dropShadow_155_536"
              />
              <feOffset dy="40" />
              <feGaussianBlur stdDeviation="25" />
              <feComposite in2="hardAlpha" operator="out" />
              <feColorMatrix
                type="matrix"
                values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.1 0"
              />
              <feBlend
                mode="normal"
                in2="effect1_dropShadow_155_536"
                result="effect2_dropShadow_155_536"
              />
              <feBlend
                mode="normal"
                in="SourceGraphic"
                in2="effect2_dropShadow_155_536"
                result="shape"
              />
            </filter>
            <filter
              id="filter1_ddd_155_536"
              x="0"
              y="0"
              width="192"
              height="192"
              filterUnits="userSpaceOnUse"
              colorInterpolationFilters="sRGB"
            >
              <feFlood floodOpacity="0" result="BackgroundImageFix" />
              <feColorMatrix
                in="SourceAlpha"
                type="matrix"
                values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0"
                result="hardAlpha"
              />
              <feMorphology
                radius="12"
                operator="erode"
                in="SourceAlpha"
                result="effect1_dropShadow_155_536"
              />
              <feOffset dy="16" />
              <feGaussianBlur stdDeviation="10" />
              <feComposite in2="hardAlpha" operator="out" />
              <feColorMatrix
                type="matrix"
                values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.1 0"
              />
              <feBlend
                mode="normal"
                in2="BackgroundImageFix"
                result="effect1_dropShadow_155_536"
              />
              <feColorMatrix
                in="SourceAlpha"
                type="matrix"
                values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0"
                result="hardAlpha"
              />
              <feMorphology
                radius="10"
                operator="erode"
                in="SourceAlpha"
                result="effect2_dropShadow_155_536"
              />
              <feOffset dy="40" />
              <feGaussianBlur stdDeviation="25" />
              <feComposite in2="hardAlpha" operator="out" />
              <feColorMatrix
                type="matrix"
                values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.1 0"
              />
              <feBlend
                mode="normal"
                in2="effect1_dropShadow_155_536"
                result="effect2_dropShadow_155_536"
              />
              <feColorMatrix
                in="SourceAlpha"
                type="matrix"
                values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0"
                result="hardAlpha"
              />
              <feOffset dy="4" />
              <feGaussianBlur stdDeviation="2" />
              <feComposite in2="hardAlpha" operator="out" />
              <feColorMatrix
                type="matrix"
                values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.25 0"
              />
              <feBlend
                mode="normal"
                in2="effect2_dropShadow_155_536"
                result="effect3_dropShadow_155_536"
              />
              <feBlend
                mode="normal"
                in="SourceGraphic"
                in2="effect3_dropShadow_155_536"
                result="shape"
              />
            </filter>
          </defs>
        </svg>
      )}
    </button>
  );
}

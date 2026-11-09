export default function FashionScanner() {
  return (
    <div
      style={{
        position: "relative",
        width: "30vh",
        height: "32.8vh",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <style>{`
        @keyframes svgRotateSlow {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        
        @keyframes svgCorePulse {
          0%, 100% { opacity: 0.6; transform: scale(0.9); }
          50% { opacity: 1; transform: scale(1.15); }
        }
        
        @keyframes svgDotPulse {
          0%, 100% { opacity: 0.3; transform: scale(0.8); }
          20% { opacity: 1; transform: scale(1.3); }
          50% { opacity: 0.8; transform: scale(1); }
        }

        .svg-center {
          transform-box: fill-box;
          transform-origin: center;
        }
      `}</style>

      <svg
        width="100%"
        height="100%"
        viewBox="-30 -10 580 630"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        style={{ overflow: "visible", zIndex: 1 }}
        aria-hidden="true"
      >
        <g filter="url(#filter0_d_197_138)">

          {/* Central glow — pulsing */}
          <g
            className="svg-center"
            filter="url(#filter1_df_197_138)"
            style={{
              animation: "svgCorePulse 2.4s cubic-bezier(0.25, 0.1, 0.25, 1) infinite",
            }}
          >
            <rect x="247.5" y="271.689" width="24.9057" height="24.9057" rx="12.4528" fill="white" fillOpacity="0.9" shapeRendering="crispEdges" />
          </g>

          {/* Innermost dashed ring — rotating */}
          <g
            className="svg-center"
            filter="url(#filter2_i_197_138)"
            style={{
              animation: "svgRotateSlow 14s linear infinite",
            }}
          >
            <rect x="155.849" y="179.849" width="208.585" height="208.585" rx="104.292" stroke="#5C4A3D" strokeOpacity="0.4" strokeWidth="9.33962" strokeDasharray="4.67 3.11" />
          </g>

          {/* Second ring (static) + corner dot (pulsing, no filter/background) */}
          <g filter="url(#filter3_dd_197_138)">
            <rect x="123.16" y="147.16" width="273.962" height="273.962" rx="136.981" stroke="#5C4A3D" strokeOpacity="0.25" strokeWidth="6.22642" shapeRendering="crispEdges" />
            <g
              className="svg-center"
              style={{
                animation: "svgDotPulse 2.8s ease-in-out infinite 0s",
              }}
            >
              <rect x="253.915" y="150.273" width="12.4528" height="12.4528" rx="6.22642" fill="#5C4A3D" />
            </g>
          </g>

          {/* Third ring (static) + corner dot (pulsing, no filter/background) */}
          <g filter="url(#filter5_dd_197_138)">
            <rect x="76.4618" y="100.463" width="367.358" height="367.358" rx="183.679" stroke="#5C4A3D" strokeOpacity="0.15" strokeWidth="6.22642" shapeRendering="crispEdges" />
            <g
              className="svg-center"
              style={{
                animation: "svgDotPulse 2.8s ease-in-out infinite 0.9s",
              }}
            >
              <rect x="428.254" y="360.416" width="12.4528" height="12.4528" rx="6.22642" fill="#5C4A3D" />
            </g>
          </g>

          {/* Outermost ring (static) + corner dot (pulsing, no filter/background) */}
          <g filter="url(#filter7_dd_197_138)">
            <rect x="29.7641" y="53.7636" width="460.755" height="460.755" rx="230.377" stroke="#5C4A3D" strokeOpacity="0.08" strokeWidth="6.22642" shapeRendering="crispEdges" />
            <g
              className="svg-center"
              style={{
                animation: "svgDotPulse 2.8s ease-in-out infinite 1.8s",
              }}
            >
              <rect x="32.8774" y="211.494" width="15.566" height="15.566" rx="7.78302" fill="#5C4A3D" />
            </g>
          </g>
        </g>

        <defs>
          <filter id="filter0_d_197_138" x="-22.9" y="-0.9" width="563.706" height="612.083" filterUnits="userSpaceOnUse" colorInterpolationFilters="sRGB">
            <feFlood floodOpacity="0" result="BackgroundImageFix" />
            <feColorMatrix in="SourceAlpha" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" result="hardAlpha" />
            <feOffset dx="-1" dy="21" />
            <feGaussianBlur stdDeviation="10.95" />
            <feComposite in2="hardAlpha" operator="out" />
            <feColorMatrix type="matrix" values="0 0 0 0 0.278846 0 0 0 0 0.278846 0 0 0 0 0.278846 0 0 0 0.25 0" />
            <feBlend mode="multiply" in2="BackgroundImageFix" result="effect1_dropShadow_197_138" />
            <feBlend mode="normal" in="SourceGraphic" in2="effect1_dropShadow_197_138" result="shape" />
          </filter>
          <filter id="filter1_df_197_138" x="224.151" y="248.34" width="71.6039" height="71.6044" filterUnits="userSpaceOnUse" colorInterpolationFilters="sRGB">
            <feFlood floodOpacity="0" result="BackgroundImageFix" />
            <feColorMatrix in="SourceAlpha" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" result="hardAlpha" />
            <feOffset />
            <feGaussianBlur stdDeviation="11.6745" />
            <feComposite in2="hardAlpha" operator="out" />
            <feColorMatrix type="matrix" values="0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0" />
            <feBlend mode="normal" in2="BackgroundImageFix" result="effect1_dropShadow_197_138" />
            <feBlend mode="normal" in="SourceGraphic" in2="effect1_dropShadow_197_138" result="shape" />
            <feGaussianBlur stdDeviation="1.5566" result="effect2_foregroundBlur_197_138" />
          </filter>
          <filter id="filter2_i_197_138" x="151.179" y="175.18" width="217.924" height="221.924" filterUnits="userSpaceOnUse" colorInterpolationFilters="sRGB">
            <feFlood floodOpacity="0" result="BackgroundImageFix" />
            <feBlend mode="normal" in="SourceGraphic" in2="BackgroundImageFix" result="shape" />
            <feColorMatrix in="SourceAlpha" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" result="hardAlpha" />
            <feOffset dy="6" />
            <feGaussianBlur stdDeviation="2" />
            <feComposite in2="hardAlpha" operator="arithmetic" k2="-1" k3="1" />
            <feColorMatrix type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.25 0" />
            <feBlend mode="normal" in2="shape" result="effect1_innerShadow_197_138" />
          </filter>
          <filter id="filter3_dd_197_138" x="116.047" y="138.047" width="288.188" height="291.189" filterUnits="userSpaceOnUse" colorInterpolationFilters="sRGB">
            <feFlood floodOpacity="0" result="BackgroundImageFix" />
            <feColorMatrix in="SourceAlpha" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" result="hardAlpha" />
            <feOffset dy="-2" />
            <feGaussianBlur stdDeviation="2" />
            <feComposite in2="hardAlpha" operator="out" />
            <feColorMatrix type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.25 0" />
            <feBlend mode="normal" in2="BackgroundImageFix" result="effect1_dropShadow_197_138" />
            <feColorMatrix in="SourceAlpha" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" result="hardAlpha" />
            <feOffset dy="1" />
            <feGaussianBlur stdDeviation="2" />
            <feComposite in2="hardAlpha" operator="out" />
            <feColorMatrix type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.25 0" />
            <feBlend mode="normal" in2="effect1_dropShadow_197_138" result="effect2_dropShadow_197_138" />
            <feBlend mode="normal" in="SourceGraphic" in2="effect2_dropShadow_197_138" result="shape" />
          </filter>
          <filter id="filter5_dd_197_138" x="69.3486" y="90.3496" width="381.585" height="385.584" filterUnits="userSpaceOnUse" colorInterpolationFilters="sRGB">
            <feFlood floodOpacity="0" result="BackgroundImageFix" />
            <feColorMatrix in="SourceAlpha" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" result="hardAlpha" />
            <feOffset dy="-3" />
            <feGaussianBlur stdDeviation="2" />
            <feComposite in2="hardAlpha" operator="out" />
            <feColorMatrix type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.25 0" />
            <feBlend mode="normal" in2="BackgroundImageFix" result="effect1_dropShadow_197_138" />
            <feColorMatrix in="SourceAlpha" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" result="hardAlpha" />
            <feOffset dy="1" />
            <feGaussianBlur stdDeviation="2" />
            <feComposite in2="hardAlpha" operator="out" />
            <feColorMatrix type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.25 0" />
            <feBlend mode="normal" in2="effect1_dropShadow_197_138" result="effect2_dropShadow_197_138" />
            <feBlend mode="normal" in="SourceGraphic" in2="effect2_dropShadow_197_138" result="shape" />
          </filter>
          <filter id="filter7_dd_197_138" x="22.6509" y="43.6504" width="474.981" height="479.98" filterUnits="userSpaceOnUse" colorInterpolationFilters="sRGB">
            <feFlood floodOpacity="0" result="BackgroundImageFix" />
            <feColorMatrix in="SourceAlpha" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" result="hardAlpha" />
            <feOffset dy="-3" />
            <feGaussianBlur stdDeviation="2" />
            <feComposite in2="hardAlpha" operator="out" />
            <feColorMatrix type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.25 0" />
            <feBlend mode="normal" in2="BackgroundImageFix" result="effect1_dropShadow_197_138" />
            <feColorMatrix in="SourceAlpha" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" result="hardAlpha" />
            <feOffset dy="2" />
            <feGaussianBlur stdDeviation="2" />
            <feComposite in2="hardAlpha" operator="out" />
            <feColorMatrix type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.25 0" />
            <feBlend mode="normal" in2="effect1_dropShadow_197_138" result="effect2_dropShadow_197_138" />
            <feBlend mode="normal" in="SourceGraphic" in2="effect2_dropShadow_197_138" result="shape" />
          </filter>
        </defs>
      </svg>
    </div>
  );
}
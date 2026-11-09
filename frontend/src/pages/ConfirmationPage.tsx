import { useTrialRoom } from "../context/TrialRoomContext";
import NavigationButton from "../components/NavigationButton";
import PrimaryButton from "../components/PrimaryButton";
import { API_BASE_URL } from "../utils/api";

type PageProps = {
  onRetake: () => void;
  onContinue: () => void;
  onHome: () => void;
};

export default function ConfirmationPage({ onRetake, onContinue, onHome }: PageProps) {
  const { capturedImage } = useTrialRoom();

  const displayImage = capturedImage || "";

  // The visitor has accepted this photo: start its slow analysis now, so the try-on doesn't wait for it
  const handleContinue = () => {
    if (capturedImage) {
      fetch(`${API_BASE_URL}/api/prepare-capture`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: capturedImage }),
      }).catch((err) => console.warn("[PREPARE CAPTURE FAILED] The try-on will do this work itself.", err));
    }
    onContinue();
  };

  return (
    <div
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        background: "transparent",
        overflow: "hidden",
      }}
    >
      {/* Frame 65 (matching CapturePage Frame 62 layout) */}
      <div
        style={{
          position: "absolute",
          width: "49.5vh",
          height: "auto",
          left: "calc(50% - 24.75vh)",
          top: "10.5vh",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: "2.0vh",
        }}
      >
        {/* selection-header */}
        <div
          style={{
            display: "flex",
            flexDirection: "row",
            alignItems: "center",
            justifyContent: "center",
            gap: "1.0vh",
            width: "100%",
            height: "1.83vh",
          }}
        >
          {/* Left Line */}
          <div
            style={{
              boxSizing: "border-box",
              width: "2.5vh",
              height: "0px",
              border: "0.25vh solid #5C4A3D",
            }}
          />

          {/* IMAGE CAPTURED */}
          <span
            style={{
              fontFamily: "'Montserrat', sans-serif",
              fontWeight: 700,
              fontSize: "1.5vh",
              lineHeight: "1.83vh",
              textTransform: "uppercase",
              color: "#5C4A3D",
              textAlign: "center",
              whiteSpace: "nowrap",
            }}
          >
            IMAGE CAPTURED
          </span>

          {/* Right Line */}
          <div
            style={{
              boxSizing: "border-box",
              width: "2.5vh",
              height: "0px",
              border: "0.25vh solid #5C4A3D",
            }}
          />
        </div>

        {/* Frame 58 */}
        <div
          style={{
            display: "flex",
            flexDirection: "row",
            alignItems: "center",
            justifyContent: "center",
            gap: "1.26vh",
            width: "100%",
            height: "55.32vh",
            position: "relative",
          }}
        >

          {/* Camera wrapper — allows home button to overflow the clipped camera box */}
          <div
            style={{
              position: "relative",
              width: "47.07vh",
              height: "100%",
            }}
          >
            {/* Glowing Camera Outline Box (holds captured image) */}
            <div
              style={{
                boxSizing: "border-box",
                width: "100%",
                height: "100%",
                background: "rgba(255, 255, 255, 0.05)",
                border: "0.126vh solid rgba(255, 255, 255, 0.3)",
                boxShadow: "0px 1.0vh 4.03vh rgba(0, 0, 0, 0.37), inset -2px -2px 20px #D4C7BC, inset 2px 2px 20px #B9AEA4",
                backdropFilter: "blur(12px)",
                WebkitBackdropFilter: "blur(12px)",
                borderRadius: "4.03vh",
                position: "relative",
                overflow: "hidden",
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
              }}
            >
              {/* Captured Image Display */}
              <img
                src={displayImage}
                alt="Captured Frame Preview"
                style={{
                  width: "100%",
                  height: "100%",
                  objectFit: "cover",
                }}
              />

              {/* Corner Brackets */}
              {/* Top-Left Corner */}
              <div
                style={{
                  boxSizing: "border-box",
                  position: "absolute",
                  width: "5.04vh",
                  height: "5.04vh",
                  left: "-2px",
                  top: "-2px",
                  borderWidth: "0.38vh 0px 0px 0.38vh",
                  borderStyle: "solid",
                  borderColor: "#F3E5D8",
                  filter: "drop-shadow(0px 0px 1.89vh rgba(196, 164, 132, 0.5))",
                  borderTopLeftRadius: "4.03vh",
                  pointerEvents: "none",
                  zIndex: 3,
                }}
              />
              {/* Top-Right Corner */}
              <div
                style={{
                  boxSizing: "border-box",
                  position: "absolute",
                  width: "5.04vh",
                  height: "5.04vh",
                  right: "-2px",
                  top: "-2px",
                  borderWidth: "0.38vh 0.38vh 0px 0px",
                  borderStyle: "solid",
                  borderColor: "#F3E5D8",
                  filter: "drop-shadow(0px 0px 1.89vh rgba(196, 164, 132, 0.5))",
                  borderTopRightRadius: "4.03vh",
                  pointerEvents: "none",
                  zIndex: 3,
                }}
              />
              {/* Bottom-Left Corner */}
              <div
                style={{
                  boxSizing: "border-box",
                  position: "absolute",
                  width: "5.04vh",
                  height: "5.04vh",
                  left: "-2px",
                  bottom: "-2px",
                  borderWidth: "0px 0px 0.38vh 0.38vh",
                  borderStyle: "solid",
                  borderColor: "#F3E5D8",
                  filter: "drop-shadow(0px 0px 1.89vh rgba(196, 164, 132, 0.5))",
                  borderBottomLeftRadius: "4.03vh",
                  pointerEvents: "none",
                  zIndex: 3,
                }}
              />
              {/* Bottom-Right Corner */}
              <div
                style={{
                  boxSizing: "border-box",
                  position: "absolute",
                  width: "5.04vh",
                  height: "5.04vh",
                  right: "-2px",
                  bottom: "-2px",
                  borderWidth: "0px 0.38vh 0.38vh 0px",
                  borderStyle: "solid",
                  borderColor: "#F3E5D8",
                  filter: "drop-shadow(0px 0px 1.89vh rgba(196, 164, 132, 0.5))",
                  borderBottomRightRadius: "4.03vh",
                  pointerEvents: "none",
                  zIndex: 3,
                }}
              />
            </div>

            {/* Home button — centered on bottom border, outside the clipped camera box */}
            <NavigationButton
              type="home"
              onClick={onHome}
              style={{
                position: "absolute",
                bottom: "-3.52vh",
                left: "45%",
                transform: "translateX(-50%)",
                width: "7.05vh",
                height: "7.05vh",
                zIndex: 10,
              }}
            />

            {/* Right-Side Floating Actions */}
            <div
              style={{
                position: "absolute",
                right: "-3.52vh", // half of width to overlap border
                top: "50%",
                transform: "translateY(-50%)",
                display: "flex",
                flexDirection: "column",
                gap: "1.5vh",
                zIndex: 10,
              }}
            >
              {/* Continue / Next Button */}
              <button
                onClick={handleContinue}
                style={{
                  display: "flex",
                  flexDirection: "row",
                  justifyContent: "center",
                  alignItems: "center",
                  padding: "0px",
                  isolation: "isolate",
                  width: "7.05vh",
                  height: "7.05vh",
                  background: "linear-gradient(270deg, rgba(92, 74, 61, 0.9) -16.96%, rgba(194, 156, 129, 0.9) 89.29%)",
                  boxShadow: "0px 40px 50px -10px rgba(0, 0, 0, 0.1), 0px 16px 20px -12px rgba(0, 0, 0, 0.1)",
                  borderRadius: "19998px",
                  border: "none",
                  cursor: "pointer",
                  outline: "none",
                  position: "relative",
                }}
              >
                <svg
                  viewBox="76 39 40 34"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                  style={{
                    width: "40%",
                    height: "40%",
                    pointerEvents: "none",
                  }}
                >
                  <path d="M114.846 57.6584L100.783 71.7209C100.343 72.1612 99.7457 72.4085 99.123 72.4085C98.5004 72.4085 97.9032 72.1612 97.4629 71.7209C97.0226 71.2806 96.7752 70.6834 96.7752 70.0607C96.7752 69.438 97.0226 68.8409 97.4629 68.4006L107.523 58.3439H78.8125C78.1909 58.3439 77.5948 58.097 77.1552 57.6574C76.7157 57.2179 76.4688 56.6218 76.4688 56.0002C76.4688 55.3786 76.7157 54.7824 77.1552 54.3429C77.5948 53.9033 78.1909 53.6564 78.8125 53.6564H107.523L97.4668 43.5939C97.0265 43.1536 96.7791 42.5564 96.7791 41.9338C96.7791 41.3111 97.0265 40.7139 97.4668 40.2736C97.9071 39.8333 98.5043 39.5859 99.127 39.5859C99.7496 39.5859 100.347 39.8333 100.787 40.2736L114.85 54.3361C115.068 54.5541 115.241 54.8132 115.36 55.0984C115.478 55.3836 115.538 55.6893 115.538 55.998C115.538 56.3067 115.476 56.6123 115.357 56.8973C115.239 57.1822 115.065 57.4408 114.846 57.6584Z" fill="white" />
                </svg>
              </button>

              {/* Retake Button */}
              <button
                onClick={onRetake}
                style={{
                  display: "flex",
                  flexDirection: "row",
                  justifyContent: "center",
                  alignItems: "center",
                  padding: "0px",
                  isolation: "isolate",
                  width: "7.05vh",
                  height: "7.05vh",
                  background: "linear-gradient(270deg, rgba(92, 74, 61, 0.9) -16.96%, rgba(194, 156, 129, 0.9) 89.29%)",
                  boxShadow: "0px 40px 50px -10px rgba(0, 0, 0, 0.1), 0px 16px 20px -12px rgba(0, 0, 0, 0.1)",
                  borderRadius: "19998px",
                  border: "none",
                  cursor: "pointer",
                  outline: "none",
                  position: "relative",
                }}
              >
                <svg
                  viewBox="70 33 52 45"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                  style={{
                    width: "45%",
                    height: "45%",
                    pointerEvents: "none",
                  }}
                >
                  <path d="M114.75 39.125H108.253L105.059 34.3344C104.887 34.0779 104.656 33.8675 104.384 33.722C104.112 33.5764 103.808 33.5002 103.5 33.5H88.5C88.1916 33.5002 87.8881 33.5764 87.6162 33.722C87.3443 33.8675 87.1126 34.0779 86.9414 34.3344L83.7445 39.125H77.25C75.7582 39.125 74.3274 39.7176 73.2725 40.7725C72.2176 41.8274 71.625 43.2582 71.625 44.75V71C71.625 72.4918 72.2176 73.9226 73.2725 74.9775C74.3274 76.0324 75.7582 76.625 77.25 76.625H114.75C116.242 76.625 117.673 76.0324 118.727 74.9775C119.782 73.9226 120.375 72.4918 120.375 71V44.75C120.375 43.2582 119.782 41.8274 118.727 40.7725C117.673 39.7176 116.242 39.125 114.75 39.125ZM116.625 71C116.625 71.4973 116.427 71.9742 116.076 72.3258C115.724 72.6775 115.247 72.875 114.75 72.875H77.25C76.7527 72.875 76.2758 72.6775 75.9242 72.3258C75.5725 71.9742 75.375 71.4973 75.375 71V44.75C75.375 44.2527 75.5725 43.7758 75.9242 43.4242C76.2758 43.0725 76.7527 42.875 77.25 42.875H84.75C85.0588 42.8752 85.3628 42.7991 85.6351 42.6536C85.9074 42.508 86.1396 42.2975 86.3109 42.0406L89.5031 37.25H102.495L105.689 42.0406C105.86 42.2975 106.093 42.508 106.365 42.6536C106.637 42.7991 106.941 42.8752 107.25 42.875H114.75C115.247 42.875 115.724 43.0725 116.076 43.4242C116.427 43.7758 116.625 44.2527 116.625 44.75V71ZM107.25 48.5V54.125C107.25 54.6223 107.052 55.0992 106.701 55.4508C106.349 55.8025 105.872 56 105.375 56H99.75C99.2527 56 98.7758 55.8025 98.4242 55.4508C98.0725 55.0992 97.875 54.6223 97.875 54.125C97.875 53.6277 98.0725 53.1508 98.4242 52.7992C98.7758 52.4475 99.2527 52.25 99.75 52.25H100.957C99.6619 51.1153 98.0169 50.4589 96.2964 50.3902C94.5759 50.3216 92.8838 50.8448 91.5023 51.8727C91.1048 52.1716 90.6048 52.3005 90.1123 52.2308C89.6198 52.1611 89.1752 51.8987 88.8762 51.5012C88.5772 51.1037 88.4483 50.6037 88.518 50.1111C88.5877 49.6186 88.8501 49.174 89.2477 48.875C91.3319 47.3233 93.8874 46.5374 96.4834 46.6496C99.0795 46.7619 101.557 47.7655 103.5 49.4914V48.5C103.5 48.0027 103.698 47.5258 104.049 47.1742C104.401 46.8225 104.878 46.625 105.375 46.625C105.872 46.625 106.349 46.8225 106.701 47.1742C107.052 47.5258 107.25 48.0027 107.25 48.5ZM103.123 62.375C103.422 62.7722 103.55 63.2719 103.481 63.7642C103.412 64.2564 103.149 64.7009 102.752 65C100.668 66.5517 98.1126 67.3376 95.5166 67.2254C92.9205 67.1131 90.4425 66.1095 88.5 64.3836V65.375C88.5 65.8723 88.3025 66.3492 87.9508 66.7008C87.5992 67.0525 87.1223 67.25 86.625 67.25C86.1277 67.25 85.6508 67.0525 85.2992 66.7008C84.9475 66.3492 84.75 65.8723 84.75 65.375V59.75C84.75 59.2527 84.9475 58.7758 85.2992 58.4242C85.6508 58.0725 86.1277 57.875 86.625 57.875H92.25C92.7473 57.875 93.2242 58.0725 93.5758 58.4242C93.9275 58.7758 94.125 59.2527 94.125 59.75C94.125 60.2473 93.9275 60.7242 93.5758 61.0758C93.2242 61.4275 92.7473 61.625 92.25 61.625H91.043C92.3381 62.7597 93.9831 63.4161 95.7036 63.4848C97.4241 63.5534 99.1162 63.0302 100.498 62.0023C100.895 61.7037 101.395 61.5752 101.887 61.6451C102.38 61.715 102.824 61.9775 103.123 62.375Z" fill="white" />
                </svg>
              </button>
            </div>
          </div>

        </div>

        {/* Frame 64 */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: "2.5vh",
            width: "100%",
            height: "auto",
            marginTop: "7vh", // Pushed down to fill empty bottom space
          }}
        >
          {/* IMAGE CAPTURED label */}
          <span
            style={{
              width: "100%",
              height: "3.15vh",
              fontFamily: "'Montserrat', sans-serif",
              fontWeight: 700,
              fontSize: "1.9vh",
              lineHeight: "3.15vh",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              textAlign: "center",
              letterSpacing: "0.26vh",
              textTransform: "uppercase",
              color: "#5C4A3D",
              marginTop: "1vh",
              whiteSpace: "nowrap",
            }}
          >
            IMAGE CAPTURED
          </span>

          {/* Subtext */}
          <span
            style={{
              fontFamily: "'Poppins', sans-serif",
              fontWeight: 500,
              fontSize: "1.8vh", // 20px
              lineHeight: "1.81vh", // 28.8px
              letterSpacing: "0.03vh", // 0.5px
              textAlign: "center",
              verticalAlign: "middle",
              color: "#5C4A3D",
            }}
          >
            Your body scan is ready for virtual try-on.<br />
            Retake if needed, or continue to choose your outfit.
          </span>

        </div>
      </div>
    </div>
  );
}

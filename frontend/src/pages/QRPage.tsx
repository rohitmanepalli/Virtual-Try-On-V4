import QRCard from "../components/QRCard";
import VisitorDetailsPanel from "../components/VisitorDetailsPanel";
import NavigationButton from "../components/NavigationButton";
import { useTrialRoom } from "../context/TrialRoomContext";

type PageProps = {
  onHome: () => void;
};

export default function QRPage({ onHome }: PageProps) {
  const { generatedImage, qrCodeBase64 } = useTrialRoom();

  return (
    <div
      style={{
        flex: 1,
        width: "100%",
        height: "100%",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        background: "transparent",
        position: "relative",
      }}
    >
      {/* Top Header Area */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          width: "100%",
          marginBottom: "2vh",
          gap: "1.5vh",
        }}
      >
        <div
          style={{
            display: "flex",
            flexDirection: "row",
            alignItems: "center",
            justifyContent: "center",
            gap: "1.5vh",
            width: "100%",
          }}
        >
          {/* Left Divider */}
          <div style={{ width: "3.0vh", height: "0.2vh", background: "#5C4A3D" }} />

          <h1
            style={{
              fontFamily: "'Poppins', sans-serif",
              fontWeight: 600,
              fontSize: "2.4vh",
              textTransform: "uppercase",
              color: "#5C4A3D",
              textAlign: "center",
              whiteSpace: "nowrap",
              margin: 0,
              letterSpacing: "0.12vh",
              display: "flex",
              alignItems: "center",
            }}
          >
            SCAN TO <span style={{ color: "#5C4A3D", marginLeft: "0.5vh" }}>DOWNLOAD</span>
          </h1>

          {/* Right Divider */}
          <div style={{ width: "3.0vh", height: "0.2vh", background: "#5C4A3D" }} />
        </div>

        {/* Subtitle */}
        <span
          style={{
            fontFamily: "'Poppins', sans-serif",
            fontWeight: 500,
            fontSize: "2.0vh",
            lineHeight: "2.9vh",
            letterSpacing: "0.0504vh",
            color: "#5C4A3D",
            textAlign: "center",
            width: "41.4vh",
          }}
        >
          Scan the QR code to download and save your virtual try-on.
        </span>
      </div>

      {/* Main Preview Container */}
      <div
        style={{
          boxSizing: "border-box",
          width: "38.18vh",
          height: "49.5vh",
          position: "relative",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          margin: "0.76vh 0 2.52vh 0",
        }}
      >
        <div
          style={{
            width: "100%",
            height: "100%",
            overflow: "hidden",
          }}
        >
          <img
            src={generatedImage || ""}
            alt="Generated Outfit"
            style={{
              width: "100%",
              height: "100%",
              objectFit: "contain", // never crop the fixed 3:4 render
            }}
          />
        </div>

      </div>

      {/* QR Code Section */}
      <div
        style={{
          border: "0.8vh solid #5C4A3D",
          borderRadius: "1.3vh",
          overflow: "hidden",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          marginTop: "1vh",
        }}
      >
        <QRCard qrCodeSrc={qrCodeBase64} />
      </div>

      {/* Bottom Button Area */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          width: "100%",
          marginTop: "1.5vh",
        }}
      >
        {/* Home Button centered below */}
        <div style={{ display: "flex", justifyContent: "center", gap: "3vh", marginTop: "1.5vh" }}>
          <NavigationButton
            type="home"
            onClick={onHome}
            style={{ width: "7.05vh", height: "7.05vh" }}
          />
        </div>
      </div>

      <VisitorDetailsPanel />
    </div>
  );
}

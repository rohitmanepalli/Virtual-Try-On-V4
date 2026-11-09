import { useEffect, useState } from "react";

type PageProps = {
  onComplete: () => void;
  onCalibrate?: () => void;
};

export default function CameraSelectionPage({ onComplete, onCalibrate }: PageProps) {
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchCameras() {
      try {
        // Trigger permission prompt in browser if necessary by requesting a dummy stream
        const dummyStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        
        const allDevices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = allDevices.filter((d) => d.kind === "videoinput");
        setDevices(videoDevices);
        
        // Stop dummy stream
        dummyStream.getTracks().forEach((track) => track.stop());
      } catch (err) {
        console.error("Error fetching cameras:", err);
      } finally {
        setLoading(false);
      }
    }
    fetchCameras();
  }, []);

  const handleSelectCamera = (deviceId: string) => {
    localStorage.setItem("selectedCameraId", deviceId);
    onComplete();
  };

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
      {/* Header */}
      <div
        style={{
          display: "flex",
          flexDirection: "row",
          alignItems: "center",
          justifyContent: "center",
          gap: "1.5vh",
          width: "100%",
          marginBottom: "4vh",
        }}
      >
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
          SELECT <span style={{ color: "rgba(92, 74, 61, 0.4)", marginLeft: "0.5vh" }}>CAMERA</span>
        </h1>
        <div style={{ width: "3.0vh", height: "0.2vh", background: "#5C4A3D" }} />
      </div>

      {/* Camera List */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "2vh",
          width: "60vh",
          maxHeight: "60vh",
          overflowY: "auto",
          padding: "2vh",
        }}
      >
        {loading ? (
          <span
            style={{
              fontFamily: "'Poppins', sans-serif",
              fontSize: "2vh",
              color: "#5C4A3D",
              textAlign: "center",
            }}
          >
            Loading cameras...
          </span>
        ) : devices.length > 0 ? (
          devices.map((device, index) => (
            <button
              key={device.deviceId}
              onClick={() => handleSelectCamera(device.deviceId)}
              style={{
                width: "100%",
                padding: "2vh",
                background: "rgba(255, 255, 255, 0.2)",
                border: "0.2vh solid rgba(92, 74, 61, 0.3)",
                borderRadius: "1.5vh",
                backdropFilter: "blur(1vh)",
                WebkitBackdropFilter: "blur(1vh)",
                cursor: "pointer",
                transition: "all 0.3s ease",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = "rgba(255, 255, 255, 0.4)";
                e.currentTarget.style.borderColor = "#5C4A3D";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = "rgba(255, 255, 255, 0.2)";
                e.currentTarget.style.borderColor = "rgba(92, 74, 61, 0.3)";
              }}
            >
              <span
                style={{
                  fontFamily: "'Poppins', sans-serif",
                  fontWeight: 500,
                  fontSize: "1.8vh",
                  color: "#5C4A3D",
                  pointerEvents: "none",
                }}
              >
                {device.label || `Camera ${index + 1}`}
              </span>
            </button>
          ))
        ) : (
          <span
            style={{
              fontFamily: "'Poppins', sans-serif",
              fontSize: "2vh",
              color: "#5C4A3D",
              textAlign: "center",
            }}
          >
            No cameras found.
          </span>
        )}
      </div>

      {onCalibrate && (
        <button
          onClick={onCalibrate}
          style={{
            marginTop: "2.5vh",
            background: "rgba(92, 74, 61, 0.1)",
            border: "0.15vh dashed rgba(92, 74, 61, 0.5)",
            borderRadius: "1vh",
            color: "#5C4A3D",
            padding: "1vh 2vh",
            fontFamily: "'Poppins', sans-serif",
            fontSize: "1.4vh",
            fontWeight: 600,
            cursor: "pointer",
            transition: "all 0.2s ease",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "rgba(92, 74, 61, 0.2)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "rgba(92, 74, 61, 0.1)";
          }}
        >
          ⚙ Calibrate Camera Distance Constant
        </button>
      )}
    </div>
  );
}

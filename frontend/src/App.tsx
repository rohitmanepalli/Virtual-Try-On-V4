import { useState, useEffect, useRef } from "react";
import HoloboxFrame from "./components/HoloboxFrame";
import WelcomePage from "./pages/WelcomePage";
import ClothingIntroPage from "./pages/ClothingIntroPage";
import CapturePage from "./pages/CapturePage";
import ConfirmationPage from "./pages/ConfirmationPage";
import OutfitSelectionPage from "./pages/OutfitSelectionPage";
import GeneratingPage from "./pages/GeneratingPage";
import ResultPage from "./pages/ResultPage";
import QRPage from "./pages/QRPage";
import CameraSelectionPage from "./pages/CameraSelectionPage";
import BeautyStudioPage from "./pages/BeautyStudioPage";

function CameraCalibrationPage({ onBack }: { onBack: () => void }) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        height: "100%",
        gap: "2vh",
        color: "#5C4A3D",
        textAlign: "center",
        padding: "2vh",
      }}
    >
      <h2 style={{ fontFamily: "'Poppins', sans-serif", margin: 0 }}>Camera Calibration</h2>
      <p style={{ fontFamily: "'Poppins', sans-serif", fontSize: "1.4vh", maxWidth: "40vh" }}>
        Focal length and distance parameters are configured in the environment. Stand at 2.5m distance to verify focal calibration.
      </p>
      <button
        onClick={onBack}
        style={{
          marginTop: "2vh",
          padding: "1vh 2.5vh",
          background: "linear-gradient(135deg, #B79579 0%, #5C4A3D 100%)",
          border: "none",
          borderRadius: "999vh",
          color: "#FFFFFF",
          fontFamily: "'Poppins', sans-serif",
          fontWeight: 600,
          cursor: "pointer",
        }}
      >
        BACK TO CAMERA SELECTION
      </button>
    </div>
  );
}

import { TrialRoomProvider } from "./context/TrialRoomContext";

import { useTrialRoom } from "./context/TrialRoomContext";
import { logEvent } from "./utils/clientLog";
import backgroundAudioSrc from "./assets/audio/background_audio.wav";

const BACKGROUND_AUDIO_VOLUME = 0.15;

// A visitor who walks away must not leave their photo on screen for the next person
const IDLE_RESET_MS = 3 * 60 * 1000;
const SCREENS_WITHOUT_IDLE_RESET: Screen[] = ["camera-selection", "calibration", "welcome", "clothing-intro", "generating"];

type Screen =
  | "camera-selection"
  | "calibration"
  | "welcome"
  | "clothing-intro"
  | "capture"
  | "confirmation"
  | "outfit-selection"
  | "beauty-studio"
  | "generating"
  | "result"
  | "qr";

function AppContent() {
  const [currentScreen, setCurrentScreen] = useState<Screen>(() => {
    if (typeof window !== "undefined") {
      const search = window.location.search;
      if (search.includes("screen=capture")) return "capture";
      if (search.includes("screen=welcome")) return "welcome";
      if (search.includes("screen=beauty")) return "beauty-studio";
      if (search.includes("calibrate")) return "calibration";
    }
    return "camera-selection";
  });
  const { resetSession, capturedImage } = useTrialRoom();

  const navigateTo = (screen: Screen) => {
    logEvent(`screen: ${currentScreen} -> ${screen}`);
    setCurrentScreen(screen);
    if (screen === "welcome") {
      resetSession();
    }
  };

  useEffect(() => {
    if (SCREENS_WITHOUT_IDLE_RESET.includes(currentScreen)) return;
    let timer = 0;
    const restart = () => {
      window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        logEvent(`idle reset from ${currentScreen}`);
        setCurrentScreen("welcome");
        resetSession();
      }, IDLE_RESET_MS);
    };
    const events = ["pointerdown", "keydown"];
    events.forEach((name) => window.addEventListener(name, restart));
    restart();
    return () => {
      window.clearTimeout(timer);
      events.forEach((name) => window.removeEventListener(name, restart));
    };
    // resetSession is recreated each render and always acts on the current session
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentScreen]);

  const renderScreen = () => {
    switch (currentScreen) {
      case "camera-selection":
        return (
          <CameraSelectionPage
            onComplete={() => navigateTo("welcome")}
            onCalibrate={() => navigateTo("calibration")}
          />
        );

      case "calibration":
        return <CameraCalibrationPage onBack={() => navigateTo("camera-selection")} />;

      case "welcome":
        return <WelcomePage onNext={() => navigateTo("clothing-intro")} onBeauty={() => navigateTo("beauty-studio")} />;

      case "clothing-intro":
        return <ClothingIntroPage onComplete={() => navigateTo("capture")} />;

      case "capture":
        return (
          <CapturePage
            onHome={() => navigateTo("welcome")}
            onCapture={() => navigateTo("confirmation")}
          />
        );

      case "confirmation":
        return (
          <ConfirmationPage
            onRetake={() => navigateTo("capture")}
            onContinue={() => navigateTo("outfit-selection")}
            onHome={() => navigateTo("welcome")}
          />
        );

      case "outfit-selection":
        return (
          <OutfitSelectionPage
            onBack={() => navigateTo("confirmation")}
            onHome={() => navigateTo("welcome")}
            onContinue={() => navigateTo("generating")}
            onBeauty={() => navigateTo("beauty-studio")}
          />
        );

      case "beauty-studio":
        return (
          <BeautyStudioPage
            // Arriving from the welcome screen there is no body capture yet: clothing starts at the capture step
            onClothing={() => navigateTo(capturedImage ? "outfit-selection" : "capture")}
            onHome={() => navigateTo("welcome")}
          />
        );

      case "generating":
        return (
          <GeneratingPage
            onComplete={() => navigateTo("result")}
            onBack={() => navigateTo("outfit-selection")}
            onHome={() => navigateTo("welcome")}
          />
        );

      case "result":
        return (
          <ResultPage
            onTrySomethingElse={() => navigateTo("outfit-selection")}
            onLikeOutfit={() => navigateTo("qr")}
            onBack={() => navigateTo("outfit-selection")}
            onHome={() => navigateTo("welcome")}
          />
        );

      case "qr":
        return <QRPage onHome={() => navigateTo("welcome")} />;

      default:
        return <WelcomePage onNext={() => navigateTo("capture")} />;
    }
  };

  return <HoloboxFrame>{renderScreen()}</HoloboxFrame>;
}

function App() {
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = BACKGROUND_AUDIO_VOLUME;
    }
  }, []);

  return (
    <TrialRoomProvider>
      {/* Background Audio element that persists across all screens */}
      <audio
        ref={audioRef}
        src={backgroundAudioSrc}
        autoPlay
        loop
        playsInline
      />
      <AppContent />
    </TrialRoomProvider>
  );
}

export default App;
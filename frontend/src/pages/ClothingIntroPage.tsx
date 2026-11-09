import { useEffect, useRef } from "react";
import clothingIntroVideo from "../assets/clothing_intro.mp4";

type PageProps = {
  onComplete: () => void;
};

// Plays once after the Clothing button on the welcome screen, then hands over to the capture step.
export default function ClothingIntroPage({ onComplete }: PageProps) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    // If the browser blocks playback with sound, play muted. If it still won't play, don't strand the user here.
    video.play().catch(() => {
      video.muted = true;
      video.play().catch(onComplete);
    });
    // Runs once on mount: a re-render must not restart the video
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    // Positioned against the frame's safe-zone container, so inset 0 also covers its padding
    <div style={{ position: "absolute", inset: 0, background: "#EDEDEB", zIndex: 20, display: "flex", justifyContent: "center" }}>
      <video
        ref={videoRef}
        src={clothingIntroVideo}
        playsInline
        preload="auto"
        onEnded={onComplete}
        onError={onComplete}
        // The presenter fills the video top to bottom, so it is fitted to the full height rather than cropped.
        // Its backdrop shade varies, so no flat fill matches it: fade the sides out to avoid a visible edge line.
        style={{
          height: "100%",
          aspectRatio: "9 / 16",
          display: "block",
          maskImage: "linear-gradient(to right, transparent, #000 8%, #000 92%, transparent)",
        }}
      />
    </div>
  );
}

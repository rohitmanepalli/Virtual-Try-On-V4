import { useEffect, useState, useRef } from "react";
import FashionScanner from "../components/FashionScanner";
import NavigationButton from "../components/NavigationButton";
import { useTrialRoom } from "../context/TrialRoomContext";
import { API_BASE_URL } from "../utils/api";
import male1 from "../assets/dress/male1.png";
import male2 from "../assets/dress/male2.png";
import female1 from "../assets/dress/female1.png";
import female2 from "../assets/dress/female2.png";
import female3 from "../assets/dress/female3.png";
import male3 from "../assets/dress/male3.png";
import blazer1 from "../assets/dress/blazer1.png";
import blazer2 from "../assets/dress/blazer2.png";
import kurti1 from "../assets/dress/kurti1.png";
import salwar_suit1 from "../assets/dress/salwar_suit1.png";
import lehenga1 from "../assets/dress/lehenga1.png";
import kerala_saree1 from "../assets/dress/kerala_saree1.png";
import kurta1 from "../assets/dress/kurta1.png";
import sherwani1 from "../assets/dress/sherwani1.png";
import royal_sherwani1 from "../assets/dress/royal_sherwani1.png";
import kurta_dhoti1 from "../assets/dress/kurta_dhoti1.png";
import kurta_pajama1 from "../assets/dress/kurta_pajama1.png";
import mundu_shirt1 from "../assets/dress/mundu_shirt1.png";
import outfit1 from "../assets/outfit_1.png";
import outfit2 from "../assets/outfit_2.png";
import outfit4 from "../assets/outfit_4.png";

import generatingVO from "../assets/audio/generating_vo.mp3";


type PageProps = {
  onComplete: () => void;
  onBack?: () => void;
  onHome?: () => void;
};

const MOCK_MODE = false;

const OUTFITS_MAP: Record<string, string> = {
  "female1": female1,
  "female2": female2,
  "female3": female3,
  "kurti1": kurti1,
  "salwar_suit1": salwar_suit1,
  "lehenga1": lehenga1,
  "kerala_saree1": kerala_saree1,
  "kurta1": kurta1,
  "sherwani1": sherwani1,
  "royal_sherwani1": royal_sherwani1,
  "kurta_dhoti1": kurta_dhoti1,
  "kurta_pajama1": kurta_pajama1,
  "mundu_shirt1": mundu_shirt1,
  "male1": male1,
  "male2": male2,
  "male3": male3,
  "blazer1": blazer1,
  "blazer2": blazer2,
  "outfit1": outfit1,
  "outfit2": outfit2,
  "outfit4": outfit4,
};

const STATUS_MESSAGES = [
  "Curating Your Style",
  "Styling Your Look",
  "AI Stylist at Work",
  "Matching Your Outfit",
  "Analyzing Your Fashion Profile",
  "Preparing Your Signature Look",
  "Refining Your Style",
  "Tailoring Your Experience",
  "Applying Your Selected Outfit",
  "Preparing Your Fashion Preview"
];

const FASHION_FACTS = [
  { prefix: "✨ Did you know?", text: "Neutral colors create over 70% of luxury fashion collections." },
  { prefix: "✨ Fashion Fact", text: "Layering adds depth and sophistication to your outfit." },
  { prefix: "✨ Style Tip", text: "Well-fitted clothing instantly enhances your appearance." },
  { prefix: "✨ AI Insight", text: "Your selected outfit is being matched to your body proportions." },
  { prefix: "✨ Fashion Tip", text: "Accessories can completely transform the same outfit." },
  { prefix: "✨ Did you know?", text: "Denim has remained a fashion staple for over 150 years." },
  { prefix: "✨ Style Insight", text: "The AI is optimizing your outfit for balance and proportion." }
];

const convertImageToBase64 = async (imageSrc: string): Promise<string> => {
  const response = await fetch(imageSrc);
  const blob = await response.blob();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
};

export default function GeneratingPage({ onComplete, onBack, onHome }: PageProps) {
  const {
    sessionId,
    capturedImage,
    selectedOutfit,
    selectedSize,
    fitState,
    bodyProfile,
    setGeneratedImage,
    setLastGoodRender,
    setQrCodeBase64,
    setSelectedSize,
    setRecommendedSize,
    setAvailableSizes,
    setSizeFitQuality,
    setSizeConfidence,
    setSizeFitNotes,
    setSizeRenderInCache,
    setSizeFits,
    setWontFitInCache,
  } = useTrialRoom();
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [apiDone, setApiDone] = useState(false);
  const [minTimeElapsed, setMinTimeElapsed] = useState(false);

  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = 1.0;
    }
  }, []);

  // Status message rotation
  const [statusIndex, setStatusIndex] = useState(0);
  const [fadeStatus, setFadeStatus] = useState(true);

  // Fashion facts rotation
  const [factIndex, setFactIndex] = useState(0);
  const [fadeFact, setFadeFact] = useState(true);

  // Enforce a minimum display time for scanner engagement
  useEffect(() => {
    const timer = setTimeout(() => {
      setMinTimeElapsed(true);
    }, 4500); // 4.5 seconds to showcase the premium scanner flow
    return () => clearTimeout(timer);
  }, []);

  // Transition when API is ready and min time elapsed
  useEffect(() => {
    if (apiDone && minTimeElapsed && !errorMsg) {
      onComplete();
    }
  }, [apiDone, minTimeElapsed, errorMsg, onComplete]);

  // Status messages rotation timer (3.5s)
  useEffect(() => {
    const interval = setInterval(() => {
      setFadeStatus(false);
      setTimeout(() => {
        setStatusIndex((prev) => (prev + 1) % STATUS_MESSAGES.length);
        setFadeStatus(true);
      }, 300);
    }, 3500);

    return () => clearInterval(interval);
  }, []);

  // Fashion facts rotation timer (5s)
  useEffect(() => {
    const interval = setInterval(() => {
      setFadeFact(false);
      setTimeout(() => {
        setFactIndex((prev) => (prev + 1) % FASHION_FACTS.length);
        setFadeFact(true);
      }, 300);
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  // One try-on per visit to this page: StrictMode re-runs effects, and the state set below would re-trigger this one
  const tryOnStartedRef = useRef(false);

  useEffect(() => {
    if (tryOnStartedRef.current) return;
    tryOnStartedRef.current = true;

    const runTryOn = async () => {
      try {
        if (!capturedImage) {
          throw new Error("No user image captured.");
        }
        if (!selectedOutfit) {
          throw new Error("No outfit selected.");
        }

        const outfitSrc = OUTFITS_MAP[selectedOutfit];
        if (!outfitSrc) {
          throw new Error(`Outfit asset not found for: ${selectedOutfit}`);
        }

        if (MOCK_MODE) {
          console.log("[FRONTEND TRYON] MOCK_MODE is on — skipping network call. Staying on this page.");
          // Intentionally do NOT call setApiDone(true) here, so the page never advances.
          // Flip MOCK_MODE to false (or remove it) when you want real API calls again.
          return;
        }

        console.log(`[FRONTEND TRYON] Fetching dress image and converting to base64...`);
        const dressBase64 = await convertImageToBase64(outfitSrc);

        console.log(`[FRONTEND TRYON] Sending request to virtual try-on backend at ${API_BASE_URL}/api/tryon...`);
        const response = await fetch(`${API_BASE_URL}/api/tryon`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            user_image: capturedImage,
            dress_image: dressBase64,
            selected_size: selectedSize || null,
            body_profile: bodyProfile,
            session_id: sessionId,
            garment_id: selectedOutfit,
          }),
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.detail || `Server responded with status ${response.status}`);
        }

        const data = await response.json();
        if (data.verification_failed || data.fallback_used) {
          // No fresh render was produced — never present it as a normal result.
          setErrorMsg(data.message || "We couldn't generate a preview — please try again.");
          return;
        }
        if (data.wont_fit && !data.result_image) {
          // A pre-selected size that won't fit and could not be depicted believably: say so, there is nothing to show
          setErrorMsg(data.message || "This size won't fit you.");
          return;
        }

        if (data.success && data.result_image) {
          console.log(`[FRONTEND TRYON] Successfully received generated image from backend!`);
          setGeneratedImage(data.result_image);
          const initialSize = data.selected_size || selectedSize || data.size_recommendation?.recommended_size;
          if (data.wont_fit) {
            // Never a fallback for other sizes
            if (initialSize) {
              setWontFitInCache(initialSize, { message: data.message, illustrationOnly: false, fitCheck: data.fit_check || null }, selectedOutfit);
            }
          } else {
            setLastGoodRender(data.result_image);
          }
          if (initialSize) {
            setSizeRenderInCache(initialSize, data.result_image, selectedOutfit);
          }
          if (data.qr_code_base64) {
            console.log(`[FRONTEND TRYON] Received QR code from backend.`);
            setQrCodeBase64(data.qr_code_base64);
          }

          // Populate size recommendation from tryon response if available
          if (data.size_recommendation) {
            const rec = data.size_recommendation;
            setRecommendedSize(rec.recommended_size);
            setAvailableSizes(rec.available_sizes || []);
            setSelectedSize(data.selected_size || rec.recommended_size);
            setSizeConfidence(rec.size_confidence);
            setSizeFitNotes(rec.fit_notes);
            setSizeFits(rec.size_fits || {});
            setSizeFitQuality(data.size_fit_quality || "perfect");
            console.log(`[FRONTEND SIZE] Size recommendation from tryon: ${rec.recommended_size}`);
          } else if (bodyProfile) {
            // Fallback: call /api/recommend-size separately
            try {
              const clothCategory = data.cloth_analysis?.category || "";
              console.log(`[FRONTEND SIZE] Calling /api/recommend-size for category: ${clothCategory}...`);
              const sizeResp = await fetch(`${API_BASE_URL}/api/recommend-size`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  body_profile: bodyProfile,
                  garment_category: clothCategory,
                }),
              });
              if (sizeResp.ok) {
                const sizeData = await sizeResp.json();
                if (sizeData.success && sizeData.size_recommendation) {
                  const rec = sizeData.size_recommendation;
                  setRecommendedSize(rec.recommended_size);
                  setAvailableSizes(rec.available_sizes || []);
                  setSelectedSize(rec.recommended_size);
                  setSizeConfidence(rec.size_confidence);
                  setSizeFitNotes(rec.fit_notes);
                  setSizeFits(rec.size_fits || {});
                  setSizeFitQuality("perfect");
                  console.log(`[FRONTEND SIZE] Recommended: ${rec.recommended_size} (${Math.round(rec.size_confidence * 100)}% confidence)`);
                }
              }
            } catch (sizeErr) {
              console.warn("[FRONTEND SIZE] Size recommendation request failed:", sizeErr);
            }
          }

          setApiDone(true);
        } else {
          throw new Error("Invalid response format from server.");
        }
      } catch (err) {
        const error = err as Error;
        console.error("Virtual Try-On API Call failed:", error);
        setErrorMsg(error.message || "An error occurred.");
      }
    };

    runTryOn();
  }, [capturedImage, selectedOutfit, fitState, bodyProfile, sessionId, setGeneratedImage, setLastGoodRender, setQrCodeBase64, setSelectedSize, setRecommendedSize, setAvailableSizes, setSizeFitQuality, setSizeConfidence, setSizeFitNotes]);

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
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "6vh 0 7vh 0",
        gap: "6vh",
      }}
    >
      <audio ref={audioRef} src={generatingVO} autoPlay />
      {/* Top Header Block */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: "1vh",
          width: "100%",
          textAlign: "center",
        }}
      >
        {/* Top Instruction */}
        <div
          style={{
            display: "flex",
            flexDirection: "row",
            alignItems: "center",
            padding: "0px",
            gap: "1.6vh", // ~16px
            opacity: 0.8,
          }}
        >
          {/* Horizontal Divider Left */}
          <div
            style={{
              width: "3.0vh", // ~30px
              height: "0.2vh", // ~2px
              background: "#5C4A3D",
            }}
          />

          {/* Container for Text */}
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "flex-start",
              padding: "0px",
            }}
          >
            <span
              style={{
                fontFamily: "'Poppins', sans-serif",
                fontWeight: 600,
                fontSize: "2.4vh", // ~24px
                lineHeight: "1.6vh", // ~16px
                display: "flex",
                alignItems: "center",
                letterSpacing: "0.12vh", // ~1.2px
                textTransform: "uppercase",
                color: "#5C4A3D",
              }}
            >
              SET UP
            </span>
          </div>

          {/* Horizontal Divider Right */}
          <div
            style={{
              width: "3.1vh", // ~31px
              height: "0.2vh", // ~2px
              background: "#5C4A3D",
            }}
          />
        </div>
      </div>

      {/* Center AI Fashion Scanner */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: "1.8vh",
        }}
      >
        <FashionScanner />

        {/* Status Card (Glassmorphism) */}
        <div
          style={{
            boxSizing: "border-box",
            display: "flex",
            flexDirection: "column",
            alignItems: "flex-start",
            padding: "2.4vh",
            gap: "1.6vh",
            width: "55vh", // Matches ~568px
            // background: "rgba(255, 255, 255, 0.15)",
            background: "transparent",
            // border: "0.1vh solid rgba(255, 255, 255, 0.3)",
            // boxShadow: "0px 0.8vh 3.2vh rgba(0, 0, 0, 0.05)",
            // backdropFilter: "blur(1vh)",
            // WebkitBackdropFilter: "blur(1vh)",
            borderRadius: "2.4vh",
            zIndex: 1,
          }}
        >
          {/* Container (Header) */}
          <div
            style={{
              display: "flex",
              flexDirection: "row",
              alignItems: "center",
              gap: "1.6vh",
              width: "100%",
            }}
          >
            {/* Border (Icon Container) */}

            <span
              style={{
                fontFamily: "'Poppins', sans-serif",
                fontWeight: 700,
                fontSize: "1.8vh",
                letterSpacing: "0.045vh",
                textTransform: "uppercase",
                color: "#7C5E43",
                transition: "opacity 0.3s ease",
                opacity: fadeStatus ? 1 : 0,
                flex: 1,
                textAlign: "center"
              }}
            >
              {STATUS_MESSAGES[statusIndex]}
            </span>
          </div>

          {/* Static Bottom Text */}
          <p
            style={{
              fontFamily: "'Poppins', sans-serif",
              fontWeight: 500,
              fontSize: "1.9vh",
              lineHeight: 1.5,
              letterSpacing: "0.045vh",
              color: "rgba(58, 49, 42, 0.7)",
              margin: 0,
              textAlign: "center",
              transition: "opacity 0.3s ease",
              opacity: fadeFact ? 1 : 0,
            }}
          >
            {FASHION_FACTS[factIndex].text}
          </p>
        </div>
      </div>

      {/* Bottom Interaction Area */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: "2vh", // ~20px
        }}
      >
        {/* Home Button */}
        {onHome && (
          <NavigationButton
            type="home"
            onClick={onHome}
            style={{ width: "7.05vh", height: "7.05vh" }}
          />
        )}

        <span
          style={{
            fontFamily: "'Poppins', sans-serif",
            fontWeight: 500,
            fontSize: "2.88vh", // ~28.8px
            lineHeight: "2.9vh", // ~29px
            letterSpacing: "0.0504vh", // ~0.504px
            textTransform: "uppercase",
            color: "#5C4A3D",
            textAlign: "center",
            display: "flex",
            alignItems: "center",
          }}
        >
          WE ARE ALMOST THERE
        </span>

        {/* Safety/Info Text */}
        <p
          style={{
            fontFamily: "'Poppins', sans-serif",
            fontWeight: 500,
            fontSize: "1.8vh", // ~20px
            lineHeight: 1.44,
            textAlign: "center",
            letterSpacing: "0.05vh",
            color: "#5C4A3D",
            margin: 0,
            maxWidth: "35vh", // Matches ~346px relative width
          }}
        >
          Our AI stylist is preparing your personalized fashion experience.
        </p>
      </div>

      {/* Backend Error Display (if any) */}
      {errorMsg && (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "1vh", marginTop: "2vh" }}>
          <p
            style={{
              fontFamily: "'Poppins', sans-serif",
              color: "#EF4444",
              fontSize: "1.05vh",
              textAlign: "center",
              lineHeight: 1.4,
              maxWidth: "36vh",
              margin: 0,
            }}
          >
            ⚠️ {errorMsg}
          </p>
          {onBack && (
            <button
              onClick={onBack}
              style={{
                marginTop: "1vh",
                padding: "0.8vh 1.5vh",
                background: "rgba(255, 43, 214, 0.2)",
                border: "0.15vh solid #FF2BD6",
                borderRadius: "1.0vh",
                color: "#F5F7FF",
                fontFamily: "'Montserrat', sans-serif",
                fontWeight: 600,
                fontSize: "1.2vh",
                cursor: "pointer",
                letterSpacing: "0.1vh",
                outline: "none",
              }}
            >
              GO BACK
            </button>
          )}
        </div>
      )}
    </div>
  );
}

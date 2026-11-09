import { useRef } from "react";
import PrimaryButton from "../components/PrimaryButton";
import VideoEntryCard, { TRY_ON_KEYFRAMES, FOOTWEAR_KEYFRAMES } from "../components/VideoEntryCard";
import avonLogo from "../assets/avon_logo.png";
import welcomeVideo from "../assets/welcome_video.mp4";

type PageProps = {
  onNext: () => void;
  onBeauty?: () => void;
};

export default function WelcomePage({ onNext, onBeauty }: PageProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  return (
    <div
      style={{
        flex: 1,
        width: "100%",
        height: "100%",
        position: "relative",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        background: "transparent",
      }}
    >
      <video
        ref={videoRef}
        src={welcomeVideo}
        autoPlay
        muted
        loop
        playsInline
        style={{
          position: "absolute",
          top: "-12%",
          left: "-12%",
          width: "124%",
          height: "124%",
          objectFit: "cover",
          zIndex: 0,
        }}
      />
      {/* Replaces the video's "Try on" card with a tappable Clothing card */}
      <VideoEntryCard
        videoRef={videoRef}
        keyframes={TRY_ON_KEYFRAMES}
        label="Clothing"
        ariaLabel="Clothing try-on"
        icon={<SatinDressIcon />}
        onClick={onNext}
        sheenDelay="-2.2s"
      />
      {/* Replaces the video's "Footwear" card with a tappable Beauty card */}
      {onBeauty && (
        <VideoEntryCard
          videoRef={videoRef}
          keyframes={FOOTWEAR_KEYFRAMES}
          label={
            <span>
              Beauty
              {/* Bottom-right, level with the page dots under the label (card padding 0.75em / 0.9em, divided by the label's 0.78em font size) */}
              <img
                src={avonLogo}
                alt="AVON"
                style={{ position: "absolute", right: "0.96em", bottom: "1.15em", height: "2.6em", width: "2.6em", borderRadius: "0.22em", display: "block" }}
              />
            </span>
          }
          ariaLabel="AVON beauty studio"
          icon={<SatinLipstickIcon />}
          onClick={onBeauty}
        />
      )}
      {/* Frame 55 Overlay */}
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "2% 5% 10% 5%",
          background: "transparent",
          zIndex: 10,
          animation: "fadeIn 0.8s ease-out forwards",
        }}
      >
        {/* Top Section: Title Block — positioned at very top, does NOT cover the dress */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: "0.38vh",
            width: "100%",
            marginTop: "-2.5vh",
          }}
        >
          {/* WELCOME TO */}
          <span
            style={{
              fontFamily: "'Poppins', sans-serif",
              fontWeight: 400,
              fontSize: "1.51vh",
              lineHeight: 1.4,
              letterSpacing: "0.25vh",
              textTransform: "uppercase",
              color: "#5C4A3D",
              textAlign: "center",
            }}
          >
            WELCOME TO
          </span>

          {/* THE VIRTUAL TRIAL ROOM — single line, mixed colours */}
          <h1
            style={{
              fontFamily: "'Montserrat', sans-serif",
              fontWeight: 800,
              fontSize: "2.65vh",
              lineHeight: "1.2",
              letterSpacing: "0.063vh",
              textTransform: "uppercase",
              textAlign: "center",
              margin: 0,
              whiteSpace: "nowrap",
            }}
          >
            <span style={{ color: "#5C4A3D" }}>THE </span>
            <span
              style={{
                color: "#5C4A3D",
                textShadow: "0px 0px 0.88vh rgba(194, 162, 133, 0.6)",
              }}
            >
              VIRTUAL TRIAL ROOM
            </span>
          </h1>

          {/* Gradient underline */}
          <div
            style={{
              width: "6.3vh",
              border: "5px solid",
              borderImageSource: "linear-gradient(90deg, #BDA795 0%, #F6F3EF 47.6%, #C8A380 100%)",
              borderImageSlice: 1,
              marginTop: "0.25vh",
            }}
          />
        </div>

        {/* CTA Button Block */}
        <div
          style={{
            width: "100%",
            display: "flex",
            justifyContent: "center",
            gap: "1.5vh",
            marginBottom: "1.26vh",
          }}
        >
          <PrimaryButton
            variant="welcome"
            onClick={(e: React.MouseEvent) => {
              e.stopPropagation();
              onNext();
            }}
          >
            CLOTHING
          </PrimaryButton>
          {onBeauty && (
            <PrimaryButton
              variant="welcome"
              onClick={(e: React.MouseEvent) => {
                e.stopPropagation();
                onBeauty();
              }}
            >
              BEAUTY
            </PrimaryButton>
          )}
        </div>
      </div>
    </div>
  );
}

// Champagne satin slip dress on a hanger, echoing the dress in the welcome video
function SatinDressIcon() {
  return (
    <svg viewBox="0 0 100 100" style={{ width: "58%", height: "auto" }} aria-hidden>
      <defs>
        <linearGradient id="satinDress" x1="0" x2="1">
          <stop offset="0" stopColor="#B8935F" />
          <stop offset="0.38" stopColor="#F1DDB8" />
          <stop offset="0.62" stopColor="#D9B988" />
          <stop offset="1" stopColor="#A9834F" />
        </linearGradient>
      </defs>
      <ellipse cx="50" cy="95" rx="20" ry="2.5" fill="rgba(60,45,35,0.16)" />
      {/* Hanger */}
      <path d="M50 4 Q55 4 55 8 Q55 11 50 13" stroke="#8C7A68" strokeWidth="1.6" fill="none" strokeLinecap="round" />
      <path d="M50 13 L33 22 M50 13 L67 22" stroke="#8C7A68" strokeWidth="1.6" strokeLinecap="round" />
      {/* Straps */}
      <path d="M38 22 L40 34 M62 22 L60 34" stroke="#C9A877" strokeWidth="1.4" strokeLinecap="round" />
      {/* Bodice, waist and flared, asymmetric skirt */}
      <path
        d="M40 34 Q50 38 60 34 L62 50 Q50 53 38 50 Z
           M38 50 Q50 53 62 50 Q70 70 74 90 Q62 94 50 91 Q38 94 26 88 Q31 70 38 50 Z"
        fill="url(#satinDress)"
      />
      {/* Satin highlights and folds */}
      <path d="M45 37 L44 49" stroke="rgba(255,255,255,0.55)" strokeWidth="1.4" strokeLinecap="round" />
      <path d="M44 54 Q41 72 36 88" stroke="rgba(255,255,255,0.5)" strokeWidth="1.6" fill="none" strokeLinecap="round" />
      <path d="M56 54 Q60 72 63 90" stroke="rgba(120,90,50,0.35)" strokeWidth="1.2" fill="none" strokeLinecap="round" />
      <path d="M38 50 Q50 53 62 50" stroke="rgba(120,90,50,0.35)" strokeWidth="1" fill="none" />
    </svg>
  );
}

// Champagne lipstick and compact, in the same satin palette as the dress icon
function SatinLipstickIcon() {
  return (
    <svg viewBox="0 0 100 100" style={{ width: "62%", height: "auto" }} aria-hidden>
      <defs>
        <linearGradient id="satinCase" x1="0" x2="1">
          <stop offset="0" stopColor="#B8935F" />
          <stop offset="0.4" stopColor="#F1DDB8" />
          <stop offset="0.65" stopColor="#D9B988" />
          <stop offset="1" stopColor="#A9834F" />
        </linearGradient>
        <linearGradient id="satinBullet" x1="0" x2="1">
          <stop offset="0" stopColor="#8E3B3F" />
          <stop offset="0.45" stopColor="#C9746F" />
          <stop offset="1" stopColor="#7A2E33" />
        </linearGradient>
        <radialGradient id="satinPowder" cx="0.4" cy="0.35" r="0.7">
          <stop offset="0" stopColor="#F3E2CC" />
          <stop offset="1" stopColor="#D8B994" />
        </radialGradient>
      </defs>
      <ellipse cx="50" cy="92" rx="34" ry="3" fill="rgba(60,45,35,0.16)" />
      {/* Open compact */}
      <ellipse cx="66" cy="84" rx="22" ry="7" fill="url(#satinCase)" />
      <ellipse cx="66" cy="81" rx="19" ry="5.5" fill="url(#satinPowder)" />
      <path d="M57 79 Q63 77 70 79" stroke="rgba(255,255,255,0.6)" strokeWidth="1.2" fill="none" strokeLinecap="round" />
      {/* Lipstick */}
      <rect x="25" y="56" width="22" height="34" rx="2.5" fill="url(#satinCase)" />
      <rect x="27" y="45" width="18" height="12" rx="1.5" fill="url(#satinCase)" opacity="0.9" />
      <path d="M29 45 L29 28 Q29 20 43 16 L43 45 Z" fill="url(#satinBullet)" />
      <path d="M32 43 L32 28 Q32 24 37 21" stroke="rgba(255,255,255,0.45)" strokeWidth="1.5" fill="none" strokeLinecap="round" />
      <path d="M30 60 L30 86" stroke="rgba(255,255,255,0.5)" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}

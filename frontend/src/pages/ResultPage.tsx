import { useState, useEffect, useCallback } from "react";
import PrimaryButton from "../components/PrimaryButton";
import NavigationButton from "../components/NavigationButton";
import SizeSelector from "../components/SizeSelector";
import { getSizeQuality } from "../utils/sizeFit";
import WontFitNotice from "../components/WontFitNotice";
import { useTrialRoom } from "../context/TrialRoomContext";
import { API_BASE_URL } from "../utils/api";

import male1 from "../assets/dress/male1.png";
import male2 from "../assets/dress/male2.png";
import male3 from "../assets/dress/male3.png";
import female1 from "../assets/dress/female1.png";
import female2 from "../assets/dress/female2.png";
import female3 from "../assets/dress/female3.png";
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

const OUTFITS_MAP: Record<string, string> = {
  female1,
  female2,
  female3,
  kurti1,
  salwar_suit1,
  lehenga1,
  kerala_saree1,
  kurta1,
  sherwani1,
  royal_sherwani1,
  kurta_dhoti1,
  kurta_pajama1,
  mundu_shirt1,
  male1,
  male2,
  male3,
  blazer1,
  blazer2,
  outfit1,
  outfit2,
  outfit4,
};

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

type PageProps = {
  onTrySomethingElse: () => void;
  onLikeOutfit: () => void;
  onBack: () => void;
  onHome: () => void;
};

export default function ResultPage({
  onTrySomethingElse,
  onLikeOutfit,
  onBack,
  onHome,
}: PageProps) {
  const {
    sessionId,
    capturedImage,
    selectedOutfit,
    generatedImage,
    fitState,
    bodyProfile,
    selectedSize,
    recommendedSize,
    availableSizes,
    sizeFitQuality,
    sizeConfidence,
    sizeFitNotes,
    sizeRendersCache,
    sizeFits,
    wontFitCache,
    setGeneratedImage,
    setLastGoodRender,
    setQrCodeBase64,
    setSelectedSize,
    setSizeFitQuality,
    setSizeRenderInCache,
    setWontFitInCache,
  } = useTrialRoom();

  const [isUpdatingSize, setIsUpdatingSize] = useState(false);
  const [statusHint, setStatusHint] = useState<string | null>(null);
  // Size whose image is on screen (null until a size change: the initial render's size); while a new size renders
  // the previous image stays visible
  const [changedToSize, setDisplayedSize] = useState<string | null>(null);
  const displayedSize = changedToSize ?? selectedSize ?? recommendedSize;

  const sizesList = availableSizes.length > 0 ? availableSizes : ["XS", "S", "M", "L", "XL", "XXL"];
  const currentOutfit = selectedOutfit || "female1";
  const cacheKeyFor = (size: string) => `${currentOutfit}_${size.toUpperCase().trim()}`;
  const displayedWontFit = displayedSize ? wontFitCache[cacheKeyFor(displayedSize)] : undefined;

  // Auto-detect & pre-select recommended size on page load
  useEffect(() => {
    if (!selectedSize && recommendedSize) {
      setSelectedSize(recommendedSize);
      setSizeFitQuality("perfect");
    }
  }, [selectedSize, recommendedSize, setSelectedSize, setSizeFitQuality]);

  /**
   * Handle a size tap. Every size is a real render of this garment on the visitor's own body (a new Gemini
   * generation on first view, then cached per garment for the session). "Won't fit" is its own state: the message
   * is always shown, over the strained render or, when no believable render exists, over a strain illustration.
   */
  const handleSizeChange = useCallback(
    async (newSize: string) => {
      if (newSize === selectedSize || isUpdatingSize) return;
      const previousSize = displayedSize;
      setDisplayedSize(previousSize); // the image on screen stays this size until the new one is ready
      setSelectedSize(newSize);
      setStatusHint(null);
      // Fit label is known before the render arrives (body-checked by the backend when available)
      setSizeFitQuality(getSizeQuality(newSize, recommendedSize, sizesList, sizeFits));

      // Instant cache: a size already viewed for this garment switches with no API call
      const cacheKey = cacheKeyFor(newSize);
      const cachedRender = sizeRendersCache[cacheKey];
      const cachedWontFit = wontFitCache[cacheKey];
      if (cachedRender || cachedWontFit?.illustrationOnly) {
        if (cachedRender) setGeneratedImage(cachedRender);
        setDisplayedSize(newSize);
        if (!cachedWontFit) {
          setStatusHint(`Previewing size ${newSize}`);
          setTimeout(() => setStatusHint(null), 3000);
        }
        return;
      }

      if (!capturedImage) return;

      setIsUpdatingSize(true);

      try {
        const outfitKey = selectedOutfit || "female1";
        const outfitSrc = OUTFITS_MAP[outfitKey];
        if (!outfitSrc) throw new Error("Selected outfit asset not found.");

        const dressBase64 = await convertImageToBase64(outfitSrc);
        const response = await fetch(`${API_BASE_URL}/api/tryon`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_image: capturedImage,
            dress_image: dressBase64,
            selected_size: newSize,
            body_profile: bodyProfile,
            session_id: sessionId,
            garment_id: outfitKey,
          }),
        });

        if (!response.ok) {
          const err = await response.json().catch(() => ({}));
          throw new Error(err.detail || "Server error");
        }

        const data = await response.json();
        if (data.size_fit_quality) setSizeFitQuality(data.size_fit_quality);

        if (data.wont_fit) {
          // Intentional state, never an error: message always shown; the strained render when believable
          setWontFitInCache(
            newSize,
            { message: data.message, illustrationOnly: !!data.illustration_only, fitCheck: data.fit_check || null },
            currentOutfit
          );
          if (data.result_image) {
            setGeneratedImage(data.result_image);
            setSizeRenderInCache(newSize, data.result_image, currentOutfit);
            if (data.qr_code_base64) setQrCodeBase64(data.qr_code_base64);
          }
          setDisplayedSize(newSize);
        } else if (data.fallback_used) {
          // Backend could not produce a new render for this size and returned the previous one.
          // Keep the current image and don't cache it as this size's render.
          setSelectedSize(previousSize);
          setStatusHint(data.message || `Couldn't generate size ${newSize} — please try again.`);
        } else if (data.success && data.result_image) {
          setGeneratedImage(data.result_image);
          setLastGoodRender(data.result_image);
          setSizeRenderInCache(newSize, data.result_image, currentOutfit);
          if (data.qr_code_base64) setQrCodeBase64(data.qr_code_base64);
          setDisplayedSize(newSize);
          setStatusHint(`Previewing size ${newSize}`);
        } else {
          setSelectedSize(previousSize);
          setStatusHint(data.message || "We couldn't generate a preview — please try again.");
        }
      } catch (err) {
        console.error("Size re-generation error:", err);
        setSelectedSize(previousSize);
        setStatusHint("Could not update size preview. Please try again.");
      } finally {
        setIsUpdatingSize(false);
        setTimeout(() => setStatusHint(null), 4000);
      }
    },
    // cacheKeyFor / sizesList derive from selectedOutfit / availableSizes, listed here
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [
      selectedSize,
      displayedSize,
      isUpdatingSize,
      availableSizes,
      recommendedSize,
      capturedImage,
      selectedOutfit,
      bodyProfile,
      sessionId,
      sizeRendersCache,
      sizeFits,
      wontFitCache,
      setSelectedSize,
      setSizeFitQuality,
      setGeneratedImage,
      setLastGoodRender,
      setQrCodeBase64,
      setSizeRenderInCache,
      setWontFitInCache,
    ]
  );

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
      {/* Top Header */}
      <div
        style={{
          display: "flex",
          flexDirection: "row",
          alignItems: "center",
          justifyContent: "center",
          gap: "1.5vh",
          width: "100%",
          marginBottom: "2vh",
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
          YOUR <span style={{ color: "rgba(92, 74, 61, 0.4)", marginLeft: "0.5vh" }}>OUTFIT IS READY!</span>
        </h1>

        {/* Right Divider */}
        <div style={{ width: "3.0vh", height: "0.2vh", background: "#5C4A3D" }} />
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

          {/* New size rendering: the previous image stays fully visible until the new one is ready */}
          {isUpdatingSize && (
            <div
              data-testid="size-rendering"
              style={{
                position: "absolute",
                top: "1vh",
                left: "50%",
                transform: "translateX(-50%)",
                display: "flex",
                alignItems: "center",
                gap: "0.8vh",
                padding: "0.6vh 1.4vh",
                background: "rgba(255, 255, 255, 0.88)",
                border: "0.1vh solid rgba(183, 149, 121, 0.45)",
                borderRadius: "999vh",
                boxShadow: "0px 0.3vh 1vh rgba(92, 74, 61, 0.15)",
                whiteSpace: "nowrap",
                zIndex: 17,
              }}
            >
              <div
                style={{
                  width: "1.6vh",
                  height: "1.6vh",
                  border: "0.25vh solid rgba(183, 149, 121, 0.3)",
                  borderTop: "0.25vh solid #5C4A3D",
                  borderRadius: "50%",
                  animation: "spin 0.9s linear infinite",
                }}
              />
              <span
                style={{
                  fontFamily: "'Poppins', sans-serif",
                  fontWeight: 600,
                  fontSize: "1.1vh",
                  color: "#5C4A3D",
                  letterSpacing: "0.04vh",
                }}
              >
                {`Rendering size ${selectedSize || ""} on you…`}
              </span>
            </div>
          )}

          {/* Won't fit: a distinct state with its own message (over the strained render, or an illustration) */}
          {displayedWontFit && !isUpdatingSize && displayedSize === selectedSize && (
            <WontFitNotice
              message={displayedWontFit.message}
              recommendedSize={recommendedSize}
              fitCheck={displayedWontFit.fitCheck}
              illustrationOnly={displayedWontFit.illustrationOnly}
              onTryRecommended={() => recommendedSize && handleSizeChange(recommendedSize)}
            />
          )}
        </div>

        {/* Right-Side Floating Actions */}
        <div
          style={{
            position: "absolute",
            right: "-10.05vh",
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
            onClick={onLikeOutfit}
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: "12.1vh",
              height: "12.1vh",
              background: "transparent",
              border: "none",
              cursor: "pointer",
              outline: "none",
              padding: 0,
              transition: "transform 0.1s ease",
            }}
            onMouseDown={(e) => (e.currentTarget.style.transform = "scale(0.95)")}
            onMouseUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
            onMouseLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
          >
            <svg width="100%" height="100%" viewBox="0 0 192 192" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M40 56C40 25.0721 65.0721 0 96 0C126.928 0 152 25.0721 152 56C152 86.9279 126.928 112 96 112C65.0721 112 40 86.9279 40 56Z" fill="url(#paint0_linear_155_611)" fillOpacity="0.9" />
              <g filter="url(#filter1_dd_155_611)">
                <rect x="40" width="112" height="112" rx="56" fill="white" fillOpacity="0.01" shapeRendering="crispEdges" />
              </g>
              <path d="M114.846 57.6584L100.783 71.7209C100.343 72.1612 99.7457 72.4085 99.123 72.4085C98.5004 72.4085 97.9032 72.1612 97.4629 71.7209C97.0226 71.2806 96.7752 70.6834 96.7752 70.0607C96.7752 69.438 97.0226 68.8409 97.4629 68.4006L107.523 58.3439H78.8125C78.1909 58.3439 77.5948 58.097 77.1552 57.6574C76.7157 57.2179 76.4688 56.6218 76.4688 56.0002C76.4688 55.3786 76.7157 54.7824 77.1552 54.3429C77.5948 53.9033 78.1909 53.6564 78.8125 53.6564H107.523L97.4668 43.5939C97.0265 43.1536 96.7791 42.5564 96.7791 41.9338C96.7791 41.3111 97.0265 40.7139 97.4668 40.2736C97.9071 39.8333 98.5043 39.5859 99.127 39.5859C99.7496 39.5859 100.347 39.8333 100.787 40.2736L114.85 54.3361C115.068 54.5541 115.241 54.8132 115.36 55.0984C115.478 55.3836 115.538 55.6893 115.538 55.998C115.538 56.3067 115.476 56.6123 115.357 56.8973C115.239 57.1822 115.065 57.4408 114.846 57.6584Z" fill="white" />
              <defs>
                <filter id="filter1_dd_155_611" x="0" y="0" width="192" height="192" filterUnits="userSpaceOnUse" colorInterpolationFilters="sRGB">
                  <feFlood floodOpacity="0" result="BackgroundImageFix" />
                  <feColorMatrix in="SourceAlpha" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" result="hardAlpha" />
                  <feMorphology radius="12" operator="erode" in="SourceAlpha" result="effect1_dropShadow_155_611" />
                  <feOffset dy="16" />
                  <feGaussianBlur stdDeviation="10" />
                  <feComposite in2="hardAlpha" operator="out" />
                  <feColorMatrix type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.1 0" />
                  <feBlend mode="normal" in2="BackgroundImageFix" result="effect1_dropShadow_155_611" />
                  <feColorMatrix in="SourceAlpha" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" result="hardAlpha" />
                  <feMorphology radius="10" operator="erode" in="SourceAlpha" result="effect2_dropShadow_155_611" />
                  <feOffset dy="40" />
                  <feGaussianBlur stdDeviation="25" />
                  <feComposite in2="hardAlpha" operator="out" />
                  <feColorMatrix type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.1 0" />
                  <feBlend mode="normal" in2="effect1_dropShadow_155_611" result="effect2_dropShadow_155_611" />
                  <feBlend mode="normal" in="SourceGraphic" in2="effect2_dropShadow_155_611" result="shape" />
                </filter>
                <linearGradient id="paint0_linear_155_611" x1="171" y1="56" x2="52" y2="56" gradientUnits="userSpaceOnUse">
                  <stop stopColor="#5C4A3D" />
                  <stop offset="1" stopColor="#C29C81" />
                </linearGradient>
              </defs>
            </svg>
          </button>

          {/* Try On Others Button */}
          <button
            onClick={onTrySomethingElse}
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: "12.1vh",
              height: "12.1vh",
              background: "transparent",
              border: "none",
              cursor: "pointer",
              outline: "none",
              padding: 0,
              transition: "transform 0.1s ease",
            }}
            onMouseDown={(e) => (e.currentTarget.style.transform = "scale(0.95)")}
            onMouseUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
            onMouseLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
          >
            <svg width="100%" height="100%" viewBox="0 0 192 192" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M40 56C40 25.0721 65.0721 0 96 0C126.928 0 152 25.0721 152 56C152 86.9279 126.928 112 96 112C65.0721 112 40 86.9279 40 56Z" fill="url(#paint0_linear_155_615)" fillOpacity="0.9"/>
              <g filter="url(#filter1_dd_155_615)">
                <rect x="40" width="112" height="112" rx="56" fill="white" fillOpacity="0.01" shapeRendering="crispEdges"/>
              </g>
              <path d="M90.3529 40.2459C91.7508 40.094 91.8969 41.4827 92.279 42.202C92.5569 42.7319 92.9562 43.1887 93.4442 43.535C94.2469 44.0962 95.2395 44.3158 96.2044 44.1458C97.402 43.9317 98.418 43.1425 98.9215 42.0349C98.9704 41.9284 99.0653 41.695 99.0985 41.5849C99.6012 39.9082 100.773 40.1979 102.137 40.4683C103.154 40.6701 104.088 40.9232 105.059 41.3701C105.665 41.6514 106.238 41.9978 106.77 42.4032C110.438 45.1816 112.444 50.9337 113.083 55.3499C113.382 57.4198 111.801 57.5569 110.265 57.7886C109.138 57.9585 107.995 58.1886 106.854 58.3417C106.886 59.2038 106.851 60.3194 106.852 61.2019L106.859 67.1944L106.861 70.6132C106.863 72.3066 107.179 74.0159 104.886 74.1648L92.5119 74.1667L88.2614 74.1691C87.3426 74.1694 85.9124 74.3175 85.0983 73.8827C84.8531 73.7519 84.4217 73.2495 84.3625 72.9542C84.2097 72.1917 84.2584 71.364 84.2635 70.5839L84.2802 67.8753L84.2769 58.3425C83.8143 58.2823 83.3278 58.1987 82.8682 58.1183C81.7127 57.8956 80.426 57.7765 79.2897 57.5011C77.5464 57.0788 78.0234 55.1819 78.2735 53.9191C78.9335 50.5872 80.2836 47.0759 82.336 44.3718C84.3505 41.7178 87.1901 40.6213 90.3529 40.2459ZM81.032 54.8471C82.1096 55.0203 83.1882 55.1874 84.2677 55.3484C84.2357 54.0896 83.8058 51.6605 85.7046 51.6693C86.1032 51.6725 86.4822 51.8424 86.7497 52.138C87.253 52.6906 87.1187 53.8042 87.1199 54.5348C87.1181 55.4808 87.124 56.4269 87.1374 57.373C87.169 60.0569 87.1796 62.7413 87.1692 65.4257C87.1558 67.3749 87.1116 69.2968 87.1195 71.2499C90.1586 71.2796 93.1977 71.2874 96.2372 71.2737C98.8298 71.2851 101.423 71.2796 104.015 71.2566L104.009 69.7124C103.972 65.6612 103.965 61.6101 103.991 57.5585L103.997 54.5489C103.998 53.7883 103.848 52.6827 104.391 52.1141C104.661 51.834 105.031 51.6721 105.42 51.664C106.506 51.6396 106.928 52.5647 106.897 53.5019C106.877 54.0987 106.85 54.7757 106.873 55.3671L110.103 54.85C109.345 51.4216 107.712 46.4776 104.606 44.4823C103.7 43.8999 102.558 43.5497 101.501 43.3713C100.647 45.0572 99.4059 46.2245 97.5883 46.8213C95.9227 47.3549 94.1137 47.2101 92.5542 46.4185C91.3749 45.8141 90.5074 44.9776 89.8628 43.8313C89.7259 43.5879 89.6579 43.3466 89.4522 43.4018C87.0948 43.8655 85.5071 44.8255 84.163 46.8547C82.6547 49.1315 81.5565 52.1856 81.032 54.8471Z" fill="#FEFEFE"/>
                <path d="M94.1989 20.005C94.2974 20.0007 94.3958 19.9992 94.4946 20.0004C95.2993 20.017 96.0442 20.8243 96.631 21.3372L99.0134 23.4388L101.66 25.7695C102.011 26.0783 102.758 26.7017 103.062 27.0244C103.4 27.3853 103.58 27.8665 103.561 28.3609C103.542 28.8268 103.321 29.2741 102.972 29.5731C102.459 30.0123 101.888 30.4367 101.352 30.8486L98.1259 33.3017C97.5821 33.7105 97.0431 34.1249 96.5083 34.5448C95.8778 35.0431 95.1954 35.7458 94.3419 35.6921C93.9208 35.6654 93.5271 35.4722 93.2485 35.1552C92.7331 34.5687 92.707 33.486 93.2536 32.976C94.4669 31.8446 95.9532 30.8226 97.2286 29.762C92.5588 29.4458 87.5681 30.5191 83.3721 32.5642C77.1356 35.6469 72.3667 41.065 70.1005 47.6423C69.1916 50.2373 68.7297 52.9449 68.6185 55.6848C68.5879 56.4379 68.6838 57.2113 68.1039 57.7976C67.8004 58.1039 67.3864 58.2746 66.9554 58.2715C66.5303 58.2695 66.1236 58.0973 65.8264 57.7933C65.2806 57.2348 65.3551 56.3867 65.3679 55.6371C65.5008 47.8756 68.8368 40.0953 74.389 34.684C79.1401 30.0058 85.3431 27.0856 91.9761 26.4047C93.9067 26.198 95.2501 26.2112 97.1427 26.235C95.9028 25.0733 94.5407 23.9914 93.3255 22.8028C92.8324 22.3204 92.6974 21.4673 93.0306 20.8612C93.3099 20.3529 93.6665 20.1674 94.1989 20.005Z" fill="#FEFEFE"/>
                <path d="M68.5195 64.2227C69.2298 64.2828 69.6175 64.5355 70.2467 64.8352L72.3018 65.816L75.416 67.3184C76.0622 67.6297 76.792 67.9547 77.3957 68.3305C77.6014 68.4582 77.766 68.7199 77.865 68.9402C78.0512 69.3461 78.0595 69.8109 77.8879 70.2234C77.5933 70.9219 76.8097 71.3645 76.0769 71.1727C75.4907 71.0191 74.8123 70.6492 74.2631 70.3852C73.4288 69.9813 72.5919 69.5828 71.7524 69.1898C72.1826 69.8969 72.5327 70.5637 72.997 71.2836C74.474 73.5781 76.2819 75.6418 78.3622 77.4082C86.0258 83.966 96.5766 86.284 106.004 82.2348C107.134 81.7551 108.229 81.1965 109.28 80.5633C110.228 79.9809 111.415 78.7598 112.545 79.8074C112.882 80.1184 113.071 80.5578 113.066 81.0168C113.065 81.5645 112.828 81.968 112.447 82.3367C111.591 82.9398 110.8 83.4652 109.884 83.9766C103.49 87.5457 95.411 88.4457 88.3749 86.4461C80.164 84.1449 73.2111 78.6621 69.059 71.2141C68.8481 72.0262 68.6908 72.8629 68.4873 73.6777C68.2245 74.7293 68.2347 76.0031 66.9614 76.3531C66.556 76.4652 66.1224 76.4062 65.7617 76.1902C65.0893 75.7871 64.891 75.0809 65.0538 74.352C65.7076 71.4258 66.3622 68.4981 67.0443 65.5785C67.2415 64.7344 67.6939 64.3891 68.5195 64.2227Z" fill="#FEFEFE"/>
                <path d="M111.703 30.8306C112.1 30.7692 112.523 30.8673 112.86 31.074C113.827 31.667 114.839 32.4583 115.684 33.2051C119.907 36.9656 123.014 41.8149 124.666 47.2225C127.219 55.7394 126.175 63.6831 121.991 71.4613L124.851 70.4683C125.614 70.2062 126.631 69.712 127.407 70.0589C127.793 70.2363 128.091 70.564 128.231 70.9656C128.384 71.398 128.354 71.789 128.163 72.198C127.897 72.7812 127.458 72.9937 126.892 73.1956C126.167 73.4538 125.444 73.7128 124.721 73.9726C122.764 74.6847 120.804 75.3855 118.84 76.0753C117.99 76.3488 117.297 76.2374 116.727 75.457C116.632 75.3269 116.525 74.8339 116.503 74.662C116.13 71.7984 115.719 68.9386 115.368 66.0722C115.297 65.4874 115.269 65.0488 115.66 64.5495C116.471 63.5081 118.165 63.8742 118.45 65.1734C118.585 65.7871 118.64 66.3812 118.716 66.9835C118.848 68.0249 118.999 69.0585 119.107 70.0949C119.439 69.416 119.811 68.766 120.14 68.073C120.938 66.4007 121.569 64.6538 122.025 62.8574C124.342 53.5176 121.572 43.6474 114.732 36.8773C113.96 36.109 113.138 35.3922 112.272 34.7315C111.613 34.2196 110.587 33.6538 110.382 32.8314C110.275 32.3953 110.353 31.9346 110.596 31.5574C110.878 31.1192 111.22 30.9436 111.703 30.8306Z" fill="#FEFEFE"/>
                <defs>
                  <filter id="filter1_dd_155_615" x="0" y="0" width="192" height="192" filterUnits="userSpaceOnUse" colorInterpolationFilters="sRGB">
                    <feFlood floodOpacity="0" result="BackgroundImageFix"/>
                    <feColorMatrix in="SourceAlpha" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" result="hardAlpha"/>
                    <feMorphology radius="12" operator="erode" in="SourceAlpha" result="effect1_dropShadow_155_615"/>
                    <feOffset dy="16"/>
                    <feGaussianBlur stdDeviation="10"/>
                    <feComposite in2="hardAlpha" operator="out"/>
                    <feColorMatrix type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.1 0"/>
                    <feBlend mode="normal" in2="BackgroundImageFix" result="effect1_dropShadow_155_615"/>
                    <feColorMatrix in="SourceAlpha" type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0" result="hardAlpha"/>
                    <feMorphology radius="10" operator="erode" in="SourceAlpha" result="effect2_dropShadow_155_615"/>
                    <feOffset dy="40"/>
                    <feGaussianBlur stdDeviation="25"/>
                    <feComposite in2="hardAlpha" operator="out"/>
                    <feColorMatrix type="matrix" values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.1 0"/>
                    <feBlend mode="normal" in2="effect1_dropShadow_155_615" result="effect2_dropShadow_155_615"/>
                    <feBlend mode="normal" in="SourceGraphic" in2="effect2_dropShadow_155_615" result="shape"/>
                  </filter>
                  <linearGradient id="paint0_linear_155_615" x1="171" y1="56" x2="52" y2="56" gradientUnits="userSpaceOnUse">
                    <stop stopColor="#5C4A3D"/>
                    <stop offset="1" stopColor="#C29C81"/>
                  </linearGradient>
                </defs>
              </svg>
            </button>
          </div>
        </div>

      {/* 6-Option Size Selector replacing 3-Option FitFinder Style Selector */}
      <SizeSelector
        availableSizes={availableSizes.length > 0 ? availableSizes : ["XS", "S", "M", "L", "XL", "XXL"]}
        selectedSize={selectedSize || recommendedSize}
        recommendedSize={recommendedSize}
        sizeFitQuality={sizeFitQuality}
        sizeConfidence={sizeConfidence}
        sizeFitNotes={sizeFitNotes}
        sizeFits={sizeFits}
        onChange={handleSizeChange}
        disabled={isUpdatingSize}
        compact
        style={{ margin: "0.4vh 0 0.2vh 0", zIndex: 12 }}
      />

      {/* Real-time Status / Fallback Notice Hint */}
      {statusHint && (
        <div
          style={{
            padding: "0.6vh 1.8vh",
            background: "rgba(92, 74, 61, 0.08)",
            border: "0.1vh solid rgba(183, 149, 121, 0.3)",
            borderRadius: "999vh",
            maxWidth: "42vh",
            textAlign: "center",
            margin: "0.3vh 0",
          }}
        >
          <span
            style={{
              fontFamily: "'Poppins', sans-serif",
              fontWeight: 500,
              fontSize: "1.1vh",
              color: "#5C4A3D",
              lineHeight: 1.2,
            }}
          >
            {statusHint}
          </span>
        </div>
      )}

      {/* Bottom Button Area */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          width: "100%",
          marginTop: "1.0vh",
        }}
      >
        {/* Back and Home Buttons centered below */}
        <div style={{ display: "flex", justifyContent: "center", gap: "3vh", marginTop: "1.5vh" }}>
          <NavigationButton
            type="home"
            onClick={onHome}
            style={{ width: "7.05vh", height: "7.05vh" }}
          />
        </div>

        {/* Texts Area */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            marginTop: "2vh",
            gap: "1.5vh",
          }}
        >
          <span
            style={{
              fontFamily: "'Poppins', sans-serif",
              fontWeight: 500,
              fontSize: "2.88vh",
              lineHeight: "2.9vh",
              letterSpacing: "0.0504vh",
              textTransform: "uppercase",
              color: "#5C4A3D",
              textAlign: "center",
            }}
          >
            STYLED SUCCESSFULLY
          </span>
          
          <span
            style={{
              fontFamily: "'Poppins', sans-serif",
              fontWeight: 500,
              fontSize: "2.0vh",
              lineHeight: "2.9vh",
              letterSpacing: "0.0504vh",
              color: "#5C4A3D",
              textAlign: "center",
              width: "44.4vh",
            }}
          >
            Your outfit has been prepared just for you. Take a closer look before continuing.
          </span>
        </div>
      </div>
    </div>
  );
}

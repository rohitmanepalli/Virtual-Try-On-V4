import { useEffect, useRef } from "react";
import NavigationButton from "../components/NavigationButton";
import PrimaryButton from "../components/PrimaryButton";
import OutfitCard from "../components/OutfitCard";
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
import { useTrialRoom } from "../context/TrialRoomContext";
import SizeSelector from "../components/SizeSelector";
import { API_BASE_URL } from "../utils/api";

type Outfit = {
  id: string;
  label: string;
  imageSrc: string;
  category: string;
  // One short line under the name: craft/occasion for traditional wear (a region only where the catalog records
  // one), fabric/cut/occasion for Western wear. Customer-facing cultural text: every line is reviewed before it
  // ships. Left out while an outfit's name and image disagree, and until its line has been reviewed.
  description?: string;
};

const OUTFITS: Outfit[] = [
  { id: "female1", label: "Smart Casual Shirt", imageSrc: female1, category: "shirt" },
  { id: "female2", label: "Floral Tunic", imageSrc: female2, category: "tunic" },
  { id: "female3", label: "Tailored Cream Shirt", imageSrc: female3, category: "shirt",
    description: "A tailored button-down for everyday polish" },
  { id: "kurti1", label: "Floral Panel Kurti", imageSrc: kurti1, category: "kurti",
    description: "Casual ethnic wear in block-printed florals, with palazzos" },
  { id: "salwar_suit1", label: "Embroidered Salwar Suit", imageSrc: salwar_suit1, category: "salwar_suit",
    description: "Festive wear in gold zari, with a sequinned dupatta" },
  { id: "lehenga1", label: "Embroidered Bridal Lehenga", imageSrc: lehenga1, category: "lehenga",
    description: "Bridal wear in thread embroidery and mirror-work" },
  { id: "kerala_saree1", label: "Kerala Kasavu Saree", imageSrc: kerala_saree1, category: "saree",
    description: "A traditional kasavu weave from Kerala" },
  { id: "kurta1", label: "Silk Kurta Set", imageSrc: kurta1, category: "kurta",
    description: "Festive wear in lustrous, finely embroidered raw silk" },
  { id: "sherwani1", label: "Embroidered Sherwani", imageSrc: sherwani1, category: "sherwani",
    description: "Tone-on-tone embroidery with a softly draped dhoti" },
  { id: "royal_sherwani1", label: "Layered Gold Sherwani", imageSrc: royal_sherwani1, category: "royal_sherwani",
    description: "Wedding attire with zardozi borders and a turban" },
  { id: "kurta_dhoti1", label: "Kurta with Dhoti", imageSrc: kurta_dhoti1, category: "kurta_dhoti",
    description: "Traditional everyday wear with a pleated, bordered dhoti" },
  { id: "kurta_pajama1", label: "Magenta Kurta with Pants", imageSrc: kurta_pajama1, category: "kurta_pajama",
    description: "Easy everyday wear: a long cotton kurta with button loops" },
  { id: "mundu_shirt1", label: "Shirt with Kasavu Mundu", imageSrc: mundu_shirt1, category: "mundu_shirt",
    description: "The classic Kerala mundu, edged in gold kasavu" },
  { id: "male1", label: "Formal Shirt", imageSrc: male1, category: "shirt",
    description: "Crisp woven cotton for sharp office days" },
  { id: "male2", label: "Cream Button-Down Shirt", imageSrc: male2, category: "shirt" },
  { id: "male3", label: "Color-Block Zip Polo", imageSrc: male3, category: "polo_shirt" },
  { id: "blazer1", label: "Textured Grey Blazer", imageSrc: blazer1, category: "blazer",
    description: "Textured tweed with a structured, tailored fit" },
  { id: "blazer2", label: "Charcoal Blazer", imageSrc: blazer2, category: "blazer",
    description: "Fine wool tailoring, finished with a pocket square" },
  { id: "outfit1", label: "Navy Pocket Shirt", imageSrc: outfit1, category: "shirt" },
  { id: "outfit2", label: "Brown Knit Polo", imageSrc: outfit2, category: "polo_shirt" },
];

type PageProps = {
  onBack: () => void;
  onHome: () => void;
  onContinue: () => void;
  onBeauty?: () => void;
};

export default function OutfitSelectionPage({
  onBack,
  onHome,
  onContinue,
  onBeauty,
}: PageProps) {
  const {
    sessionId,
    selectedOutfit,
    setSelectedOutfit,
    selectedSize,
    setSelectedSize,
    recommendedSize,
    setRecommendedSize,
    availableSizes,
    setAvailableSizes,
    sizeFitQuality,
    setSizeFitQuality,
    sizeConfidence,
    setSizeConfidence,
    sizeFitNotes,
    setSizeFitNotes,
    sizeFits,
    setSizeFits,
    bodyProfile,
  } = useTrialRoom();

  const thumbnailRefs = useRef<Record<string, HTMLDivElement | null>>({});

  const activeIndex = selectedOutfit
    ? Math.max(0, OUTFITS.findIndex((o) => o.id === selectedOutfit))
    : 0;

  // Auto-select initial outfit on mount if not already chosen
  useEffect(() => {
    if (!selectedOutfit) {
      setSelectedOutfit(OUTFITS[0].id);
    }
  }, [selectedOutfit, setSelectedOutfit]);

  // Smoothly scroll active thumbnail into view when selected outfit changes
  useEffect(() => {
    if (selectedOutfit && thumbnailRefs.current[selectedOutfit]) {
      thumbnailRefs.current[selectedOutfit]?.scrollIntoView({
        behavior: "smooth",
        block: "nearest",
        inline: "center",
      });
    }
  }, [selectedOutfit]);

  // Recompute size recommendation dynamically on mount and whenever active garment or body profile changes
  useEffect(() => {
    const currentOutfitId = selectedOutfit || OUTFITS[0].id;
    const currentOutfit = OUTFITS.find((o) => o.id === currentOutfitId) || OUTFITS[0];
    const category = currentOutfit?.category || "";

    fetch(`${API_BASE_URL}/api/recommend-size`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        body_profile: bodyProfile,
        garment_category: category,
      }),
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.success && data.size_recommendation) {
          const rec = data.size_recommendation;
          setRecommendedSize(rec.recommended_size);
          setSelectedSize(rec.recommended_size);
          setAvailableSizes(rec.available_sizes || []);
          setSizeConfidence(rec.size_confidence);
          setSizeFitNotes(rec.fit_notes);
          setSizeFits(rec.size_fits || {});
          setSizeFitQuality("perfect");
        }
      })
      .catch((err) => {
        console.warn("[OUTFIT SIZE] Size recommendation fetch error:", err);
      });
  }, [
    selectedOutfit,
    bodyProfile,
    sessionId,
    setRecommendedSize,
    setSelectedSize,
    setAvailableSizes,
    setSizeConfidence,
    setSizeFitNotes,
    setSizeFits,
    setSizeFitQuality,
  ]);

  const handlePrev = () => {
    const newIdx = (activeIndex - 1 + OUTFITS.length) % OUTFITS.length;
    setSelectedOutfit(OUTFITS[newIdx].id);
  };

  const handleNext = () => {
    const newIdx = (activeIndex + 1) % OUTFITS.length;
    setSelectedOutfit(OUTFITS[newIdx].id);
  };

  const handleContinue = () => {
    if (!selectedOutfit) {
      setSelectedOutfit(OUTFITS[activeIndex].id);
    }
    onContinue();
  };

  const currentOutfit = OUTFITS[activeIndex] || OUTFITS[0];

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
      {/* Main Layout Container */}
      <div
        style={{
          position: "absolute",
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: "1.4vh",
          top: 0,
          left: 0,
          zIndex: 5,
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
          }}
        >
          {/* Left Line */}
          <div style={{ boxSizing: "border-box", width: "2.5vh", height: "0px", border: "0.25vh solid #5C4A3D" }} />

          {/* SELECT STYLE text */}
          <span
            style={{
              fontFamily: "'Montserrat', sans-serif",
              fontWeight: 700,
              fontSize: "1.4vh",
              lineHeight: "1.8vh",
              textTransform: "uppercase",
              color: "#5C4A3D",
              textAlign: "center",
              whiteSpace: "nowrap",
            }}
          >
            SELECT STYLE
          </span>

          {/* Right Line */}
          <div style={{ boxSizing: "border-box", width: "2.5vh", height: "0px", border: "0.25vh solid #5C4A3D" }} />
        </div>


        {/* Carousel Row (Large Focal Preview) */}
        <div
          style={{
            display: "flex",
            flexDirection: "row",
            justifyContent: "center",
            alignItems: "center",
            gap: "5vh",
            width: "100%",
          }}
        >
          {/* chevron-left (Brown outline button) */}
          <button
            onClick={handlePrev}
            aria-label="Previous outfit"
            style={{
              width: "5.2vh",
              height: "5.2vh",
              border: "0.18vh solid #5C4A3D",
              borderRadius: "50%",
              background: "rgba(255, 255, 255, 0.25)",
              backdropFilter: "blur(6px)",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#5C4A3D",
              zIndex: 10,
              outline: "none",
              transition: "transform 0.15s ease",
            }}
            onMouseDown={(e) => {
              e.currentTarget.style.transform = "scale(0.92)";
            }}
            onMouseUp={(e) => {
              e.currentTarget.style.transform = "scale(1)";
            }}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" style={{ width: "45%", height: "45%" }}>
              <polyline points="15 18 9 12 15 6"></polyline>
            </svg>
          </button>

          {/* cards-track (Large garment preview) */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              position: "relative",
              height: "32vh",
              width: "28vh",
            }}
          >
            {/* Grounding shadow */}
            <div
              style={{
                position: "absolute",
                bottom: "1.5vh",
                width: "18vh",
                height: "2.0vh",
                borderRadius: "50%",
                background:
                  "radial-gradient(ellipse at center, rgba(0,0,0,0.45) 0%, rgba(0,0,0,0.20) 45%, rgba(0,0,0,0) 75%)",
                filter: "blur(1.0vh)",
                zIndex: 0,
              }}
            />

            {/* Soft ambient glow */}
            <div
              style={{
                position: "absolute",
                width: "24vh",
                height: "28vh",
                borderRadius: "50%",
                background:
                  "radial-gradient(ellipse at center, rgba(255,255,255,0.12) 0%, rgba(255,255,255,0.04) 55%, rgba(255,255,255,0) 75%)",
                filter: "blur(1.2vh)",
                zIndex: 0,
              }}
            />

            {OUTFITS.map((outfit, index) => (
              <img
                key={outfit.id}
                src={outfit.imageSrc}
                alt={outfit.label}
                onClick={() => setSelectedOutfit(outfit.id)}
                style={{
                  position: "absolute",
                  height: "28vh",
                  maxWidth: "26vh",
                  objectFit: "contain",
                  cursor: "pointer",
                  filter:
                    "drop-shadow(0px 0.8vh 1.4vh rgba(0,0,0,0.4)) drop-shadow(0px 0px 1.0vh rgba(0,0,0,0.2))",
                  opacity: index === activeIndex ? 1 : 0,
                  transform: index === activeIndex ? "scale(1)" : "scale(0.94)",
                  transition: "opacity 0.35s ease-in-out, transform 0.35s ease-in-out",
                  pointerEvents: index === activeIndex ? "auto" : "none",
                  zIndex: 1,
                }}
              />
            ))}
          </div>

          {/* chevron-right (Brown outline button) */}
          <button
            onClick={handleNext}
            aria-label="Next outfit"
            style={{
              width: "5.2vh",
              height: "5.2vh",
              border: "0.18vh solid #5C4A3D",
              borderRadius: "50%",
              background: "rgba(255, 255, 255, 0.25)",
              backdropFilter: "blur(6px)",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#5C4A3D",
              zIndex: 10,
              outline: "none",
              transition: "transform 0.15s ease",
            }}
            onMouseDown={(e) => {
              e.currentTarget.style.transform = "scale(0.92)";
            }}
            onMouseUp={(e) => {
              e.currentTarget.style.transform = "scale(1)";
            }}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" style={{ width: "45%", height: "45%" }}>
              <polyline points="9 18 15 12 9 6"></polyline>
            </svg>
          </button>
        </div>

        {/* Active Outfit Name & Category Badge */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.8vh",
            background: "rgba(255, 255, 255, 0.45)",
            border: "0.12vh solid rgba(92, 74, 61, 0.22)",
            borderRadius: "9999px",
            padding: "0.35vh 1.4vh",
            backdropFilter: "blur(8px)",
            marginTop: "-0.5vh",
          }}
        >
          <span
            style={{
              fontFamily: "'Montserrat', sans-serif",
              fontWeight: 700,
              fontSize: "1.2vh",
              color: "#5C4A3D",
              textTransform: "uppercase",
              letterSpacing: "0.06vh",
            }}
          >
            {currentOutfit.label}
          </span>
          <span
            style={{
              display: "inline-block",
              width: "0.35vh",
              height: "0.35vh",
              borderRadius: "50%",
              backgroundColor: "#B79579",
            }}
          />
          <span
            style={{
              fontFamily: "'Poppins', sans-serif",
              fontWeight: 600,
              fontSize: "1.05vh",
              color: "#856A54",
              textTransform: "capitalize",
            }}
          >
            {currentOutfit.category.replace("_", " ")}
          </span>
        </div>

        {/* Outfit description: one short line, same muted style as the fit notes. The line keeps its height when an
            outfit has no description, so the rail below does not jump while swiping. */}
        <span
          style={{
            fontFamily: "'Poppins', sans-serif",
            fontWeight: 400,
            fontSize: "0.85vh",
            color: "rgba(92, 74, 61, 0.65)",
            textAlign: "center",
            maxWidth: "42vh",
            lineHeight: 1.3,
            minHeight: "1.3em",
            marginTop: "-0.4vh",
          }}
        >
          {currentOutfit.description ?? ""}
        </span>

        {/* Visible Outfits Thumbnail Rail (All 12 dresses visible side by side) */}
        <div
          style={{
            width: "48vh",
            maxWidth: "94%",
            position: "relative",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: "0.4vh",
            zIndex: 10,
          }}
        >
          <div
            style={{
              width: "100%",
              display: "flex",
              flexDirection: "row",
              alignItems: "center",
              gap: "0.9vh",
              overflowX: "auto",
              padding: "0.4vh 0.6vh",
              scrollbarWidth: "none",
              msOverflowStyle: "none",
              scrollSnapType: "x mandatory",
              WebkitOverflowScrolling: "touch",
            }}
          >
            {OUTFITS.map((outfit) => {
              const isSel = outfit.id === selectedOutfit;
              return (
                <div
                  key={outfit.id}
                  ref={(el) => {
                    thumbnailRefs.current[outfit.id] = el;
                  }}
                  style={{
                    scrollSnapAlign: "center",
                    flexShrink: 0,
                  }}
                >
                  <OutfitCard
                    imageSrc={outfit.imageSrc}
                    label={outfit.label}
                    category={outfit.category}
                    selected={isSel}
                    onClick={() => setSelectedOutfit(outfit.id)}
                  />
                </div>
              );
            })}
          </div>

          {/* Subtitle helper */}
          <span
            style={{
              fontFamily: "'Poppins', sans-serif",
              fontSize: "0.9vh",
              fontWeight: 500,
              color: "rgba(92, 74, 61, 0.6)",
              letterSpacing: "0.03vh",
            }}
          >
            Swipe or tap to select • {activeIndex + 1} of {OUTFITS.length} outfits
          </span>
        </div>

        {/* Size Selector (6-position pills with recommended badge) */}
        <SizeSelector
          selectedSize={selectedSize}
          recommendedSize={recommendedSize}
          availableSizes={availableSizes}
          sizeFitQuality={sizeFitQuality}
          sizeConfidence={sizeConfidence}
          sizeFitNotes={sizeFitNotes}
          sizeFits={sizeFits}
          onChange={(newSize) => setSelectedSize(newSize)}
          compact
          style={{ margin: "0.2vh 0 0 0", zIndex: 12 }}
        />

        {/* TAP TO TRY IT ON button */}
        <button
          onClick={handleContinue}
          style={{
            position: "relative",
            display: "flex",
            flexDirection: "row",
            justifyContent: "center",
            alignItems: "center",
            padding: "0px",
            gap: "1.45vh",
            width: "28.85vh",
            height: "5.4vh",
            background: "linear-gradient(90deg, #B79579 0%, #90755E 48.08%, #CFB499 100%)",
            border: "0.1vh solid #DBB894",
            boxShadow: "0px 0.25vh 0.25vh rgba(0, 0, 0, 0.25), 0px 2.26vh 4.5vh -1.08vh rgba(0, 0, 0, 0.25)",
            borderRadius: "900vh",
            zIndex: 15,
            cursor: "pointer",
            outline: "none",
            transition: "transform 0.15s ease",
          }}
          onMouseDown={(e) => {
            e.currentTarget.style.transform = "scale(0.97)";
          }}
          onMouseUp={(e) => {
            e.currentTarget.style.transform = "scale(1)";
          }}
        >
          <span
            style={{
              fontFamily: "'Poppins', sans-serif",
              fontWeight: 700,
              fontSize: "1.4vh",
              lineHeight: "1.4vh",
              display: "flex",
              alignItems: "center",
              textAlign: "center",
              letterSpacing: "0.2vh",
              textTransform: "uppercase",
              color: "#FFFFFF",
            }}
          >
            TAP TO TRY IT ON
          </span>

          {/* Sparkle Icon */}
          <div
            style={{
              width: "1.8vh",
              height: "1.8vh",
              position: "relative",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <svg viewBox="0 0 30 30" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ width: "100%", height: "100%" }}>
              <path d="M15 0C15 8.28427 8.28427 15 0 15C8.28427 15 15 21.7157 15 30C15 21.7157 21.7157 15 30 15C21.7157 15 15 8.28427 15 0Z" fill="#FFFFFF" />
            </svg>
          </div>
        </button>

        {/* Home Button */}
        <NavigationButton
          type="home"
          onClick={onHome}
          style={{ width: "6.2vh", height: "6.2vh", position: "relative" }}
        />

        {/* Info Text Block */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: "0.6vh",
            width: "80%",
          }}
        >
          <span
            style={{
              fontFamily: "'Montserrat', sans-serif",
              fontWeight: 700,
              fontSize: "1.4vh",
              color: "#5C4A3D",
              textTransform: "uppercase",
              letterSpacing: "0.08vh",
              textAlign: "center",
            }}
          >
            FIND THE LOOK THAT SUITS YOU BEST
          </span>
          <span
            style={{
              fontFamily: "'Poppins', sans-serif",
              fontWeight: 500,
              fontSize: "1.1vh",
              color: "#5C4A3D",
              textAlign: "center",
              maxWidth: "45vh",
              lineHeight: "1.5vh",
            }}
          >
            Browse our curated collection and choose a style to preview instantly with your personalized virtual try-on.
          </span>
        </div>
      </div>
    </div>
  );
}

import { useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from "react";
import avonLogo from "../assets/avon_logo.png";
import NavigationButton from "../components/NavigationButton";
import QRCard from "../components/QRCard";
import { useTrialRoom } from "../context/TrialRoomContext";
import { API_BASE_URL } from "../utils/api";

type Finish = "matte" | "satin" | "gloss";

type Shade = {
  id: string;
  name: string;
  category: string;
  hex_color: string;
  finish_default: Finish;
};

type PowderFinish = "matte" | "natural_velvet" | "luminous";
type Coverage = "sheer" | "medium";
type Powder = { id: string; name: string; hex_tint: string; tinted: boolean; finish_default: PowderFinish };
// Which product is in the look: lipstick or compact powder
type Mode = "lipstick" | "powder";

type FaceBox = { x: number; y: number; w: number; h: number };

const FINISHES: { id: Finish; label: string }[] = [
  { id: "matte", label: "Matte" },
  { id: "satin", label: "Cream" },
  { id: "gloss", label: "Glossy" },
];


const POWDER_FINISHES: { id: PowderFinish; label: string }[] = [
  { id: "matte", label: "Matte" },
  { id: "natural_velvet", label: "Natural Velvet" },
  { id: "luminous", label: "Luminous" },
];

// Coverage is fixed: the visitor picks shade and finish only
const POWDER_COVERAGE: Coverage = "medium";

const BROWN = "#5C4A3D";
const PREVIEW_DEBOUNCE_MS = 120;
// The preview spans this multiple of the face box, so a full-length capture is shown zoomed in on the face
const FACE_ZOOM = 1.6;
// Slider position, sent as intensity = value / 100; the backend maps it to opacity on a curve (beauty_tuning).
// 1 is the lightest setting (a tinted balm); 0 would mean no lipstick at all, which the Lipstick tab already covers.
const INTENSITY_MIN = 1;
const INTENSITY_DEFAULT = 75;

/** Positions an image inside a square box so the face box is centred and fills 1/FACE_ZOOM of it. */
function zoomedImageStyle(box: FaceBox | null, natural: [number, number] | null): CSSProperties {
  if (!box || !natural) {
    return { position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "contain" };
  }
  const [iw, ih] = natural;
  const side = Math.max(box.w * iw, box.h * ih) * FACE_ZOOM;
  const cx = (box.x + box.w / 2) * iw;
  const cy = (box.y + box.h / 2) * ih;
  return {
    position: "absolute",
    width: `${(iw / side) * 100}%`,
    height: `${(ih / side) * 100}%`,
    left: `${50 - (cx / side) * 100}%`,
    top: `${50 - (cy / side) * 100}%`,
    maxWidth: "none",
  };
}

const sectionLabel: CSSProperties = {
  fontFamily: "'Montserrat', sans-serif",
  fontWeight: 700,
  fontSize: "0.95vh",
  letterSpacing: "0.08vh",
  textTransform: "uppercase",
  color: "rgba(92, 74, 61, 0.75)",
};

const pillButton = (primary: boolean): CSSProperties => ({
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  gap: "0.8vh",
  height: "4.6vh",
  padding: "0 2.2vh",
  borderRadius: "900vh",
  border: primary ? "0.1vh solid #DBB894" : `0.15vh solid ${BROWN}`,
  background: primary ? "linear-gradient(90deg, #B79579 0%, #90755E 48.08%, #CFB499 100%)" : "rgba(255, 255, 255, 0.35)",
  boxShadow: primary ? "0px 0.25vh 0.25vh rgba(0, 0, 0, 0.25)" : "none",
  color: primary ? "#FFFFFF" : BROWN,
  fontFamily: "'Poppins', sans-serif",
  fontWeight: 700,
  fontSize: "1.15vh",
  letterSpacing: "0.12vh",
  textTransform: "uppercase",
  cursor: "pointer",
  outline: "none",
});

function Chip({ selected, onClick, children }: { selected: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "0.6vh 1.5vh",
        borderRadius: "9999px",
        border: `0.12vh solid ${selected ? "#90755E" : "rgba(92, 74, 61, 0.3)"}`,
        background: selected ? "#90755E" : "rgba(255, 255, 255, 0.45)",
        color: selected ? "#FFFFFF" : BROWN,
        fontFamily: "'Poppins', sans-serif",
        fontWeight: 600,
        fontSize: "1.05vh",
        cursor: "pointer",
        outline: "none",
      }}
    >
      {children}
    </button>
  );
}

type PageProps = {
  onClothing: () => void;
  onHome: () => void;
};

export default function BeautyStudioPage({ onClothing, onHome }: PageProps) {
  const { sessionId, capturedImage } = useTrialRoom();

  const [shades, setShades] = useState<Shade[]>([]);
  const [shade, setShade] = useState<Shade | null>(null);
  const [finish, setFinish] = useState<Finish>("satin");
  const [intensity, setIntensity] = useState(INTENSITY_DEFAULT);
  // A close-up taken here is used only for lipstick; the clothing capture (capturedImage) is never replaced
  const [closeUp, setCloseUp] = useState<string | null>(null);
  const portrait = closeUp ?? capturedImage;
  const [cameraOn, setCameraOn] = useState(!capturedImage);
  const [natural, setNatural] = useState<[number, number] | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [faceBox, setFaceBox] = useState<FaceBox | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [split, setSplit] = useState(50);
  const [saving, setSaving] = useState(false);
  const [qr, setQr] = useState<string | null>(null);
  const [savedImage, setSavedImage] = useState<string | null>(null);
  const [powders, setPowders] = useState<Powder[]>([]);
  const [powder, setPowder] = useState<Powder | null>(null);
  const [powderFinish, setPowderFinish] = useState<PowderFinish>("natural_velvet");
  const [mode, setMode] = useState<Mode>("lipstick");
  const wantsLipstick = mode === "lipstick";
  const wantsPowder = mode === "powder";

  const requestSeq = useRef(0);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/beauty/powders`)
      .then((res) => res.json())
      .then((data: Powder[]) => {
        setPowders(data);
        if (data.length) {
          setPowder(data[0]);
        }
      })
      .catch((err) => console.warn("[BEAUTY] Powder catalog fetch error:", err));
    fetch(`${API_BASE_URL}/api/beauty/shades`)
      .then((res) => res.json())
      .then((data: Shade[]) => {
        setShades(data);
        if (data.length) {
          setShade(data[0]);
        }
      })
      .catch((err) => {
        console.warn("[BEAUTY] Shade catalog fetch error:", err);
        setError("The shade catalog could not be loaded. Please try again.");
      });
  }, []);

  const families = useMemo(() => {
    const groups = new Map<string, Shade[]>();
    shades.forEach((s) => groups.set(s.category, [...(groups.get(s.category) ?? []), s]));
    return [...groups.entries()];
  }, [shades]);

  const lookReady = (!wantsLipstick || !!shade) && (!wantsPowder || !!powder);

  const requestBody = (includeQr: boolean) =>
    JSON.stringify({
      image: portrait,
      session_id: sessionId,
      lipstick:
        wantsLipstick && shade
          ? { shade_id: shade.id, shade_name: shade.name, hex_color: shade.hex_color, finish, intensity: intensity / 100 }
          : null,
      compact_powder:
        wantsPowder && powder
          ? { shade_id: powder.id, shade_name: powder.name, hex_tint: powder.hex_tint, finish: powderFinish, coverage: POWDER_COVERAGE }
          : null,
      include_qr: includeQr,
    });

  // Live preview: debounced so dragging the slider sends one request per pause, and stale replies are dropped
  useEffect(() => {
    if (!portrait || !lookReady || cameraOn) return;
    const seq = ++requestSeq.current;
    const timer = setTimeout(() => {
      setLoading(true);
      fetch(`${API_BASE_URL}/api/beauty/tryon`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: requestBody(false),
      })
        .then((res) => res.json())
        .then((data) => {
          if (seq !== requestSeq.current) return;
          if (data.success) {
            setResult(data.result_image);
            setFaceBox(data.face_box ?? null);
            setError(null);
          } else {
            setResult(null);
            setError(data.message || "The preview could not be created.");
          }
        })
        .catch((err) => {
          console.warn("[BEAUTY] Try-on error:", err);
          if (seq === requestSeq.current) setError("Could not reach the beauty studio. Please try again.");
        })
        .finally(() => {
          if (seq === requestSeq.current) setLoading(false);
        });
    }, PREVIEW_DEBOUNCE_MS);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [portrait, mode, shade, finish, intensity, powder, powderFinish, sessionId, cameraOn]);

  // Close-up camera (uses the camera chosen at setup)
  useEffect(() => {
    if (!cameraOn) return;
    let alive = true;
    let stream: MediaStream | null = null;
    const cameraId = localStorage.getItem("selectedCameraId");
    const size = { width: { ideal: 1920 }, height: { ideal: 1080 } };
    navigator.mediaDevices
      .getUserMedia({ video: cameraId ? { deviceId: { exact: cameraId }, ...size } : { facingMode: "user", ...size } })
      .then((s) => {
        if (!alive) {
          s.getTracks().forEach((t) => t.stop());
          return;
        }
        stream = s;
        if (videoRef.current) videoRef.current.srcObject = s;
      })
      .catch((err) => {
        console.error("[BEAUTY] Camera error:", err);
        if (alive) setError("The camera is not available.");
      });
    return () => {
      alive = false;
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, [cameraOn]);

  const takeCloseUp = () => {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0);
    setCloseUp(canvas.toDataURL("image/jpeg", 0.92));
    setResult(null);
    setFaceBox(null);
    setNatural(null);
    setError(null);
    setCameraOn(false);
  };

  const useCapturePhoto = () => {
    setCloseUp(null);
    setResult(null);
    setFaceBox(null);
    setNatural(null);
    setError(null);
    setCameraOn(false);
  };

  const selectShade = (s: Shade) => {
    setShade(s);
  };

  const saveLook = () => {
    if (!portrait || !lookReady || saving) return;
    setSaving(true);
    fetch(`${API_BASE_URL}/api/beauty/tryon`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: requestBody(true),
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.success && data.qr_code_base64) {
          // The saved look may be the studio portrait (its own framing), so it is shown with the QR, not in the preview
          setSavedImage(data.result_image);
          setQr(data.qr_code_base64);
        } else {
          setError(data.message || "Your look could not be saved.");
        }
      })
      .catch(() => setError("Could not reach the beauty studio. Please try again."))
      .finally(() => setSaving(false));
  };

  // A close-up is shown framed exactly as the camera showed it; only the full-length capture is zoomed to the face
  const imgStyle: CSSProperties = closeUp
    ? { position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }
    : zoomedImageStyle(faceBox, natural);
  const lookLabel = [
    wantsLipstick && shade && `${shade.name} ${FINISHES.find((f) => f.id === finish)?.label}`,
    wantsPowder && powder && `${powder.name} ${POWDER_FINISHES.find((f) => f.id === powderFinish)?.label} powder`,
  ]
    .filter(Boolean)
    .join(" · ");
  const selectPowder = (p: Powder) => {
    setPowder(p);
  };

  return (
    <div style={{ position: "absolute", inset: 0, overflow: "hidden" }}>
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: "1.3vh",
          zIndex: 5,
        }}
      >

        {/* Brand mark: small, sized to the header text */}
        <img src={avonLogo} alt="AVON" style={{ width: "5.6vh", height: "5.6vh", borderRadius: "0.6vh", display: "block" }} />

        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", gap: "1.0vh" }}>
          <div style={{ width: "2.5vh", border: `0.25vh solid ${BROWN}` }} />
          <span
            style={{
              fontFamily: "'Montserrat', sans-serif",
              fontWeight: 700,
              fontSize: "1.4vh",
              textTransform: "uppercase",
              color: BROWN,
              whiteSpace: "nowrap",
            }}
          >
            SELECT STYLE
          </span>
          <div style={{ width: "2.5vh", border: `0.25vh solid ${BROWN}` }} />
        </div>


        {/* Preview with the lipstick shade catalog beside it (at hand height on the kiosk, not below everything) */}
        <div style={{ position: "relative", marginBottom: "3.5vh" }}>
        {/* Preview: live camera for a close-up, or before/after split on the face */}
        <div
          style={{
            position: "relative",
            // Narrower than the clothing window, same height; capped by screen width so the shades beside it always fit
            width: "min(34vh, calc(100vw - 12vh))",
            height: "47vh",
            borderRadius: "2vh",
            overflow: "hidden",
            background: "rgba(255, 255, 255, 0.3)",
            border: "0.15vh solid rgba(92, 74, 61, 0.25)",
            boxShadow: "0px 1vh 2.5vh rgba(0, 0, 0, 0.18)",
          }}
        >
          {cameraOn ? (
            <>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                style={{ width: "100%", height: "100%", objectFit: "cover", transform: "scaleX(-1)" }}
              />
            </>
          ) : portrait ? (
            <>
              <img
                src={portrait}
                alt="Before"
                onLoad={(e) => setNatural([e.currentTarget.naturalWidth, e.currentTarget.naturalHeight])}
                style={imgStyle}
              />
              {result && (
                <>
                  <img
                    src={result}
                    alt="With makeup"
                    style={{ ...imgStyle, clipPath: `inset(0 0 0 ${split}%)` }}
                  />
                  <div
                    style={{
                      position: "absolute",
                      top: 0,
                      bottom: 0,
                      left: `${split}%`,
                      width: "0.3vh",
                      marginLeft: "-0.15vh",
                      background: "#FFFFFF",
                      boxShadow: "0 0 0.8vh rgba(0,0,0,0.4)",
                      pointerEvents: "none",
                    }}
                  />
                  <span style={{ ...sectionLabel, position: "absolute", left: "1vh", bottom: "1vh", color: "#FFFFFF", textShadow: "0 0 0.6vh rgba(0,0,0,0.6)" }}>
                    Before
                  </span>
                  <span style={{ ...sectionLabel, position: "absolute", right: "1vh", bottom: "1vh", color: "#FFFFFF", textShadow: "0 0 0.6vh rgba(0,0,0,0.6)" }}>
                    After
                  </span>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={split}
                    onChange={(e) => setSplit(Number(e.target.value))}
                    aria-label="Compare before and after"
                    // touchAction none: on the kiosk touchscreen a finger drag moves the split instead of panning/zooming the page
                    style={{ position: "absolute", inset: 0, width: "100%", height: "100%", margin: 0, opacity: 0, cursor: "ew-resize", touchAction: "none" }}
                  />
                </>
              )}
              {loading && !result && (
                <div
                  style={{
                    position: "absolute",
                    inset: 0,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    background: "rgba(255,255,255,0.35)",
                    fontFamily: "'Poppins', sans-serif",
                    fontWeight: 600,
                    fontSize: "1.2vh",
                    color: BROWN,
                  }}
                >
                  Applying your look…
                </div>
              )}
            </>
          ) : null}
        </div>

          {wantsLipstick && (
            <div
              style={{
                // Out of flow so the preview stays centred on screen; one fixed column (all shades fit the preview height)
                position: "absolute",
                left: "calc(100% + 1vh)",
                top: "50%",
                transform: "translateY(-50%)",
                maxHeight: "100%",
                width: "3.2vh",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: "0.9vh",
                padding: "0.6vh",
              }}
            >
              {families.map(([family, list]) => (
                <div key={family} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "0.5vh" }}>
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "0.9vh" }}>
                    {list.map((s) => {
                      const isSel = s.id === shade?.id;
                      return (
                        <button
                          key={s.id}
                          onClick={() => selectShade(s)}
                          aria-label={s.name}
                          title={s.name}
                          style={{
                            width: "3.2vh",
                            height: "3.2vh",
                            borderRadius: "50%",
                            background: s.hex_color,
                            border: isSel ? "0.3vh solid #FFFFFF" : "0.15vh solid rgba(255,255,255,0.6)",
                            boxShadow: isSel ? `0 0 0 0.25vh ${BROWN}, 0 0.4vh 0.8vh rgba(0,0,0,0.3)` : "0 0.3vh 0.6vh rgba(0,0,0,0.25)",
                            cursor: "pointer",
                            outline: "none",
                            transition: "box-shadow 0.15s ease",
                          }}
                        />
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Home button: centred on the preview's bottom border, as on the clothing capture page */}
          <NavigationButton
            type="home"
            onClick={onHome}
            style={{
              position: "absolute",
              bottom: "-3.52vh",
              // Centred with left alone: the button sets its own transform (press scale), which would drop a translateX
              left: "calc(50% - 3.525vh)",
              width: "7.05vh",
              height: "7.05vh",
              zIndex: 10,
            }}
          />

          {/* Compact powder shades: right of the preview */}
          {wantsPowder && powders.length > 0 && (
            <div
              style={{
                position: "absolute",
                left: "calc(100% + 1.6vh)",
                top: "50%",
                transform: "translateY(-50%)",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: "0.5vh",
                padding: "0.6vh",
              }}
            >
              <span style={sectionLabel}>Powder</span>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 3.2vh)", gap: "0.6vh" }}>
                {powders.map((p) => {
                  const isSel = p.id === powder?.id;
                  return (
                    <button
                      key={p.id}
                      onClick={() => selectPowder(p)}
                      aria-label={`${p.name} powder`}
                      title={p.name}
                      style={{
                        width: "3.2vh",
                        height: "3.2vh",
                        borderRadius: "50%",
                        // Translucent: a frosted swatch, since it adds no colour
                        background: p.tinted
                          ? p.hex_tint
                          : "radial-gradient(circle at 35% 30%, #FFFFFF 0%, #F1ECE6 55%, #DCD4CB 100%)",
                        border: isSel ? "0.3vh solid #FFFFFF" : "0.15vh solid rgba(255,255,255,0.7)",
                        boxShadow: isSel ? `0 0 0 0.25vh ${BROWN}, 0 0.3vh 0.6vh rgba(0,0,0,0.3)` : "0 0.25vh 0.5vh rgba(0,0,0,0.22)",
                        cursor: "pointer",
                        outline: "none",
                      }}
                    />
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Shade name / status line */}
        <span
          style={{
            fontFamily: "'Poppins', sans-serif",
            fontWeight: error ? 500 : 600,
            fontSize: "1.15vh",
            color: error ? "#9E3A2A" : BROWN,
            textAlign: "center",
            maxWidth: "42vh",
            minHeight: "1.5vh",
          }}
        >
          {error ?? (saving ? "Saving your look…" : lookLabel)}
        </span>


        {wantsLipstick && (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "1.0vh" }}>
            {/* Lipstick finish */}
            <div style={{ display: "flex", gap: "0.8vh" }}>
              {FINISHES.map((f) => (
                <Chip key={f.id} selected={f.id === finish} onClick={() => setFinish(f.id)}>
                  {f.label}
                </Chip>
              ))}
            </div>

            {/* Lipstick intensity */}
            <div style={{ display: "flex", alignItems: "center", gap: "1.2vh", width: "36vh" }}>
              <span style={sectionLabel}>Sheer</span>
              <input
                type="range"
                min={INTENSITY_MIN}
                max={100}
                value={intensity}
                onChange={(e) => setIntensity(Number(e.target.value))}
                aria-label="Lipstick intensity"
                style={{ flex: 1, accentColor: "#90755E", cursor: "pointer" }}
              />
              <span style={sectionLabel}>Full</span>
            </div>
          </div>
        )}

        {wantsPowder && powders.length > 0 && (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "0.8vh" }}>
            <span style={sectionLabel}>Compact powder</span>
            <div style={{ display: "flex", gap: "0.8vh" }}>
              {POWDER_FINISHES.map((f) => (
                <Chip key={f.id} selected={f.id === powderFinish} onClick={() => setPowderFinish(f.id)}>
                  {f.label}
                </Chip>
              ))}
            </div>
          </div>
        )}

        {/* Actions */}
        <div style={{ display: "flex", gap: "1.4vh" }}>
          {cameraOn ? (
            <>
              <button onClick={takeCloseUp} style={pillButton(true)}>
                Capture close-up
              </button>
              {capturedImage && (
                <button onClick={useCapturePhoto} style={pillButton(false)}>
                  Cancel
                </button>
              )}
            </>
          ) : (
            <>
              <button onClick={() => setCameraOn(true)} style={pillButton(false)}>
                {closeUp ? "Retake close-up" : "Take close-up"}
              </button>
              {closeUp && capturedImage && (
                <button onClick={useCapturePhoto} style={pillButton(false)}>
                  Use my photo
                </button>
              )}
              <button
                onClick={saveLook}
                disabled={!result || saving}
                style={{ ...pillButton(true), opacity: !result || saving ? 0.6 : 1 }}
              >
                {saving ? "Saving…" : "Save look"}
              </button>
            </>
          )}
        </div>

      </div>

      {/* Save look: QR for mobile download */}
      {qr && (
        <div
          onClick={() => setQr(null)}
          style={{
            position: "absolute",
            inset: 0,
            zIndex: 30,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "rgba(40, 30, 22, 0.45)",
            backdropFilter: "blur(6px)",
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: "1.5vh",
              padding: "3vh 4vh",
              borderRadius: "2.5vh",
              background: "rgba(255, 250, 245, 0.95)",
              boxShadow: "0 2vh 4vh rgba(0,0,0,0.3)",
            }}
          >
            <span style={{ fontFamily: "'Poppins', sans-serif", fontWeight: 600, fontSize: "2.0vh", color: BROWN, textTransform: "uppercase", letterSpacing: "0.12vh" }}>
              Scan to download
            </span>
            <span style={{ fontFamily: "'Poppins', sans-serif", fontSize: "1.2vh", color: BROWN }}>{lookLabel}</span>
            {savedImage && (
              <img
                src={savedImage}
                alt="Your saved look"
                style={{ height: "30vh", maxWidth: "36vh", objectFit: "contain", borderRadius: "1.2vh" }}
              />
            )}
            <div style={{ border: `0.8vh solid ${BROWN}`, borderRadius: "1.3vh", overflow: "hidden" }}>
              <QRCard qrCodeSrc={qr} />
            </div>
            <button onClick={() => setQr(null)} style={pillButton(false)}>
              Done
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

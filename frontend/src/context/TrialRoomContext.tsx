/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useState, type ReactNode, useEffect, useRef } from "react";
import { API_BASE_URL } from "../utils/api";

export type FitState = "fitted" | "standard" | "relaxed";

export type BodyProfile = {
  height_ratio: number;
  shoulder_width_ratio: number;
  torso_aspect_ratio: number;
  shoulder_to_hip_ratio: number;
  chest_proportion: number;
  waist_proportion: number;
  hip_proportion: number;
  body_type: string;
  confidence_score: number;
  frame_count_sampled: number;
  keypoints_summary?: Record<string, unknown>;
};

// Fit of one size on this visitor, from the backend (smaller sizes are checked against the visitor's body)
export type SizeFit = {
  fit_quality: string;
  fit_label: string;
  size_offset: number;
};

// "Won't fit" details returned with a won't-fit size: size steps each dimension exceeds the size, and where it strains
export type FitCheck = {
  body_check?: Record<string, number>;
  limiting_dimension?: string | null;
  reason?: string | null;
  strain_zones?: string[];
  render_rejected?: string;
};

export type WontFitInfo = {
  message: string;
  illustrationOnly: boolean;
  fitCheck: FitCheck | null;
};

export type MeasurementField = "height" | "shoulder_width" | "chest_width" | "waist_width";

export type BodyMeasurement = {
  value_cm: number;
  confidence: "measured" | "estimated";
  source: string;
};

// Informational cm estimates (display only) — never sent to size recommendation or try-on
export type BodyMeasurements = {
  height: BodyMeasurement | null;
  shoulder_width: BodyMeasurement | null;
  chest_width: BodyMeasurement | null;
  waist_width: BodyMeasurement | null;
  assumed_distance_m: number;
};

type TrialRoomState = {
  sessionId: string;
  capturedImage: string | null;
  selectedOutfit: string | null;
  generatedImage: string | null;
  lastGoodRender: string | null;
  qrCodeBase64: string | null;
  fitState: FitState;
  bodyProfile: BodyProfile | null;
  bodyMeasurements: BodyMeasurements | null;
  // Visitor correction multiplier applied to every displayed measurement (1 = system estimate)
  measurementScale: number;
  selectedSize: string | null;
  recommendedSize: string | null;
  availableSizes: string[];
  sizeFitQuality: string | null;
  sizeConfidence: number | null;
  sizeFitNotes: string | null;
  sizeRendersCache: Record<string, string>;
  sizeFits: Record<string, SizeFit>;
  // Keyed like sizeRendersCache (`${outfit}_${size}`): the won't-fit state of sizes already requested
  wontFitCache: Record<string, WontFitInfo>;
  setCapturedImage: (image: string | null) => void;
  setSelectedOutfit: (outfitId: string | null) => void;
  setGeneratedImage: (image: string | null) => void;
  setLastGoodRender: (image: string | null) => void;
  setQrCodeBase64: (qr: string | null) => void;
  setFitState: (fit: FitState) => void;
  setBodyProfile: (profile: BodyProfile | null) => void;
  setBodyMeasurements: (measurements: BodyMeasurements | null) => void;
  setMeasurementScale: (scale: number) => void;
  setSelectedSize: (size: string | null) => void;
  setRecommendedSize: (size: string | null) => void;
  setAvailableSizes: (sizes: string[]) => void;
  setSizeFitQuality: (quality: string | null) => void;
  setSizeConfidence: (confidence: number | null) => void;
  setSizeFitNotes: (notes: string | null) => void;
  setSizeRenderInCache: (size: string, imageB64: string, outfitId?: string | null) => void;
  setSizeFits: (fits: Record<string, SizeFit>) => void;
  setWontFitInCache: (size: string, info: WontFitInfo, outfitId?: string | null) => void;
  resetSession: () => void;
};

const generateSessionId = (): string => {
  return "sess_" + Math.random().toString(36).substring(2, 10) + "_" + Date.now();
};

const TrialRoomContext = createContext<TrialRoomState | undefined>(undefined);

export function TrialRoomProvider({ children }: { children: ReactNode }) {
  const [sessionId, setSessionId] = useState<string>(generateSessionId);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [selectedOutfit, _setSelectedOutfit] = useState<string | null>(null);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [lastGoodRender, setLastGoodRender] = useState<string | null>(null);
  const [qrCodeBase64, setQrCodeBase64] = useState<string | null>(null);
  const [fitState, setFitState] = useState<FitState>("standard");
  const [bodyProfile, setBodyProfile] = useState<BodyProfile | null>(null);
  const [bodyMeasurements, setBodyMeasurements] = useState<BodyMeasurements | null>(null);
  const [measurementScale, setMeasurementScale] = useState(1);
  const [selectedSize, setSelectedSize] = useState<string | null>(null);
  const [recommendedSize, setRecommendedSize] = useState<string | null>(null);
  const [availableSizes, setAvailableSizes] = useState<string[]>([]);
  const [sizeFitQuality, setSizeFitQuality] = useState<string | null>(null);
  const [sizeConfidence, setSizeConfidence] = useState<number | null>(null);
  const [sizeFitNotes, setSizeFitNotes] = useState<string | null>(null);
  const [sizeRendersCache, setSizeRendersCache] = useState<Record<string, string>>({});
  const [sizeFits, setSizeFits] = useState<Record<string, SizeFit>>({});
  const [wontFitCache, setWontFitCache] = useState<Record<string, WontFitInfo>>({});

  const setSelectedOutfit = (outfitId: string | null) => {
    _setSelectedOutfit((prevOutfit) => {
      if (outfitId && prevOutfit && outfitId !== prevOutfit) {
        // Clear garment/render-specific data only;
        // bodyProfile, calibration, capturedImage, sessionId are strictly preserved.
        setLastGoodRender(null);
        setGeneratedImage(null);
      }
      return outfitId;
    });
  };

  const setSizeRenderInCache = (size: string, imageB64: string, outfitId?: string | null) => {
    if (!size) return;
    const outfitKey = outfitId || selectedOutfit || "default";
    const sKey = `${outfitKey}_${size.toUpperCase().trim()}`;
    setSizeRendersCache((prev) => ({
      ...prev,
      [sKey]: imageB64,
    }));
  };

  const setWontFitInCache = (size: string, info: WontFitInfo, outfitId?: string | null) => {
    if (!size) return;
    const sKey = `${outfitId || selectedOutfit || "default"}_${size.toUpperCase().trim()}`;
    setWontFitCache((prev) => ({ ...prev, [sKey]: info }));
  };

  // Closing or reloading the app ends the visit: the server must not keep this visitor's images until they expire
  const sessionIdRef = useRef(sessionId);
  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);
  useEffect(() => {
    const purgeOnLeave = () => {
      fetch(`${API_BASE_URL}/api/session/purge`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionIdRef.current }),
        keepalive: true,
      }).catch(() => undefined);
    };
    window.addEventListener("pagehide", purgeOnLeave);
    return () => window.removeEventListener("pagehide", purgeOnLeave);
  }, []);

  const resetSession = () => {
    // Purge server-side session biometrics & destroy ephemeral Fernet key
    if (sessionId) {
      fetch(`${API_BASE_URL}/api/session/purge`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      }).catch((err) => {
        console.warn("[SESSION PURGE WARNING]", err);
      });
    }

    // Reset frontend state & generate fresh session ID for next visitor
    setCapturedImage(null);
    setSelectedOutfit(null);
    setGeneratedImage(null);
    setLastGoodRender(null);
    setQrCodeBase64(null);
    setFitState("standard");
    setBodyProfile(null);
    setBodyMeasurements(null);
    setMeasurementScale(1);
    setSelectedSize(null);
    setRecommendedSize(null);
    setAvailableSizes([]);
    setSizeFitQuality(null);
    setSizeConfidence(null);
    setSizeFitNotes(null);
    setSizeRendersCache({});
    setSizeFits({});
    setWontFitCache({});
    setSessionId(generateSessionId());
  };

  return (
    <TrialRoomContext.Provider
      value={{
        sessionId,
        capturedImage,
        selectedOutfit,
        generatedImage,
        lastGoodRender,
        qrCodeBase64,
        fitState,
        bodyProfile,
        bodyMeasurements,
        measurementScale,
        selectedSize,
        recommendedSize,
        availableSizes,
        sizeFitQuality,
        sizeConfidence,
        sizeFitNotes,
        sizeRendersCache,
        sizeFits,
        wontFitCache,
        setCapturedImage,
        setSelectedOutfit,
        setGeneratedImage,
        setLastGoodRender,
        setQrCodeBase64,
        setFitState,
        setBodyProfile,
        setBodyMeasurements,
        setMeasurementScale,
        setSelectedSize,
        setRecommendedSize,
        setAvailableSizes,
        setSizeFitQuality,
        setSizeConfidence,
        setSizeFitNotes,
        setSizeRenderInCache,
        setSizeFits,
        setWontFitInCache,
        resetSession,
      }}
    >
      {children}
    </TrialRoomContext.Provider>
  );
}

export function useTrialRoom() {
  const context = useContext(TrialRoomContext);
  if (context === undefined) {
    throw new Error("useTrialRoom must be used within a TrialRoomProvider");
  }
  return context;
}

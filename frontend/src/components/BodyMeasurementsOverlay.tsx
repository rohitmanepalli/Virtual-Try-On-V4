import { useState, type CSSProperties } from "react";
import { useTrialRoom } from "../context/TrialRoomContext";
import { API_BASE_URL } from "../utils/api";

// Informational only: these values never feed the size recommendation or try-on.
// The result image shows height and shoulder; chest/waist are still computed but not displayed.
export type CorrectableField = "height" | "shoulder_width";

const FIELDS: Record<CorrectableField, { label: string; min: number; max: number }> = {
  height: { label: "Height", min: 100, max: 230 },
  shoulder_width: { label: "Shoulder", min: 25, max: 70 },
};
const FIELD_ORDER: CorrectableField[] = ["height", "shoulder_width"];
const STEPS = [-5, -1, 1, 5];

const chipStyle: CSSProperties = {
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  minWidth: "7.5vh",
  padding: "0.35vh 0.9vh",
  background: "rgba(255, 255, 255, 0.45)",
  border: "0.12vh solid rgba(183, 149, 121, 0.45)",
  borderRadius: "1.2vh",
  fontFamily: "'Montserrat', sans-serif",
  color: "#5C4A3D",
};

const stepButtonStyle: CSSProperties = {
  minWidth: "3.6vh",
  height: "3.4vh",
  border: "0.12vh solid rgba(183, 149, 121, 0.6)",
  borderRadius: "999vh",
  background: "rgba(255, 255, 255, 0.6)",
  fontFamily: "'Montserrat', sans-serif",
  fontWeight: 700,
  fontSize: "1.15vh",
  color: "#5C4A3D",
  cursor: "pointer",
};

function useBodyMeasurements() {
  const { sessionId, bodyMeasurements, measurementScale, setMeasurementScale } = useTrialRoom();

  const value = (field: CorrectableField): number | null => {
    const m = bodyMeasurements?.[field];
    return m ? Math.round(m.value_cm * measurementScale) : null;
  };
  const prefix = (field: CorrectableField) => (bodyMeasurements?.[field]?.confidence === "measured" ? "" : "~");

  const correct = (field: CorrectableField, correctedCm: number) => {
    const estimate = bodyMeasurements?.[field];
    const before = value(field);
    if (!estimate || before === null || correctedCm === before) return;
    // Image proportions don't depend on distance, so one corrected value rescales every measurement
    setMeasurementScale(correctedCm / estimate.value_cm);
    fetch(`${API_BASE_URL}/api/measurement-correction`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        field,
        displayed_cm: before,
        corrected_cm: correctedCm,
      }),
    }).catch((err) => console.warn("[MEASUREMENT CORRECTION LOG FAILED]", err));
  };

  return { value, prefix, correct };
}

/** Plain black text in the top-right corner of the result image (no box). Tap a line to correct it. */
export function BodyMeasurementsOverlay({
  onCorrect,
  style,
}: {
  onCorrect: (field: CorrectableField) => void;
  style?: CSSProperties;
}) {
  const { value, prefix } = useBodyMeasurements();
  const shown = FIELD_ORDER.filter((f) => value(f) !== null);
  if (shown.length === 0) return null;

  return (
    <div
      style={{
        position: "absolute",
        top: "2.6vh",
        right: "1.0vh",
        zIndex: 5,
        display: "flex",
        flexDirection: "column",
        alignItems: "flex-end",
        ...style,
      }}
    >
      {shown.map((f) => (
        <span
          key={f}
          role="button"
          aria-label={`Correct ${FIELDS[f].label.toLowerCase()}`}
          onClick={() => onCorrect(f)}
          style={{
            // Padding only enlarges the touch target; nothing is drawn behind the text
            padding: "0.4vh 0.5vh",
            fontFamily: "'Montserrat', sans-serif",
            fontSize: "1.0vh",
            fontWeight: 600,
            lineHeight: 1.2,
            color: "#000000",
            whiteSpace: "nowrap",
            cursor: "pointer",
            userSelect: "none",
          }}
        >
          {FIELDS[f].label} {prefix(f)}
          {value(f)} cm ✎
        </span>
      ))}
    </div>
  );
}

/** The −5/−1/+1/+5 correction steppers for one field; closes after save or cancel. */
export function MeasurementCorrection({
  field,
  onClose,
  style,
}: {
  field: CorrectableField;
  onClose: () => void;
  style?: CSSProperties;
}) {
  const { value, correct } = useBodyMeasurements();
  const { label, min, max } = FIELDS[field];
  const [draft, setDraft] = useState(() => value(field) ?? min);

  const save = () => {
    correct(field, draft);
    onClose();
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: "0.5vh",
        userSelect: "none",
        ...style,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: "0.6vh" }}>
        <span style={{ ...chipStyle, flexDirection: "row", gap: "0.5vh", fontSize: "1.1vh", fontWeight: 600 }}>
          {label}
        </span>
        {STEPS.slice(0, 2).map((step) => (
          <button key={step} style={stepButtonStyle} onClick={() => setDraft(Math.max(min, draft + step))}>
            {step}
          </button>
        ))}
        <span
          style={{
            minWidth: "7vh",
            textAlign: "center",
            fontFamily: "'Montserrat', sans-serif",
            fontWeight: 800,
            fontSize: "1.6vh",
            color: "#5C4A3D",
          }}
        >
          {draft} cm
        </span>
        {STEPS.slice(2).map((step) => (
          <button key={step} style={stepButtonStyle} onClick={() => setDraft(Math.min(max, draft + step))}>
            +{step}
          </button>
        ))}
        <button
          style={{ ...stepButtonStyle, background: "#15803D", color: "#FFFFFF", border: "none" }}
          onClick={save}
          aria-label="Save correction"
        >
          ✓
        </button>
        <button style={stepButtonStyle} onClick={onClose} aria-label="Cancel">
          ✕
        </button>
      </div>
      <span style={{ fontFamily: "'Poppins', sans-serif", fontSize: "0.85vh", color: "rgba(92, 74, 61, 0.6)" }}>
        Adjusting one value rescales the others
      </span>
    </div>
  );
}

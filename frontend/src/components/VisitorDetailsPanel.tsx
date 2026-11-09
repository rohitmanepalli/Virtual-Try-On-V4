import { useState, type CSSProperties } from "react";
import { useTrialRoom } from "../context/TrialRoomContext";
import { API_BASE_URL } from "../utils/api";

// Bump whenever the consent wording below changes: each saved visitor records the version they agreed to
const CONSENT_VERSION = "2026-11-05-draft";

const FONT = "'Poppins', sans-serif";
const BROWN = "#5C4A3D";

const fieldStyle: CSSProperties = {
  boxSizing: "border-box",
  width: "100%",
  padding: "1vh 1.2vh",
  fontFamily: FONT,
  fontSize: "1.7vh",
  color: BROWN,
  background: "#FFFFFF",
  border: `0.15vh solid ${BROWN}`,
  borderRadius: "0.8vh",
};

const buttonStyle: CSSProperties = {
  flex: 1,
  padding: "1vh",
  fontFamily: FONT,
  fontSize: "1.6vh",
  fontWeight: 600,
  borderRadius: "999vh",
  border: `0.15vh solid ${BROWN}`,
  cursor: "pointer",
};

/** Optional and skippable: nothing is sent unless the visitor ticks consent and taps Save. Values never reach the logs. */
export default function VisitorDetailsPanel() {
  const { sessionId } = useTrialRoom();
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [phone, setPhone] = useState("");
  const [gender, setGender] = useState("");
  const [consent, setConsent] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canSave = consent && !saving && Boolean(name.trim() || phone.trim() || gender);

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/visitors`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          consent,
          consent_version: CONSENT_VERSION,
          name: name.trim() || null,
          phone: phone.trim() || null,
          gender: gender || null,
        }),
      });
      if (response.ok) {
        setSaved(true);
        setOpen(false);
        setName("");
        setPhone("");
        return;
      }
      const data = await response.json().catch(() => ({}));
      setError(typeof data.detail === "string" ? data.detail : "Please check your details and try again.");
    } catch {
      setError("Your details could not be saved right now.");
    } finally {
      setSaving(false);
    }
  };

  if (!open) {
    return (
      <button
        type="button"
        disabled={saved}
        onClick={() => setOpen(true)}
        style={{
          position: "absolute",
          right: "1.5vh",
          bottom: "1.5vh",
          maxWidth: "16vh",
          background: "transparent",
          border: "none",
          fontFamily: FONT,
          fontSize: "1.4vh",
          color: BROWN,
          textDecoration: saved ? "none" : "underline",
          cursor: saved ? "default" : "pointer",
        }}
      >
        {saved ? "Details saved" : "Save my details (optional)"}
      </button>
    );
  }

  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        zIndex: 20,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "rgba(0, 0, 0, 0.45)",
      }}
    >
      <div
        role="dialog"
        aria-label="Save my details"
        style={{
          boxSizing: "border-box",
          width: "40vh",
          padding: "2.4vh",
          display: "flex",
          flexDirection: "column",
          gap: "1.2vh",
          background: "#F7F4F0",
          borderRadius: "1.6vh",
          fontFamily: FONT,
          color: BROWN,
        }}
      >
        <h2 style={{ margin: 0, fontSize: "2vh", fontWeight: 600 }}>Save my details</h2>
        <input style={fieldStyle} placeholder="Name (optional)" autoComplete="off" maxLength={100} value={name} onChange={(e) => setName(e.target.value)} />
        <input style={fieldStyle} placeholder="Phone number (optional)" autoComplete="off" inputMode="tel" maxLength={24} value={phone} onChange={(e) => setPhone(e.target.value)} />
        <select style={fieldStyle} aria-label="Gender (optional)" value={gender} onChange={(e) => setGender(e.target.value)}>
          <option value="">Gender (optional)</option>
          <option value="female">Female</option>
          <option value="male">Male</option>
          <option value="non_binary">Non-binary</option>
          <option value="prefer_not_to_say">Prefer not to say</option>
        </select>
        <label style={{ display: "flex", gap: "1vh", alignItems: "flex-start", fontSize: "1.4vh", lineHeight: "2vh" }}>
          <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} style={{ width: "2vh", height: "2vh", flexShrink: 0 }} />
          <span>
            I agree that these details can be saved with the outfits and sizes I tried, for future visits. My photo and try-on
            image are not saved with them. I can ask staff to delete my details at any time.
          </span>
        </label>
        {error && <span role="alert" style={{ fontSize: "1.4vh", color: "#A32A2A" }}>{error}</span>}
        <div style={{ display: "flex", gap: "1vh" }}>
          <button type="button" onClick={() => setOpen(false)} style={{ ...buttonStyle, background: "transparent", color: BROWN }}>
            No thanks
          </button>
          <button
            type="button"
            disabled={!canSave}
            onClick={() => void save()}
            style={{ ...buttonStyle, background: BROWN, color: "#FFFFFF", opacity: canSave ? 1 : 0.45 }}
          >
            {saving ? "Saving..." : "Save"}
          </button>
        </div>
      </div>
    </div>
  );
}

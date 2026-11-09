# Virtual Trial Room (VTO + Beauty Studio): Functional Specification

**Version:** 3.1
**Product type:** In-store kiosk application (Windows desktop / Electron front end + local FastAPI back end)

---

## 1. Purpose

Virtual Trial Room is a self-service kiosk. A shopper stands in front of a camera and:

1. **Clothing try-on:** sees themselves wearing a catalogue garment in a chosen size, without changing clothes.
2. **Beauty Studio:** previews lipstick and compact-powder shades live on their own face.

The shopper can then take the result home on their phone by scanning a QR code. No account or login is needed, and biometric data is deleted when the session ends.

## 2. Users

| User | Description |
|---|---|
| **Shopper (visitor)** | Uses the kiosk unassisted with a touch screen and a camera. |
| **Store staff / operator** | Starts the kiosk, picks the camera, calibrates it, and handles failures. |
| **Administrator** | Installs the app, configures the `.env` file (API key, thresholds), and reviews cost and latency logs. |

## 3. Scope

**In scope**
- Camera selection and calibration
- Full-body capture with on-screen guidance
- Body-proportion analysis and size recommendation (XS–XXL)
- AI-generated try-on images for a fixed garment catalogue
- Lipstick and powder try-on (live preview plus a saved studio portrait)
- QR-code download to a phone on the same Wi-Fi network
- Session purge and zero-retention handling of personal data

**Out of scope**
- User accounts, purchase or checkout, and inventory lookup
- Garments uploaded by the shopper
- Storing shopper photos after the session (except optional calibration aids, which are off by default)

## 4. Screen Flow

```
Camera Selection ──► Welcome ──► Capture ──► Confirmation ──► Outfit Selection ──► Generating ──► Result ──► QR
       │                 │                        ▲  (Retake)        │      ▲                        │
       ▼                 └────────► Beauty Studio ◄──────────────────┘      └── Try something else ──┘
  Calibration
```

The **Home** button on any screen returns to Welcome and resets the session.

## 5. Functional Requirements

### FR-1 Camera Selection and Calibration
| ID | Requirement |
|---|---|
| FR-1.1 | On launch, the operator chooses the camera to use. |
| FR-1.2 | The operator can open the calibration screen. The focal length can be calibrated for each kiosk using a person of known height at 2.5 m (`tools/calibrate_focal_length.py`). |
| FR-1.3 | URL shortcuts skip directly to a screen: `?screen=welcome`, `?screen=capture`, `?screen=beauty`, `?calibrate`. |

### FR-2 Welcome
| ID | Requirement |
|---|---|
| FR-2.1 | Shows a welcome video and two entry points: **Clothing Try-On** and **Beauty Studio**. |
| FR-2.2 | Arriving at Welcome always resets the previous shopper's session. |
| FR-2.3 | Background music loops at low volume (15%) on every screen. |

### FR-3 Capture
| ID | Requirement |
|---|---|
| FR-3.1 | A live camera feed is shown with pose tracking to guide the shopper into the frame. |
| FR-3.2 | The shopper taps **Start Capture**, and a 5 → 1 countdown follows (1 second per step). |
| FR-3.3 | When the countdown ends, the photo is taken only if the shopper faces the camera and is centred. If not, the message "We couldn't see you — tap Start Capture to try again" is shown. |
| FR-3.4 | Up to 2 extra "hold" frames are captured to reduce measurement jitter. |
| FR-3.5 | A pre-flight check (`/api/capture-preflight`) confirms that a usable face is present. If it fails, the reason is shown along with "Tap Start Capture to retake". |
| FR-3.6 | Body-profile extraction (`/api/extract-profile`) computes the body proportions, body type, torso fullness, and display-only measurements in cm. |
| FR-3.7 | The recommended capture distance is 2–3 m, with 2.5 m as the optimum. |

### FR-4 Confirmation
| ID | Requirement |
|---|---|
| FR-4.1 | Shows the captured photo with **Retake** or **Continue**. |
| FR-4.2 | Can overlay estimated measurements (height, shoulder, chest, waist). These are informational only and never used for sizing. |
| FR-4.3 | The shopper can correct a measurement (`/api/measurement-correction`). Corrections are logged only when `MEASUREMENT_CORRECTION_LOG=true`. |

### FR-5 Outfit Selection and Sizing
| ID | Requirement |
|---|---|
| FR-5.1 | Shows the garment catalogue (see Appendix A), covering western and traditional Indian wear for women and men. |
| FR-5.2 | When a garment is selected, the system requests a size recommendation (`/api/recommend-size`) from the body profile and the garment category. |
| FR-5.3 | Sizes are **XS, S, M, L, XL, XXL**. The recommended size is highlighted, and each size shows a fit label (for example "Relaxed fit") and a fit quality. |
| FR-5.4 | The shopper can pick any size and a fit style: **Fitted** (tailored cut), **Standard**, or **Relaxed** (casual drape). |
| FR-5.5 | From this screen the shopper can switch to Beauty Studio without losing the capture. |

### FR-6 Generating (Try-On Pipeline)
| ID | Requirement |
|---|---|
| FR-6.1 | Shows a progress screen with rotating voice-over messages ("Styling Your Look", "Tailoring Your Experience", and others). |
| FR-6.2 | **Person validation:** a person detector (YOLOX-S) and pose landmarks (MediaPipe) confirm that one person is visible with the nose, shoulders and hips in frame. Otherwise the shopper is asked to realign. |
| FR-6.3 | **Generation:** Google Gemini (`gemini-3.1-flash-image`) renders the shopper wearing the garment at the chosen size and fit. The request is conditioned on body shape and garment analysis (category, fabric, structure, sleeve length, hem length). |
| FR-6.4 | **Identity preservation:** the shopper's real face is warped and composited onto the AI output. A face-similarity score is calculated. |
| FR-6.5 | **Quality gates:** silhouette verification (threshold 0.60), a garment coverage check (sleeve and hem), a limb check, and a check that different sizes produce visibly different renders. |
| FR-6.6 | **Retries:** at most 3 generation attempts per request (`GENERATION_MAX_ATTEMPTS`). If every attempt fails, the last good render is served with `verification_failed` / `fallback_used` set. |
| FR-6.7 | **Won't fit:** if the selected size is too small for the body, the result shows the garment visibly too small together with a message that suggests the recommended size. If no believable render is possible, an illustration is shown instead. |
| FR-6.8 | Each render is watermarked, saved for download, and returned with a QR code. |

### FR-7 Result
| ID | Requirement |
|---|---|
| FR-7.1 | Shows the try-on image with the size and fit label. |
| FR-7.2 | The shopper can change size or fit and regenerate. Each change is a new full generation. |
| FR-7.3 | Actions: **Try something else** (back to outfits) or **I like this outfit** (QR screen). |

### FR-8 QR Download
| ID | Requirement |
|---|---|
| FR-8.1 | Shows a QR code with the text "Scan the QR code to download and save your virtual try-on." |
| FR-8.2 | The QR code opens `/download/{image_id}`, a mobile page that previews the image and offers a download (`/api/image/{image_id}?download=true`). |
| FR-8.3 | The QR base URL is `PUBLIC_BASE_URL` when it is set to a non-localhost address. Otherwise it is the kiosk's LAN IP on port 8888, so the phone must be on the same Wi-Fi. |
| FR-8.4 | Only the most recent 10 images are kept (`MAX_TEMP_IMAGES`). Older links stop working. |

### FR-9 Beauty Studio
| ID | Requirement |
|---|---|
| FR-9.1 | Can be reached from Welcome (no body capture needed) or from Outfit Selection. |
| FR-9.2 | Loads the lipstick shades (`/api/beauty/shades`) and powder shades (`/api/beauty/powders`). See Appendix B. |
| FR-9.3 | Lipstick options: shade, finish (**Matte / Satin / Gloss**), and intensity (0–1, default 0.85). |
| FR-9.4 | Powder options: shade, finish (**Matte / Natural Velvet / Luminous**), and coverage (**Sheer / Medium**). Translucent powder adds no colour. |
| FR-9.5 | Modes: lipstick only, powder only, or full makeover (both). |
| FR-9.6 | **Live preview** is processed locally at no cost. Previews are never saved to disk. |
| FR-9.7 | **Save look** makes one studio portrait with Gemini (lighting and backdrop), puts the real face back, applies the makeup, and returns a QR code. If any step fails, the local result is saved instead. |
| FR-9.8 | The face area is returned as a bounding box so the UI can zoom into a full-length capture. |
| FR-9.9 | If a photo cannot be read, the message is "The photo could not be read. Please retake it." |

### FR-10 Session and Privacy
| ID | Requirement |
|---|---|
| FR-10.1 | Each shopper session has its own encryption key (Fernet). The body profile is stored only as ciphertext in memory. |
| FR-10.2 | Sessions expire after 15 minutes (`SESSION_TTL_SECONDS=900`). |
| FR-10.3 | `/api/session/purge` destroys the key, so all of that session's biometric data becomes unrecoverable. |
| FR-10.4 | Calibration photo saving and correction logging are **off by default** and must stay off in production. |

## 6. API Summary

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/capture-preflight` | Face check on the capture |
| POST | `/api/extract-profile` | Body profile and measurements |
| POST | `/api/measurement-correction` | Shopper correction of displayed cm |
| POST | `/api/recommend-size` | Size recommendation for a garment |
| POST | `/api/tryon` | Full clothing try-on pipeline |
| POST | `/api/verify-silhouette` | Stand-alone silhouette check |
| POST | `/api/session/purge` | Destroy session data |
| GET | `/api/beauty/shades` | Lipstick catalogue |
| GET | `/api/beauty/powders` | Powder catalogue |
| POST | `/api/beauty/tryon` | Lipstick / powder render (preview or save) |
| GET | `/download/{image_id}` | Mobile download page |
| GET | `/api/image/{image_id}` | Raw JPEG |

## 7. Non-Functional Requirements

| Area | Requirement |
|---|---|
| Platform | Windows 10/11 and Python 3.10+. The back end runs on port 8888 and serves the built front end at `/`. |
| Start-up | `run_app.bat` creates the virtual environment, installs dependencies, warns when `.env` is missing, and opens the browser. A packaged `.exe` / Electron build is also available. |
| Performance | The person detector warms up in the background at start-up. Per-step timings are logged to `logs/tryon_timings.csv`. |
| Cost monitoring | Each Gemini call is logged to `logs/gemini_costs.csv` (tokens, USD estimate, 8-character session prefix, size). The log contains no images or personal data. |
| Connectivity | Internet is required for Gemini. LAN access is required for the QR download. |
| Privacy | No shopper images are kept beyond the 10-image download window. Biometric data is encrypted and expires. |

## 8. Configuration (`backend/.env`)

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | — | Required for generation |
| `PUBLIC_BASE_URL` | LAN IP:8888 | Base URL used in QR codes |
| `SESSION_TTL_SECONDS` | 900 | Session lifetime |
| `GENERATION_MAX_ATTEMPTS` | 3 | Gemini attempts per try-on |
| `SILHOUETTE_VERIFY_THRESHOLD` | 0.60 | Silhouette pass score |
| `YOLO_VALIDATION_ENABLED` / `BODY_VALIDATION_ENABLED` | true | Person and body gates |
| `CAPTURE_DISTANCE_MIN/MAX/OPTIMAL_METERS` | 2.0 / 3.0 / 2.5 | Capture distance |
| `MEASUREMENT_FOCAL_LENGTH_PX` | derived | Per-kiosk calibration |
| `GEMINI_COST_LOG`, `TRYON_TIMING_LOG` | true | Monitoring logs |
| `CALIBRATION_CAPTURE_SAVE`, `MEASUREMENT_CORRECTION_LOG` | false | Testing aids only |

## 9. Error Handling (Shopper-Facing)

| Situation | Behaviour |
|---|---|
| Shopper not visible or not centred at the shutter | Capture cancelled, "We couldn't see you" message |
| No usable face | Pre-flight message and a retake prompt |
| Body not fully in frame | `needs_visitor_realignment` prompt to step back and realign |
| Gemini error or rejected render | Automatic retry within the 3-attempt budget |
| All quality gates fail | Best render served and the failure logged |
| Size too small | Won't-fit notice with the recommended size |
| Beauty "Save look" fails | Local makeup result saved instead |
| `.env` missing | App starts, but generation fails, and a warning is shown at start-up |

---

## Appendix A: Garment Catalogue

| ID | Name | Category |
|---|---|---|
| female1 | Smart Casual Shirt | shirt |
| female2 | Floral Tunic | tunic |
| female3 | Tailored Cream Shirt | shirt |
| kurti1 | Floral Panel Kurti | kurti |
| salwar_suit1 | Embroidered Salwar Suit | salwar_suit |
| lehenga1 | Embroidered Bridal Lehenga | lehenga |
| kerala_saree1 | Kerala Kasavu Saree | saree |
| kurta1 | Silk Kurta Set | kurta |
| sherwani1 | Embroidered Sherwani | sherwani |
| royal_sherwani1 | Layered Gold Sherwani | royal_sherwani |
| kurta_dhoti1 | Kurta with Dhoti | kurta_dhoti |
| kurta_pajama1 | Magenta Kurta with Pants | kurta_pajama |
| mundu_shirt1 | Shirt with Kasavu Mundu | mundu_shirt |
| male1 | Formal Shirt | shirt |
| male2 | Cream Button-Down Shirt | shirt |
| male3 | Color-Block Zip Polo | polo_shirt |
| blazer1 | Textured Grey Blazer | blazer |
| blazer2 | Charcoal Blazer | blazer |
| outfit1 | Navy Pocket Shirt | shirt |
| outfit2 | Brown Knit Polo | polo_shirt |

## Appendix B: Beauty Catalogue

**Lipstick shades** (tuned for Indian skin tones)

| Family | Shades (default finish) |
|---|---|
| Roses | Dusty Rose (Matte), Deep Rose (Satin) |
| Reds | Terracotta (Matte), Brick Red (Matte), Classic Red (Satin), Sindoor Red (Gloss) |
| Berries & Wines | Berry (Satin), Maroon (Matte), Wine (Matte), Plum (Satin) |
| Browns | Chocolate Brown (Matte) |

**Compact powders** (default finish: Natural Velvet)

Warm Ivory, Light Beige, Natural Honey, Golden Medium, Caramel, Warm Tan, Deep Bronze, Espresso

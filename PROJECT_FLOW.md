# Virtual Trial Room: Project Flow and Functionalities

## Overview

Virtual Trial Room is an in-store kiosk. Shoppers stand in front of a camera and see themselves wearing store garments, or try on lipstick shades, without changing clothes or applying makeup. They take the result home by scanning a QR code with their phone.

## Architecture

The system has three parts. All of them run on the kiosk PC, except the AI image model, which runs in the cloud.

| Layer | Technology | Role |
|---|---|---|
| **Frontend** | React + TypeScript (Vite), packaged as a Windows desktop app with Electron | Kiosk screens, camera feed, live pose guidance, voice-overs and music |
| **Backend** | Python FastAPI server on port 8888 | Image checks, body analysis, sizing, try-on pipeline, lipstick, QR codes, sessions |
| **AI service** | Google Gemini (`gemini-3.1-flash-image`) | Creates the try-on image and the studio portrait |

**Local AI models (run on the kiosk, offline):**
- **MediaPipe Pose and Face Landmarker**: body points and face/lip outline
- **YOLOX-S** (run with OpenCV): detects the person in the photo

**How the parts talk to each other:**
1. The frontend sends the photo (Base64) to the backend over HTTP/JSON.
2. The backend runs the local checks, then calls Gemini over the internet.
3. The backend saves the final image and returns it with a QR code.
4. A phone scans the QR code and downloads the image directly from the kiosk over the store Wi-Fi.

```
 Kiosk PC                                             Cloud
┌──────────────────────┐    HTTP/JSON    ┌───────────────────────────┐     ┌────────────┐
│ Frontend (Electron / │ ──────────────► │ Backend (FastAPI :8888)   │ ──► │  Google    │
│ React)               │ ◄────────────── │ MediaPipe · YOLOX · OpenCV│ ◄── │  Gemini    │
└──────────────────────┘                 └─────────────┬─────────────┘     └────────────┘
                                                       │ QR link (same Wi-Fi)
                                                 ┌─────▼─────┐
                                                 │   Phone   │
                                                 └───────────┘
```

## Flow

### 1. Camera Setup
Store staff choose the kiosk camera when the app starts. They can also calibrate it once so that height and measurement estimates are accurate.

### 2. Welcome Screen
A welcome video plays with background music. The shopper chooses one of two options:
- **Clothing Try-On**
- **Beauty Studio** (lipstick)

Every return to this screen clears the previous shopper's data.

### 3. Photo Capture
- A live camera view guides the shopper to stand 2–3 m away, facing the camera.
- The shopper taps **Start Capture**, and a 5-second countdown begins.
- The photo is taken only if the shopper is visible, centred and facing forward. Otherwise they are asked to try again.
- The system checks that the face is clear, then reads the shopper's body shape and proportions.

### 4. Confirmation
The shopper sees their photo and chooses **Retake** or **Continue**. Estimated measurements (height, shoulder, chest, waist) can be shown and corrected.

### 5. Outfit Selection
- The shopper browses 20 garments: western wear (shirts, polos, blazers, tunics) and traditional wear (saree, lehenga, kurti, salwar suit, kurta sets, sherwanis, mundu, dhoti).
- The system recommends a size from **XS to XXL** based on the shopper's body shape.
- The shopper can accept the recommended size or pick another one, and can choose a fit style: **Fitted**, **Standard** or **Relaxed**.

### 6. Generating the Try-On
While a progress screen plays voice-over messages, the system:
1. Confirms that one person is fully in frame.
2. Uses AI (Google Gemini) to create an image of the shopper wearing the garment in the chosen size and fit.
3. Puts the shopper's real face back onto the image so they look like themselves.
4. Checks the quality of the result (body shape, sleeve and hem length, arms and legs). If the check fails, it retries automatically, up to 3 attempts.

If the chosen size is too small, the image shows the garment as a tight fit and suggests the right size.

### 7. Result
The shopper sees the try-on image. They can:
- change the size or fit and regenerate,
- tap **Try something else** to go back to outfits,
- tap **I like this outfit** to save it.

### 8. QR Download
A QR code appears. When the shopper scans it with a phone on the same Wi-Fi, a page opens where they can view and download the image.

## Beauty Studio (Lipstick)

1. The shopper opens Beauty Studio from the Welcome screen or the Outfit screen.
2. They pick a lipstick shade from 11 options: Roses, Reds, Berries & Wines, Browns.
3. They adjust the finish (**Matte**, **Satin** or **Gloss**) and the intensity.
4. The colour appears live on their face, instantly.
5. **Save look** creates a polished studio portrait with the lipstick applied and shows a QR code for download.

## Backend in Detail

The backend lives in `backend/backend/` and starts from `main.py`. It has three groups of API routes, which use a set of services.

### API Endpoints

| Endpoint | What it does |
|---|---|
| `POST /api/capture-preflight` | Checks that the captured photo has a clear, usable face |
| `POST /api/extract-profile` | Reads body shape and proportions from the photo, and estimates measurements |
| `POST /api/measurement-correction` | Saves a measurement the shopper corrected |
| `POST /api/recommend-size` | Recommends a size (XS–XXL) for the chosen garment |
| `POST /api/tryon` | Runs the full clothing try-on and returns the image and QR code |
| `POST /api/session/purge` | Deletes all of a shopper's data |
| `GET /api/beauty/shades` | Lists the lipstick shades |
| `POST /api/beauty/tryon` | Applies lipstick (live preview, or saved look with QR) |
| `GET /download/{id}` | Mobile download page opened from the QR code |
| `GET /api/image/{id}` | Returns the image file |

The backend also serves the built frontend, so the whole app runs from one address.

### Services (`app/services/`)

**Body and sizing**
- **Body Profile**: uses MediaPipe to read shoulder, chest, waist and hip proportions, body type, and torso fullness. It averages up to 3 frames to reduce jitter.
- **Size Chart**: matches those proportions to XS–XXL and rates how well each size fits (tight, good, loose).
- **Fit & Drape**: describes to the AI how the chosen size and fit style should hang on this shopper's body.

**Photo validation**
- **YOLOX Detector / YOLO Service**: confirms there is exactly one person in the photo.
- **Body Validation**: cross-checks YOLOX against MediaPipe to make sure the head, shoulders and hips are clearly in frame.

**Try-on generation and quality checks**
- **Gemini Client**: sends the shopper photo, the garment image and detailed instructions to Gemini, and handles retries.
- **Face Preservation**: replaces the AI-drawn face with the shopper's real face, so the result looks like them.
- **Silhouette Verifier**: checks that the body shape in the result matches the shopper's real body.
- **Garment Coverage**: checks that sleeves and hem end where the garment's do, with no extra sleeve or missing fabric.
- **Limb Check**: makes sure no arm or leg is missing from the result.

**Lipstick**
- **Lipstick Service**: finds the lips with the face landmarks and recolours them locally (no AI cost, instant).
- **Beauty Studio Service**: for **Save look**, creates a studio-quality portrait with Gemini, then applies the shopper's real face and lipstick.

**Sharing, sessions and monitoring**
- **QR Service**: creates the QR code that links to the download page on the kiosk's network address.
- **Session Manager**: gives each shopper an encrypted session that expires after 15 minutes.
- **Gemini Cost / Request Timing**: log each AI call's cost and each step's duration to CSV files in `logs/` (no photos or personal data).

### Try-On Pipeline (`/api/tryon`)

```
Photo + Garment
     │
     ▼
Person check (YOLOX) ─► Body check (MediaPipe) ─► Size & fit instructions
     │
     ▼
Gemini generates image ──► Real face restored ──► Quality checks
     ▲                                                │
     └──────────── retry (max 3 attempts) ◄── fail ───┤
                                                      ▼ pass
                                      Watermark ─► Save ─► QR code ─► Response
```

### Configuration

Settings live in `backend/backend/.env`:
- **`GEMINI_API_KEY`**: required for AI generation
- **`PUBLIC_BASE_URL`**: optional; the address used in QR codes (by default, the kiosk's Wi-Fi IP)
- Optional tuning: session timeout, number of retries, quality-check thresholds, capture distance

### Tests and Tools
- `tests/`: automated tests covering sizing, body validation, face preservation, lipstick, sessions and the try-on pipeline.
- `tools/`: helper scripts for camera calibration, latency measurement and drape validation.

## Privacy

- No login or account is required.
- Body data is encrypted for each shopper and deleted automatically after 15 minutes, or when the session ends.
- Only the 10 most recent images are kept for download. Older ones are removed.

## Running the App

- Start the app with `run_app.bat`, or run the packaged desktop app.
- Internet access is needed for AI image generation.
- The phone and the kiosk must be on the same Wi-Fi for the QR download to work.

# Virtual Trial Room — Backend

A high-performance FastAPI Python backend that powers the Virtual Trial Room application. It orchestrates local computer vision (MediaPipe), cloud generative AI (Google Gemini 3.1 Flash Image), and robust session security to provide a seamless virtual try-on experience.

## Core Features

- **Body Profile Extraction**: Uses MediaPipe Pose and Face landmarks to extract biometric proportions and recommend accurate garment sizing.
- **Generative Try-On**: Leverages Google `gemini-3.1-flash-image` for 4K portrait generation with deterministic seed stabilization and fabric clearance conditioning.
- **Pixel-Level Identity Preservation**: Ensures 100% genuine identity preservation by warping and compositing the user's real captured face onto the AI output using affine transformations and Lanczos interpolation.
- **Silhouette Verification**: Automated QA gates ensure cross-fit skeletal drift and anatomical fidelity are maintained.
- **Zero-Retention Security**: Implements ephemeral Fernet symmetric encryption per session. All biometric data is cryptographically destroyed upon session purge.
- **QR Mobile Handoff**: Generates high-contrast QR codes and dynamic LAN IP routing for touchless mobile photo downloads.

## Prerequisites

- Python 3.10+
- A valid [Gemini API Key](https://aistudio.google.com/app/apikey)

## Setup Instructions

1. **Create a virtual environment** (recommended):

   ```bash
   cd backend
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:

   Create or edit the `.env` file in the `backend` directory:

   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   PUBLIC_BASE_URL=http://localhost:8888
   ```

   - `GEMINI_API_KEY` — Your Google Gemini API key.
   - `PUBLIC_BASE_URL` — Optional. The publicly accessible URL of this server (used for QR code generation). If omitted, the server attempts to dynamically resolve your LAN IP.

## Running the Server

You can start the server using the provided batch file (on Windows):
```cmd
run_app.bat
```

Or manually via Uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8888 --reload
```

The server will start at `http://localhost:8888`. If the frontend is built, the backend will also serve the static frontend assets from `/`.

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **POST** | `/api/extract-profile` | Decodes image/frames, runs MediaPipe profile extraction, and stores the encrypted profile in session memory. |
| **POST** | `/api/recommend-size` | Evaluates the encrypted body profile against garment categories to recommend the ideal retail size. |
| **POST** | `/api/tryon` | The master orchestration pipeline for generative try-on. Handles rendering, face restoration, verification, and watermarking. |
| **POST** | `/api/session/purge` | Cryptographically destroys the session's Fernet key and purges all biometric ciphertext. |
| **GET** | `/download/{image_id}` | Serves a mobile-responsive landing page to preview and download the generated try-on photograph. |
| **GET** | `/api/image/{image_id}` | Retrieves the raw generated image file (JPEG). |

### Example: POST `/api/tryon`

**Request Body** (JSON):
```json
{
  "user_image": "data:image/png;base64,...",
  "dress_image": "data:image/png;base64,...",
  "session_id": "session_uuid_here",
  "selected_size": "M",
  "fit_state": "standard"
}
```

**Response** (JSON):
```json
{
  "success": true,
  "result_image": "data:image/jpeg;base64,...",
  "qr_code_base64": "data:image/png;base64,...",
  "silhouette_passed": true,
  "face_similarity": 0.96
}
```

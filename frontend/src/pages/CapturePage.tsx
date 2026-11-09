import { useEffect, useRef, useState, useCallback } from "react";
import NavigationButton from "../components/NavigationButton";
import PrimaryButton from "../components/PrimaryButton";
import { useTrialRoom } from "../context/TrialRoomContext";
import { API_BASE_URL } from "../utils/api";

type PageProps = {
  onHome: () => void;
  onCapture: () => void;
};

interface PoseKeypoint {
  part: string;
  score: number;
  position: { x: number; y: number };
}

interface Pose {
  score: number;
  keypoints: PoseKeypoint[];
}

interface PoseNetModel {
  estimateSinglePose(
    input: HTMLVideoElement,
    config?: { flipHorizontal: boolean }
  ): Promise<Pose>;
}

// ─── Capture Constants ───────────────────────────────────────────────────────
const SHOULDER_VISIBILITY_THRESHOLD = 0.60;
// The visitor taps Start Capture at the screen, then has the countdown (5 → 1) to get into the frame
const COUNTDOWN_FROM = 5;
const COUNTDOWN_STEP_MS = 1000;
// The visitor sees one photo; two extra frames grabbed while they hold still let the
// backend take the median of each body measurement to suppress per-frame landmark jitter.
const HOLD_FRAME_COUNT = 2;
const HOLD_FRAME_INTERVAL_MS = 150;
const FACE_CHECK_TIMEOUT_MS = 5000;
const NOT_SEEN_MESSAGE = "We couldn't see you — tap Start Capture to try again";
const RETAKE_HINT = "Tap Start Capture to retake";

/**
 * Runs the pre-flight checks (/api/capture-preflight) on a new capture. Returns the message to show when a
 * check fails (lighting, body visibility, face), else null. Fails open when the check can't run.
 */
async function runCapturePreflight(image: string): Promise<string | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), FACE_CHECK_TIMEOUT_MS);
  try {
    const res = await fetch(`${API_BASE_URL}/api/capture-preflight`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image }),
      signal: controller.signal,
    });
    if (res.status !== 422) return null;
    const data = await res.json().catch(() => ({}));
    return typeof data.detail === "string" ? data.detail : null;
  } catch (err) {
    console.warn("[PREFLIGHT CHECK FAILED] Continuing; checks are re-run or validated later.", err);
    return null;
  } finally {
    clearTimeout(timer);
  }
}

// Whether the visitor can be photographed. Distance from the camera is not measured.
//   none        — nobody detected
//   face_camera — shoulders not both visible (turned away or sideways)
//   bad_framing — shoulders off-centre or cut off by the frame edge
type FramingStatus = "none" | "face_camera" | "bad_framing" | "framed";

// Live guidance per status (the badge shows it as-is; the line under the camera in capitals)
const FRAMING_PROMPTS: Record<FramingStatus, string> = {
  none: "Stand in front of the camera",
  face_camera: "Face the camera directly",
  bad_framing: "Center yourself in the frame",
  framed: "In frame ✓",
};

export interface DiagnosticsData {
  frameWidth: number;
  frameHeight: number;
  leftShoulderScore: number;
  rightShoulderScore: number;
  minXRatio: number;
  maxXRatio: number;
  minYRatio: number;
  maxYRatio: number;
  isHorizontallyFramed: boolean;
  isVerticallyFramed: boolean;
  failReason: string;
  shoulderWidthPx: number | null;
}

/**
 * Checks that both shoulders are visible (MediaPipe/PoseNet landmarks 11/12) and sit centred and uncut in the frame.
 */
function computeFraming(
  pose: Pose,
  frameWidth: number,
  frameHeight: number
): { status: FramingStatus; diag: DiagnosticsData } {
  const emptyDiag: DiagnosticsData = {
    frameWidth: frameWidth || 0,
    frameHeight: frameHeight || 0,
    leftShoulderScore: 0,
    rightShoulderScore: 0,
    minXRatio: 0,
    maxXRatio: 0,
    minYRatio: 0,
    maxYRatio: 0,
    isHorizontallyFramed: false,
    isVerticallyFramed: false,
    failReason: !pose
      ? "No pose detected"
      : pose.score < 0.15
      ? `Overall pose score low (${(pose.score * 100).toFixed(0)}% < 15%)`
      : "Waiting for camera frame...",
    shoulderWidthPx: null,
  };

  if (!pose || pose.score < 0.15 || !frameWidth || !frameHeight || frameWidth <= 0 || frameHeight <= 0) {
    return { status: "none", diag: emptyDiag };
  }

  const kpMap: Record<string, PoseKeypoint> = {};
  for (const kp of pose.keypoints) {
    kpMap[kp.part] = kp;
  }

  const leftShoulder = kpMap["leftShoulder"];
  const rightShoulder = kpMap["rightShoulder"];

  const leftScore = leftShoulder?.score ?? 0;
  const rightScore = rightShoulder?.score ?? 0;

  // Keypoint coordinates from PoseNet: handle both raw pixels and normalized [0, 1]
  const rawMinX = leftShoulder && rightShoulder ? Math.min(leftShoulder.position.x, rightShoulder.position.x) : 0;
  const rawMaxX = leftShoulder && rightShoulder ? Math.max(leftShoulder.position.x, rightShoulder.position.x) : 0;
  const rawMinY = leftShoulder && rightShoulder ? Math.min(leftShoulder.position.y, rightShoulder.position.y) : 0;
  const rawMaxY = leftShoulder && rightShoulder ? Math.max(leftShoulder.position.y, rightShoulder.position.y) : 0;

  let shoulderWidthPx =
    leftShoulder && rightShoulder
      ? Math.hypot(
          leftShoulder.position.x - rightShoulder.position.x,
          leftShoulder.position.y - rightShoulder.position.y
        )
      : null;

  // If coordinates were normalized ([0, 1]), scale shoulderWidthPx to frame pixels
  if (shoulderWidthPx !== null && rawMaxX <= 1.0 && frameWidth > 1) {
    shoulderWidthPx = shoulderWidthPx * frameWidth;
  }

  // Convert to normalized ratios [0, 1]
  const minXRatio = rawMaxX <= 1.0 ? rawMinX : frameWidth > 0 ? rawMinX / frameWidth : 0;
  const maxXRatio = rawMaxX <= 1.0 ? rawMaxX : frameWidth > 0 ? rawMaxX / frameWidth : 0;
  const centerXRatio = (minXRatio + maxXRatio) / 2;

  const minYRatio = rawMaxY <= 1.0 ? rawMinY : frameHeight > 0 ? rawMinY / frameHeight : 0;
  const maxYRatio = rawMaxY <= 1.0 ? rawMaxY : frameHeight > 0 ? rawMaxY / frameHeight : 0;

  // Horizontal framing: shoulders not clipped at edges (6%-94%) AND torso centered in camera view (25%-75%)
  const isHorizontallyFramed =
    minXRatio >= 0.06 &&
    maxXRatio <= 0.94 &&
    centerXRatio >= 0.25 &&
    centerXRatio <= 0.75;

  // Vertical framing: shoulders below top 3% and not cut off below bottom 85%
  const isVerticallyFramed = minYRatio >= 0.03 && maxYRatio <= 0.85;

  let failReason = "None (Framed OK)";
  if (leftScore < SHOULDER_VISIBILITY_THRESHOLD || rightScore < SHOULDER_VISIBILITY_THRESHOLD) {
    failReason = `Confidence low (L: ${(leftScore * 100).toFixed(0)}%, R: ${(rightScore * 100).toFixed(0)}% < 60%)`;
  } else if (!isHorizontallyFramed) {
    if (centerXRatio < 0.25 || minXRatio < 0.06) {
      failReason = `Step to center (too far left: ${(centerXRatio * 100).toFixed(0)}%)`;
    } else if (centerXRatio > 0.75 || maxXRatio > 0.94) {
      failReason = `Step to center (too far right: ${(centerXRatio * 100).toFixed(0)}%)`;
    }
  } else if (!isVerticallyFramed) {
    if (minYRatio < 0.03) failReason = `Shoulders cut off at top (${(minYRatio * 100).toFixed(0)}%)`;
    else if (maxYRatio > 0.85) failReason = `Shoulders too low in frame (${(maxYRatio * 100).toFixed(0)}%)`;
  }

  const baseDiag: DiagnosticsData = {
    frameWidth,
    frameHeight,
    leftShoulderScore: leftScore,
    rightShoulderScore: rightScore,
    minXRatio,
    maxXRatio,
    minYRatio,
    maxYRatio,
    isHorizontallyFramed,
    isVerticallyFramed,
    failReason,
    shoulderWidthPx,
  };

  // 1. Visibility Guard: ensure both shoulders are detected with >= 0.6 confidence
  if (
    !leftShoulder ||
    !rightShoulder ||
    leftScore < SHOULDER_VISIBILITY_THRESHOLD ||
    rightScore < SHOULDER_VISIBILITY_THRESHOLD
  ) {
    return { status: "face_camera", diag: baseDiag };
  }

  // 2. Framing Check: shoulders and upper torso region within frame boundaries
  if (!isHorizontallyFramed || !isVerticallyFramed) {
    return { status: "bad_framing", diag: baseDiag };
  }

  // 3. Shoulders almost on top of each other: turned sideways
  if (!shoulderWidthPx || shoulderWidthPx <= 20) {
    return { status: "face_camera", diag: { ...baseDiag, failReason: "Shoulder width too small (<=20px)" } };
  }

  return { status: "framed", diag: baseDiag };
}

export default function CapturePage({ onHome, onCapture }: PageProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const {
    sessionId,
    setCapturedImage,
    setBodyProfile,
    setBodyMeasurements,
    setMeasurementScale,
  } = useTrialRoom();
  const [cameraActive, setCameraActive] = useState(false);

  const [modelLoading, setModelLoading] = useState(true);
  const [isCapturing, setIsCapturing] = useState(false);
  const [countdown, setCountdown] = useState<number | null>(null); // 5 → 1 after Start Capture; null when idle
  // Why the last attempt produced no photo (not in frame, or face not visible); cleared by the next Start Capture
  const [retakeNotice, setRetakeNotice] = useState<{ message: string; hint: string | null } | null>(null);
  const [framingStatus, setFramingStatus] = useState<FramingStatus>("none");
  const lastLogTimeRef = useRef(0);
  const frameCountRef = useRef(0);
  const fpsRef = useRef(0);
  const lastFpsCalcRef = useRef(Date.now());

  const posenetModelRef = useRef<PoseNetModel | null>(null);
  const framingStatusRef = useRef<FramingStatus>("none");
  const isCapturingRef = useRef(false); // Capture committed: stops the detection loop
  const burstInProgressRef = useRef(false); // Hold-still frames being grabbed; detection keeps running
  const capturePhotoRef = useRef<() => void>(() => {});

  useEffect(() => {
    let isMounted = true;
    let activeStream: MediaStream | null = null;

    const startCamera = async () => {
      try {
        const selectedCameraId = localStorage.getItem("selectedCameraId");
        const videoConstraints: MediaTrackConstraints = selectedCameraId
          ? {
              deviceId: { exact: selectedCameraId },
              width: { min: 640, ideal: 1920 },
              height: { min: 480, ideal: 1080 },
            }
          : {
              facingMode: "user",
              width: { min: 640, ideal: 1920 },
              height: { min: 480, ideal: 1080 },
            };

        const stream = await navigator.mediaDevices.getUserMedia({
          video: videoConstraints,
        });
        if (!isMounted) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        activeStream = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          const syncVideoDimensions = () => {
            if (videoRef.current) {
              const v = videoRef.current;
              if (v.videoWidth > 0 && v.width !== v.videoWidth) v.width = v.videoWidth;
              if (v.videoHeight > 0 && v.height !== v.videoHeight) v.height = v.videoHeight;
            }
          };
          videoRef.current.onloadedmetadata = syncVideoDimensions;
          videoRef.current.onresize = syncVideoDimensions;
          videoRef.current.onplaying = syncVideoDimensions;
          setCameraActive(true);
        }
      } catch (err) {
        console.error("Error accessing webcam:", err);
      }
    };

    startCamera();

    return () => {
      isMounted = false;
      if (activeStream) {
        activeStream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  // Load PoseNet model
  useEffect(() => {
    let isMounted = true;

    const checkAndLoad = setInterval(async () => {
      const globalWindow = window as unknown as {
        posenet?: {
          load(config?: {
            architecture?: string;
            outputStride?: number;
            inputResolution?: { width: number; height: number } | number;
            multiplier?: number;
            quantBytes?: number;
          }): Promise<PoseNetModel>;
        };
      };

      if (globalWindow.posenet) {
        clearInterval(checkAndLoad);
        try {
          console.log("[POSENET] Initializing model...");
          const model = await globalWindow.posenet.load({
            architecture: "MobileNetV1",
            outputStride: 16,
            inputResolution: { width: 257, height: 257 },
            multiplier: 0.5,
          });
          if (isMounted) {
            posenetModelRef.current = model;
            setModelLoading(false);
            console.log("[POSENET] Model loaded!");
          }
        } catch (err) {
          console.error("[POSENET] Error loading model:", err);
        }
      }
    }, 100);

    return () => {
      isMounted = false;
      clearInterval(checkAndLoad);
    };
  }, []);

  // Run PoseNet detection loop: live framing guidance, and the in-frame check the shutter reads
  useEffect(() => {
    if (!cameraActive || modelLoading || !videoRef.current) return;

    let animationFrameId: number;
    const video = videoRef.current;
    const model = posenetModelRef.current;

    const detect = async () => {
      if (model && video.readyState === video.HAVE_ENOUGH_DATA && !isCapturingRef.current) {
        try {
          // Explicitly sync video DOM width/height to intrinsic stream dimensions on every frame & reconnect
          if (video.videoWidth > 0 && video.width !== video.videoWidth) {
            video.width = video.videoWidth;
          }
          if (video.videoHeight > 0 && video.height !== video.videoHeight) {
            video.height = video.videoHeight;
          }

          frameCountRef.current += 1;
          const now = Date.now();
          if (now - lastFpsCalcRef.current >= 1000) {
            fpsRef.current = frameCountRef.current;
            frameCountRef.current = 0;
            lastFpsCalcRef.current = now;
          }

          const pose = await model.estimateSinglePose(video, {
            flipHorizontal: false
          });

          const frameWidth = video.videoWidth || video.width || 640;
          const frameHeight = video.videoHeight || video.height || 480;
          const { status, diag } = computeFraming(pose, frameWidth, frameHeight);

          framingStatusRef.current = status;
          setFramingStatus(status);

          // Throttled console logging (every 1 second) for live diagnosis
          if (now - lastLogTimeRef.current >= 1000) {
            lastLogTimeRef.current = now;
            console.log("[CAPTURE DIAGNOSTIC]", {
              res: `${frameWidth}x${frameHeight}`,
              fps: fpsRef.current,
              L_score: Number(diag.leftShoulderScore.toFixed(2)),
              R_score: Number(diag.rightShoulderScore.toFixed(2)),
              xBounds: `${(diag.minXRatio * 100).toFixed(1)}% - ${(diag.maxXRatio * 100).toFixed(1)}% (Pass: ${diag.isHorizontallyFramed})`,
              yBounds: `${(diag.minYRatio * 100).toFixed(1)}% - ${(diag.maxYRatio * 100).toFixed(1)}% (Pass: ${diag.isVerticallyFramed})`,
              failReason: diag.failReason,
              shoulderWidthPx: diag.shoulderWidthPx ? Math.round(diag.shoulderWidthPx) : null,
              status,
            });
          }
        } catch (err) {
          console.error("[POSENET] Estimate error:", err);
        }
      }
      animationFrameId = requestAnimationFrame(detect);
    };

    detect();

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [cameraActive, modelLoading]);


  const grabFrame = useCallback((): string | null => {
    if (!videoRef.current || !canvasRef.current) return null;
    const video = videoRef.current;
    const canvas = canvasRef.current;

    const videoRatio = video.videoWidth / video.videoHeight;
    const elementRatio = video.clientWidth / video.clientHeight;

    let drawWidth = video.videoWidth;
    let drawHeight = video.videoHeight;
    let offsetX = 0;
    let offsetY = 0;

    if (videoRatio > elementRatio) {
      drawWidth = video.videoHeight * elementRatio;
      offsetX = (video.videoWidth - drawWidth) / 2;
    } else {
      drawHeight = video.videoWidth / elementRatio;
      offsetY = (video.videoHeight - drawHeight) / 2;
    }

    canvas.width = drawWidth;
    canvas.height = drawHeight;

    const ctx = canvas.getContext("2d");
    if (!ctx) return null;

    ctx.drawImage(video, offsetX, offsetY, drawWidth, drawHeight, 0, 0, drawWidth, drawHeight);
    return canvas.toDataURL("image/png");
  }, []);

  // Shutter at the end of the countdown. Cancels unless the visitor is facing the camera and centred in the frame.
  const capturePhoto = useCallback(async () => {
    if (burstInProgressRef.current || isCapturingRef.current) return;

    if (framingStatusRef.current !== "framed") {
      console.warn("[CAPTURE CANCELLED] Visitor not in frame:", framingStatusRef.current);
      setRetakeNotice({ message: NOT_SEEN_MESSAGE, hint: null });
      return;
    }

    burstInProgressRef.current = true;
    setIsCapturing(true);

    // Capture photo + hold-still frames; abort if the visitor moves out of frame (or the page unmounts) mid-burst
    const frames: string[] = [];
    for (let i = 0; i <= HOLD_FRAME_COUNT; i++) {
      if (i > 0) await new Promise((resolve) => setTimeout(resolve, HOLD_FRAME_INTERVAL_MS));
      const frame = framingStatusRef.current === "framed" ? grabFrame() : null;
      if (!frame) {
        console.warn("[CAPTURE ABORTED] Visitor moved out of frame during capture");
        setRetakeNotice({ message: NOT_SEEN_MESSAGE, hint: null });
        burstInProgressRef.current = false;
        setIsCapturing(false);
        return;
      }
      frames.push(frame);
    }

    isCapturingRef.current = true;
    const [image, ...holdFrames] = frames;

    // Pre-flight checks before leaving this screen, so bad captures are instantly rejected and can be retaken here
    const preflightProblem = await runCapturePreflight(image);
    if (!videoRef.current) return; // Page left (e.g. Home) while the check ran
    if (preflightProblem) {
      console.warn("[CAPTURE REJECTED] Pre-flight failed:", preflightProblem);
      setRetakeNotice({ message: preflightProblem, hint: RETAKE_HINT });
      isCapturingRef.current = false;
      burstInProgressRef.current = false;
      setIsCapturing(false);
      return;
    }

    setCapturedImage(image);
    // New photo (including Retake): drop the previous capture's measurements and any visitor correction
    setBodyMeasurements(null);
    setMeasurementScale(1);

    // Trigger body profile extraction in the background
    fetch(`${API_BASE_URL}/api/extract-profile`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image,
        hold_frames: holdFrames,
        session_id: sessionId,
      }),
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.success && data.body_profile) {
          console.log("[BODY PROFILE EXTRACTED]", data.body_profile);
          setBodyProfile(data.body_profile);
        }
        if (data.success && data.body_measurements) {
          setBodyMeasurements(data.body_measurements);
        }
      })
      .catch((err) => {
        console.warn("[BODY PROFILE EXTRACTION FAILED]", err);
      });

    onCapture(); // Navigate to confirmation
  }, [grabFrame, sessionId, setCapturedImage, setBodyProfile, setBodyMeasurements, setMeasurementScale, onCapture]);

  // Countdown fires the latest capture handler without restarting its timer on handler identity changes
  useEffect(() => {
    capturePhotoRef.current = () => {
      void capturePhoto();
    };
  }, [capturePhoto]);

  const startCountdown = useCallback(() => {
    if (burstInProgressRef.current || isCapturingRef.current) return;
    setRetakeNotice(null);
    setCountdown((current) => current ?? COUNTDOWN_FROM);
  }, []);

  // One step per second: 5, 4, 3, 2, 1, then the photo. Leaving the page (Home) clears the pending step.
  useEffect(() => {
    if (countdown === null) return;
    const timer = setTimeout(() => {
      if (countdown > 1) {
        setCountdown(countdown - 1);
      } else {
        setCountdown(null);
        capturePhotoRef.current();
      }
    }, COUNTDOWN_STEP_MS);
    return () => clearTimeout(timer);
  }, [countdown]);

  const countdownActive = countdown !== null;
  const captureBusy = countdownActive || isCapturing;

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
      {/* Hidden canvas for image capture */}
      <canvas ref={canvasRef} style={{ display: "none" }} />

      {/* Frame 62 (Figma layout box, scaled using vh units) */}
      <div
        style={{
          position: "absolute",
          width: "49.5vh", // 786px / 1587px * 100vh = 49.53vh
          height: "auto",
          left: "calc(50% - 24.75vh)", // Centered horizontally
          top: "10.5vh", // Adjusted slightly upward for optimal spacing
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: "2.0vh",
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
            width: "100%",
            height: "1.83vh",
          }}
        >
          {/* Left purple Line */}
          <div
            style={{
              boxSizing: "border-box",
              width: "2.5vh",
              height: "0px",
              border: "0.25vh solid #5C4A3D",
            }}
          />

          {/* STAND IN THE HIGHLIGHTED AREA */}
          <span
            style={{
              fontFamily: "'Montserrat', sans-serif",
              fontWeight: 700,
              fontSize: "1.5vh", // 24px
              lineHeight: "1.83vh",
              textTransform: "uppercase",
              color: "#5C4A3D",
              textAlign: "center",
              whiteSpace: "nowrap",
            }}
          >
            STAND IN THE HIGHLIGHTED AREA
          </span>

          {/* Right purple Line */}
          <div
            style={{
              boxSizing: "border-box",
              width: "2.5vh",
              height: "0px",
              border: "0.25vh solid #5C4A3D",
            }}
          />
        </div>

        {/* Frame 58 */}
        <div
          style={{
            display: "flex",
            flexDirection: "row",
            alignItems: "center",
            justifyContent: "center",
            gap: "1.26vh",
            width: "100%",
            height: "55.32vh", // ~878px scaled to vh
            position: "relative",
          }}
        >

          {/* Camera wrapper — allows home button to overflow the clipped camera box */}
          <div
            style={{
              position: "relative",
              width: "47.07vh",
              height: "100%",
            }}
          >
            {/* Glowing Camera Outline Box */}
            <div
              style={{
                boxSizing: "border-box",
                width: "100%",
                height: "100%",
                background: "rgba(255, 255, 255, 0.05)",
                border: "0.126vh solid rgba(255, 255, 255, 0.3)",
                boxShadow: "0px 1.0vh 4.03vh rgba(0, 0, 0, 0.37), inset -2px -2px 20px #D4C7BC, inset 2px 2px 20px #B9AEA4",
                backdropFilter: "blur(12px)",
                WebkitBackdropFilter: "blur(12px)",
                borderRadius: "4.03vh",
                position: "relative",
                overflow: "hidden",
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
              }}
            >
              {/* Video Feed */}
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                onLoadedMetadata={(e) => {
                  const v = e.currentTarget;
                  if (v.videoWidth > 0 && v.width !== v.videoWidth) v.width = v.videoWidth;
                  if (v.videoHeight > 0 && v.height !== v.videoHeight) v.height = v.videoHeight;
                }}
                onResize={(e) => {
                  const v = e.currentTarget;
                  if (v.videoWidth > 0 && v.width !== v.videoWidth) v.width = v.videoWidth;
                  if (v.videoHeight > 0 && v.height !== v.videoHeight) v.height = v.videoHeight;
                }}
                onPlay={(e) => {
                  const v = e.currentTarget;
                  if (v.videoWidth > 0 && v.width !== v.videoWidth) v.width = v.videoWidth;
                  if (v.videoHeight > 0 && v.height !== v.videoHeight) v.height = v.videoHeight;
                }}
                onPlaying={(e) => {
                  const v = e.currentTarget;
                  if (v.videoWidth > 0 && v.width !== v.videoWidth) v.width = v.videoWidth;
                  if (v.videoHeight > 0 && v.height !== v.videoHeight) v.height = v.videoHeight;
                }}
                style={{
                  width: "100%",
                  height: "100%",
                  objectFit: "cover",
                  transform: "scaleX(-1)",
                }}
              />

              {/* Initializing Camera Overlay */}
              {!cameraActive && (
                <div
                  style={{
                    position: "absolute",
                    color: "#5C4A3D",
                    fontFamily: "'Poppins', sans-serif",
                    fontSize: "1.26vh",
                    fontWeight: 400,
                    lineHeight: "1.89vh",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    textAlign: "center",
                    letterSpacing: "0.5vh",
                    width: "24.0vh",
                    height: "1.89vh",
                    zIndex: 2,
                  }}
                >
                  INITIALIZING CAMERA...
                </div>
              )}

              {/* Framing Guidance HUD Badge (Top Center of camera frame) */}
              {cameraActive && !modelLoading && (
                <div
                  style={{
                    position: "absolute",
                    top: "1.8vh",
                    left: "50%",
                    transform: "translateX(-50%)",
                    zIndex: 8,
                    display: "flex",
                    alignItems: "center",
                    gap: "0.8vh",
                    padding: "0.6vh 1.6vh",
                    borderRadius: "9999px",
                    backdropFilter: "blur(12px)",
                    WebkitBackdropFilter: "blur(12px)",
                    background:
                      framingStatus === "framed"
                        ? "rgba(16, 185, 129, 0.22)"
                        : framingStatus === "none"
                        ? "rgba(0, 0, 0, 0.45)"
                        : "rgba(234, 179, 8, 0.25)",
                    border:
                      framingStatus === "framed"
                        ? "0.15vh solid rgba(16, 185, 129, 0.7)"
                        : framingStatus === "none"
                        ? "0.15vh solid rgba(255, 255, 255, 0.2)"
                        : "0.15vh solid rgba(234, 179, 8, 0.7)",
                    boxShadow:
                      framingStatus === "framed"
                        ? "0px 0.4vh 1.6vh rgba(16, 185, 129, 0.3)"
                        : "0px 0.4vh 1.2vh rgba(0, 0, 0, 0.2)",
                    transition: "all 0.25s ease-out",
                    whiteSpace: "nowrap",
                    pointerEvents: "none",
                  }}
                >
                  {/* Status Indicator Dot */}
                  <span
                    style={{
                      display: "inline-block",
                      width: "0.9vh",
                      height: "0.9vh",
                      borderRadius: "50%",
                      backgroundColor:
                        framingStatus === "framed" ? "#10B981" : framingStatus === "none" ? "#9CA3AF" : "#EAB308",
                      boxShadow: framingStatus === "framed" ? "0 0 0.8vh #10B981" : "none",
                    }}
                  />

                  {/* Framing Readout */}
                  <span
                    style={{
                      fontFamily: "'Montserrat', sans-serif",
                      fontWeight: 700,
                      fontSize: "1.25vh",
                      color:
                        framingStatus === "framed" ? "#10B981" : framingStatus === "none" ? "#E5E7EB" : "#FEF08A",
                      letterSpacing: "0.04vh",
                    }}
                  >
                    {FRAMING_PROMPTS[framingStatus]}
                  </span>
                </div>
              )}

              {/* Viewfinder Shoulder Alignment Guidelines */}
              {cameraActive && !modelLoading && (
                <div
                  style={{
                    position: "absolute",
                    top: "16%",
                    bottom: "22%",
                    left: "15%",
                    right: "15%",
                    border: `0.18vh dashed ${
                      framingStatus === "framed"
                        ? "rgba(21, 128, 61, 0.85)"
                        : framingStatus === "face_camera" || framingStatus === "bad_framing"
                        ? "rgba(234, 179, 8, 0.7)"
                        : "rgba(60, 46, 36, 0.55)"
                    }`,
                    borderRadius: "2.5vh",
                    pointerEvents: "none",
                    zIndex: 2,
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "space-between",
                    alignItems: "center",
                    opacity: isCapturing ? 0.35 : 0.85,
                    transition: "all 0.3s ease-in-out",
                  }}
                >
                  <span
                    style={{
                      fontFamily: "'Poppins', sans-serif",
                      fontSize: "0.95vh",
                      fontWeight: 700,
                      color:
                        framingStatus === "framed"
                          ? "#14532D"
                          : "#3D2E24",
                      marginTop: "-1.4vh",
                      letterSpacing: "0.08vh",
                      textTransform: "uppercase",
                      background:
                        framingStatus === "framed"
                          ? "rgba(240, 253, 244, 0.85)"
                          : "rgba(255, 255, 255, 0.85)",
                      padding: "0.15vh 0.8vh",
                      borderRadius: "0.4vh",
                      backdropFilter: "blur(4px)",
                      WebkitBackdropFilter: "blur(4px)",
                    }}
                  >
                    Shoulders & Torso Zone
                  </span>
                  <span
                    style={{
                      fontFamily: "'Poppins', sans-serif",
                      fontSize: "0.9vh",
                      fontWeight: 600,
                      color:
                        framingStatus === "framed"
                          ? "#14532D"
                          : "#4B5563",
                      marginBottom: "-1.4vh",
                      letterSpacing: "0.06vh",
                      textTransform: "uppercase",
                      background:
                        framingStatus === "framed"
                          ? "rgba(240, 253, 244, 0.85)"
                          : "rgba(255, 255, 255, 0.85)",
                      padding: "0.15vh 0.8vh",
                      borderRadius: "0.4vh",
                      backdropFilter: "blur(4px)",
                      WebkitBackdropFilter: "blur(4px)",
                    }}
                  >
                    {framingStatus === "framed"
                      ? "Position Ready ✓"
                      : "Keep Shoulders Inside Frame"}
                  </span>
                </div>
              )}

              {/* User Detection / Loading Model Status */}
              {cameraActive && modelLoading && (
                <div
                  style={{
                    position: "absolute",
                    color: "#C2A285",
                    fontFamily: "'Poppins', sans-serif",
                    fontSize: "1.3vh",
                    fontWeight: 600,
                    zIndex: 4,
                    background: "rgba(0, 0, 0, 0.6)",
                    padding: "0.8vh 1.5vh",
                    borderRadius: "1.5vh",
                    textAlign: "center",
                  }}
                >
                  Setting up auto-detect...
                </div>
              )}

              {/* Capture Overlay (brief hold-still while the photo is taken) */}
              {isCapturing && (
                <div
                  style={{
                    position: "absolute",
                    zIndex: 10,
                    fontFamily: "'Montserrat', sans-serif",
                    fontWeight: 700,
                    fontSize: "1.2vh",
                    letterSpacing: "0.12vh",
                    textTransform: "uppercase",
                    color: "#5C4A3D",
                    background: "rgba(243, 229, 216, 0.88)",
                    padding: "0.4vh 1.2vh",
                    borderRadius: "0.8vh",
                    backdropFilter: "blur(6px)",
                    pointerEvents: "none",
                  }}
                >
                  Hold still
                </div>
              )}

              {/* Countdown 5 → 1 between Start Capture and the photo (the key replays the pop-in for each number) */}
              {countdownActive && (
                <div
                  key={countdown}
                  style={{
                    position: "absolute",
                    zIndex: 12,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    width: "18vh",
                    height: "18vh",
                    borderRadius: "50%",
                    background: "rgba(28, 18, 14, 0.55)",
                    border: "0.3vh solid rgba(243, 229, 216, 0.85)",
                    boxShadow: "0 0 3vh rgba(196, 164, 132, 0.5)",
                    backdropFilter: "blur(6px)",
                    WebkitBackdropFilter: "blur(6px)",
                    fontFamily: "'Montserrat', sans-serif",
                    fontWeight: 800,
                    fontSize: "11vh",
                    lineHeight: 1,
                    color: "#F3E5D8",
                    animation: "pulseScale 0.4s ease-out",
                    pointerEvents: "none",
                  }}
                >
                  {countdown}
                </div>
              )}

              {/* Why the last attempt gave no photo (not in frame, or face not visible). Sits over the torso so the
                  visitor can still see their own face in the preview while fixing it. */}
              {cameraActive && !modelLoading && !captureBusy && retakeNotice && (
                <div
                  style={{
                    position: "absolute",
                    top: "64%",
                    left: "50%",
                    transform: "translate(-50%, -50%)",
                    zIndex: 14,
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    gap: "0.8vh",
                    background: "rgba(28, 18, 14, 0.94)",
                    border: "0.2vh solid #EF4444",
                    boxShadow: "0 0 3vh rgba(239, 68, 68, 0.5)",
                    borderRadius: "1.8vh",
                    padding: "1.8vh 2.6vh",
                    backdropFilter: "blur(10px)",
                    textAlign: "center",
                    maxWidth: "85%",
                    animation: "pulseScale 0.3s ease-out",
                  }}
                >
                  <div
                    style={{
                      fontFamily: "'Montserrat', sans-serif",
                      color: "#FFFFFF",
                      fontSize: "1.25vh",
                      fontWeight: 600,
                      lineHeight: 1.4,
                    }}
                  >
                    ⚠️ {retakeNotice.message}
                  </div>
                  {retakeNotice.hint && (
                    <div
                      style={{
                        fontFamily: "'Poppins', sans-serif",
                        color: "rgba(255, 255, 255, 0.7)",
                        fontSize: "1.0vh",
                        fontWeight: 400,
                      }}
                    >
                      {retakeNotice.hint}
                    </div>
                  )}
                </div>
              )}

              {/* Corner Brackets */}
              {/* Top-Left Corner */}
              <div
                style={{
                  boxSizing: "border-box",
                  position: "absolute",
                  width: "5.04vh",
                  height: "5.04vh",
                  left: "-2px",
                  top: "-2px",
                  borderWidth: "0.38vh 0px 0px 0.38vh",
                  borderStyle: "solid",
                  borderColor: "#F3E5D8",
                  filter: "drop-shadow(0px 0px 1.89vh rgba(196, 164, 132, 0.5))",
                  borderTopLeftRadius: "4.03vh",
                  pointerEvents: "none",
                  zIndex: 3,
                }}
              />
              {/* Top-Right Corner */}
              <div
                style={{
                  boxSizing: "border-box",
                  position: "absolute",
                  width: "5.04vh",
                  height: "5.04vh",
                  right: "-2px",
                  top: "-2px",
                  borderWidth: "0.38vh 0.38vh 0px 0px",
                  borderStyle: "solid",
                  borderColor: "#F3E5D8",
                  filter: "drop-shadow(0px 0px 1.89vh rgba(196, 164, 132, 0.5))",
                  borderTopRightRadius: "4.03vh",
                  pointerEvents: "none",
                  zIndex: 3,
                }}
              />
              {/* Bottom-Left Corner */}
              <div
                style={{
                  boxSizing: "border-box",
                  position: "absolute",
                  width: "5.04vh",
                  height: "5.04vh",
                  left: "-2px",
                  bottom: "-2px",
                  borderWidth: "0px 0px 0.38vh 0.38vh",
                  borderStyle: "solid",
                  borderColor: "#F3E5D8",
                  filter: "drop-shadow(0px 0px 1.89vh rgba(196, 164, 132, 0.5))",
                  borderBottomLeftRadius: "4.03vh",
                  pointerEvents: "none",
                  zIndex: 3,
                }}
              />
              {/* Bottom-Right Corner */}
              <div
                style={{
                  boxSizing: "border-box",
                  position: "absolute",
                  width: "5.04vh",
                  height: "5.04vh",
                  right: "-2px",
                  bottom: "-2px",
                  borderWidth: "0px 0.38vh 0.38vh 0px",
                  borderStyle: "solid",
                  borderColor: "#F3E5D8",
                  filter: "drop-shadow(0px 0px 1.89vh rgba(196, 164, 132, 0.5))",
                  borderBottomRightRadius: "4.03vh",
                  pointerEvents: "none",
                  zIndex: 3,
                }}
              />
            </div>

            {/* Home button — centered on bottom border, outside the clipped camera box */}
            <NavigationButton
              type="home"
              onClick={onHome}
              style={{
                position: "absolute",
                bottom: "-3.52vh",
                left: "45%",
                transform: "translateX(-50%)",
                width: "7.05vh",
                height: "7.05vh",
                zIndex: 10,
              }}
            />
          </div>

        </div>

        {/* Frame 63 */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: "2.0vh",
            width: "100%",
            height: "auto",
            marginTop: "1.5vh", // Push text and button slightly down to clear overlapping home button
          }}
        >
          {/* Instruction line: what to do before Start Capture, then live framing guidance until the photo */}
          <span
            style={{
              width: "100%",
              height: "3.15vh",
              fontFamily: "'Montserrat', sans-serif",
              fontWeight: 700,
              fontSize: "1.8vh",
              lineHeight: "3.15vh",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              textAlign: "center",
              letterSpacing: "0.2vh",
              textTransform: "uppercase",
              color: !captureBusy || framingStatus === "none"
                ? "#5C4A3D"
                : framingStatus === "framed"
                ? "#15803D"
                : "#D97706",
              marginTop: "1vh",
              whiteSpace: "nowrap",
              transition: "color 0.25s ease-out",
            }}
          >
            {!captureBusy
              ? "Tap start, then step into view"
              : framingStatus === "framed"
              ? "Hold still"
              : FRAMING_PROMPTS[framingStatus]}
          </span>

          <PrimaryButton
            onClick={startCountdown}
            disabled={modelLoading || captureBusy}
            style={{
              width: "100%",
              height: "5.8vh",
              background: captureBusy
                ? "linear-gradient(135deg, #3D2E24, #1A130E)"
                : "linear-gradient(135deg, #3D2E24, #241B15)",
              color: "#FFFFFF",
              boxShadow: captureBusy
                ? "0 0 2.5vh rgba(194, 162, 133, 0.45)"
                : "0px 1.0vh 3.0vh -0.5vh rgba(43, 32, 24, 0.4)",
              borderRadius: "14398.6px",
              flex: "none",
              order: 0,
              flexGrow: 0,
              zIndex: 0,
              pointerEvents: modelLoading || captureBusy ? "none" : undefined,
              transition: "all 0.3s ease-in-out",
              border: "0.15vh solid rgba(219, 184, 148, 0.35)",
            }}
          >
            {modelLoading
              ? "LOADING DETECTOR..."
              : isCapturing
              ? "HOLD STILL — CAPTURING..."
              : countdownActive
              ? `TAKING PHOTO IN ${countdown}...`
              : "START CAPTURE"}
          </PrimaryButton>

        </div>
      </div>
    </div>
  );
}

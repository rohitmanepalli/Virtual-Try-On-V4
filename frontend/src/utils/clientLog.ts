// Sends what happens on the kiosk (screens, button taps, console output, errors) to the backend's logs/app_<date>.log.
// Logging must never affect the visitor: failures are silent, and nothing here calls the wrapped console methods.
import { API_BASE_URL } from "./api";

type Entry = { level: string; message: string; time: string };

const FLUSH_INTERVAL_MS = 3000;
const MAX_BUFFERED = 500;
// A keepalive request (needed to send during page close) is limited to 64KB: 50 x 1000 chars stays under it
const MAX_BATCH = 50;
const MAX_MESSAGE_CHARS = 1000;

const clientId = Math.random().toString(36).slice(2, 10);
let buffer: Entry[] = [];
let sending = false;

function describe(arg: unknown): string {
  if (typeof arg === "string") return arg;
  if (arg instanceof Error) return arg.stack || `${arg.name}: ${arg.message}`;
  try {
    return JSON.stringify(arg) ?? String(arg);
  } catch {
    return String(arg);
  }
}

export function logEvent(message: string, level = "info"): void {
  buffer.push({ level, message: message.slice(0, MAX_MESSAGE_CHARS), time: new Date().toTimeString().slice(0, 8) });
  if (buffer.length > MAX_BUFFERED) buffer = buffer.slice(-MAX_BUFFERED);
}

async function flush(): Promise<void> {
  if (sending || buffer.length === 0) return;
  sending = true;
  const batch = buffer.splice(0, MAX_BATCH);
  try {
    await fetch(`${API_BASE_URL}/api/client-logs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ client_id: clientId, entries: batch }),
      keepalive: true,
    });
  } catch {
    // Backend unreachable: keep the entries for the next attempt
    buffer = [...batch, ...buffer].slice(-MAX_BUFFERED);
  } finally {
    sending = false;
  }
}

function tappedLabel(target: EventTarget | null): string | null {
  const control = target instanceof Element ? target.closest("button, a, [role='button']") : null;
  if (!control) return null;
  return (control.getAttribute("aria-label") || control.textContent || control.tagName).trim().slice(0, 60);
}

export function initClientLog(): void {
  for (const level of ["debug", "log", "info", "warn", "error"] as const) {
    const original = console[level].bind(console);
    console[level] = (...args: unknown[]) => {
      original(...args);
      logEvent(args.map(describe).join(" "), level);
    };
  }

  window.addEventListener("error", (e) => logEvent(`Uncaught ${e.message} (${e.filename}:${e.lineno})`, "error"));
  window.addEventListener("unhandledrejection", (e) => logEvent(`Unhandled rejection: ${describe(e.reason)}`, "error"));
  document.addEventListener(
    "click",
    (e) => {
      const label = tappedLabel(e.target);
      if (label !== null) logEvent(`tap: ${label}`);
    },
    true
  );
  window.addEventListener("pagehide", () => void flush());
  window.setInterval(() => void flush(), FLUSH_INTERVAL_MS);

  logEvent(`app opened: ${navigator.userAgent}`);
}

// The one place the backend URL is decided; every file imports API_BASE_URL from here.
// Set it with VITE_API_BASE_URL in .env (dev + Electron) / .env.production (npm run build). Leave it blank and a
// page served by the backend talks to its own origin; the Vite dev server and Electron (file://) use the default.
const DEFAULT_API_BASE_URL = "http://34.180.17.35:8888";

export const getApiBaseUrl = (): string => {
  const envUrl = import.meta.env.VITE_API_BASE_URL?.trim();
  if (envUrl) {
    return envUrl.replace(/\/+$/, "");
  }
  if (window.location.protocol.startsWith("http") && window.location.port !== "5173") {
    return window.location.origin;
  }
  return DEFAULT_API_BASE_URL;
};

export const API_BASE_URL = getApiBaseUrl();

import type { SizeFit } from "../context/TrialRoomContext";

export const FIT_LABELS: Record<string, string> = {
  oversized: "Very oversized",
  loose: "Loose fit",
  relaxed: "Relaxed fit",
  perfect: "Correct fit",
  snug: "Snug fit",
  tight: "Tight fit",
  does_not_fit: "Won't fit",
};

/**
 * Fit category of a size for this visitor: the backend's body-checked value when known, otherwise from the offset
 * alone (+1 relaxed, +2 loose, +3 oversized, -1 snug, -2 tight, -3 or lower won't fit).
 */
export function getSizeQuality(
  size: string,
  recommendedSize: string | null,
  allSizes: string[],
  sizeFits?: Record<string, SizeFit>
): string {
  const known = sizeFits?.[size]?.fit_quality;
  if (known) return known;
  if (!recommendedSize) return "unknown";
  const recIdx = allSizes.indexOf(recommendedSize);
  const selIdx = allSizes.indexOf(size);
  if (recIdx === -1 || selIdx === -1) return "unknown";
  const diff = selIdx - recIdx;
  if (diff === 0) return "perfect";
  if (diff === -1) return "snug";
  if (diff === -2) return "tight";
  if (diff <= -3) return "does_not_fit";
  if (diff === 1) return "relaxed";
  if (diff === 2) return "loose";
  return "oversized";
}

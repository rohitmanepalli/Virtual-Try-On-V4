"""
Limb check: a render must show every limb the visitor has, whatever the garment or size.

The model redraws the whole frame (there is no mask), so a hand can vanish into an over-long sleeve or a dupatta, and
legs or feet can be dropped when the capture is cropped above them. Pose landmarks on the capture and the render
decide: a hand seen in the capture must be in the render, and the full-length render must show both legs and feet.
"""
from typing import Any, Dict, List, Optional

HAND_SEEN = 0.5  # MediaPipe visibility at which a capture's wrist counts as visible
LIMB_PRESENT = 0.3  # lenient on the render: a cuff over the wrist or a hem on the foot lowers visibility
# ponytail: category list, not a measurement; floor-length outfits can hide knees and feet by design
FLOOR_LENGTH = ("saree", "lehenga", "gown", "anarkali")


def check_limbs(capture_lm: Any, render_lm: Any, category: Optional[str] = None) -> Dict[str, Any]:
    """passed True/False when the render's pose was measurable, None otherwise (never a failure)."""
    render_vis = getattr(render_lm, "visibility", None) or {}
    if not render_vis:
        return {"checked": False, "passed": None, "missing": []}
    capture_vis = getattr(capture_lm, "visibility", None) or {}
    floor_length = (category or "").lower() in FLOOR_LENGTH
    missing: List[str] = []
    for side in ("left", "right"):
        if capture_vis.get(f"{side}_wrist", 0.0) >= HAND_SEEN and render_vis.get(f"{side}_wrist", 0.0) < LIMB_PRESENT:
            missing.append(f"{side} hand")
        if not floor_length:
            if render_vis.get(f"{side}_knee", 0.0) < LIMB_PRESENT:
                missing.append(f"{side} leg")
            if render_vis.get(f"{side}_ankle", 0.0) < LIMB_PRESENT:
                missing.append(f"{side} foot")
    return {"checked": True, "passed": not missing, "missing": missing}


def capture_leg_text(capture_lm: Any, category: Optional[str] = None) -> Optional[str]:
    """Directive for attempt 1 when the capture does not show both legs and feet; None when it does or is unmeasured."""
    vis = getattr(capture_lm, "visibility", None) or {}
    if not vis or (category or "").lower() in FLOOR_LENGTH:
        return None
    hidden = [f"{side} {part}" for part, lm in (("leg", "knee"), ("foot", "ankle")) for side in ("left", "right")
              if vis.get(f"{side}_{lm}", 0.0) < HAND_SEEN]
    if not hidden:
        return None
    return (
        "\n[LEGS NOT IN IMAGE 1 — DRAW THEM]\n"
        f"- Image 1 is cropped and does not show the person's {', '.join(hidden)}. You MUST still draw them: continue "
        "the body below the lower edge of Image 1 with natural legs and feet in proportion to the visible body "
        "(same skin tone, same build, standing straight), so the output is a complete full-length person from head to "
        "feet. Never end the image at the waist, hips or thighs.\n"
    )


def limb_escalation_text(missing: List[str]) -> str:
    """Directive for the retry after a render lost a limb."""
    return (
        "\n[LIMB RETRY — A PREVIOUS ATTEMPT WAS REJECTED]\n"
        f"- It was missing the person's {', '.join(missing)}. Show the whole person inside the frame: both arms with "
        "both hands, both legs with both feet, in the pose of Image 1. No sleeve, dupatta, hem or frame edge hides a "
        "hand or a foot.\n"
    )


if __name__ == "__main__":
    from types import SimpleNamespace as NS

    full = {f"{s}_{p}": 0.9 for s in ("left", "right") for p in ("wrist", "knee", "ankle")}
    assert check_limbs(NS(visibility=full), NS(visibility=full))["passed"] is True
    no_hand = dict(full, left_wrist=0.1)
    assert check_limbs(NS(visibility=full), NS(visibility=no_hand))["missing"] == ["left hand"]
    # A hand hidden in the capture is not required in the render
    assert check_limbs(NS(visibility=no_hand), NS(visibility=no_hand))["passed"] is True
    no_feet = dict(full, left_ankle=0.0, right_ankle=0.0)
    assert check_limbs(NS(visibility=full), NS(visibility=no_feet))["missing"] == ["left foot", "right foot"]
    assert check_limbs(NS(visibility=full), NS(visibility=no_feet), "lehenga")["passed"] is True
    assert check_limbs(NS(visibility=full), NS(visibility={}))["passed"] is None
    # Capture cropped at the thighs: attempt 1 is told to draw legs and feet
    thighs_up = dict(full, left_knee=0.0, right_knee=0.0, left_ankle=0.0, right_ankle=0.0)
    hint = capture_leg_text(NS(visibility=thighs_up))
    assert hint and "left leg" in hint and "right foot" in hint
    assert capture_leg_text(NS(visibility=full)) is None
    assert capture_leg_text(NS(visibility=thighs_up), "saree") is None
    assert capture_leg_text(NS(visibility={})) is None
    print("limb_check OK")

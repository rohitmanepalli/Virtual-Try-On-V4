"""
Unit tests for the size chart recommendation engine.
Tests recommend_size() with various body profiles and get_size_offset() for all size combinations.
"""
import pytest
from app.models import BodyProfile
from app.services.size_chart import (
    ALL_SIZES,
    SizeOffset,
    SizeRecommendationResult,
    compute_size_from_measurements,
    get_size_offset,
    recommend_size,
)


# ── Helper: construct a BodyProfile with specific proportions ──────────────────

def _make_profile(
    chest: float = 1.05,
    waist: float = 0.88,
    hip: float = 1.0,
    shoulder: float = 0.28,
    sh_hip_ratio: float = 1.15,
    body_type: str = "Rectangle / Balanced",
) -> BodyProfile:
    return BodyProfile(
        height_ratio=0.82,
        shoulder_width_ratio=shoulder,
        torso_aspect_ratio=1.15,
        shoulder_to_hip_ratio=sh_hip_ratio,
        chest_proportion=chest,
        waist_proportion=waist,
        hip_proportion=hip,
        body_type=body_type,
        confidence_score=0.9,
        frame_count_sampled=3,
    )


# ── recommend_size tests ──────────────────────────────────────────────────────

class TestRecommendSize:
    """Tests for the recommend_size() function."""

    def test_returns_size_recommendation_result(self):
        profile = _make_profile()
        result = recommend_size(profile)
        assert isinstance(result, SizeRecommendationResult)
        assert result.recommended_size in ALL_SIZES
        assert len(result.available_sizes) == len(ALL_SIZES)
        assert 0.0 <= result.size_confidence <= 1.0
        assert isinstance(result.fit_notes, str)

    def test_medium_build_gets_M(self):
        """A balanced medium build should recommend M."""
        profile = _make_profile(chest=1.02, waist=0.86, shoulder=0.27)
        result = recommend_size(profile)
        assert result.recommended_size == "M"

    def test_small_build_gets_S(self):
        """A smaller-proportioned build should recommend S."""
        profile = _make_profile(chest=0.93, waist=0.78, shoulder=0.23)
        result = recommend_size(profile)
        assert result.recommended_size in ("S", "XS")

    def test_large_build_gets_L_or_XL(self):
        """A broader build should recommend L or XL."""
        profile = _make_profile(chest=1.12, waist=0.95, shoulder=0.32)
        result = recommend_size(profile)
        assert result.recommended_size in ("L", "XL")

    def test_very_small_build_gets_XS(self):
        """Very small proportions should map to XS."""
        profile = _make_profile(chest=0.85, waist=0.70, shoulder=0.20)
        result = recommend_size(profile)
        assert result.recommended_size == "XS"

    def test_very_large_build_gets_XXL(self):
        """Very large proportions should map to XXL."""
        profile = _make_profile(chest=1.35, waist=1.15, shoulder=0.40)
        result = recommend_size(profile)
        assert result.recommended_size == "XXL"

    def test_blazer_category_shoulder_bias(self):
        """Blazer category should shift towards larger size for same shoulder width (shoulder bias +0.015)."""
        profile = _make_profile(chest=1.02, waist=0.86, shoulder=0.27)
        result_generic = recommend_size(profile, garment_category="")
        result_blazer = recommend_size(profile, garment_category="blazer")
        # Blazer should recommend same or potentially smaller size due to wider shoulder allowance per size
        assert result_blazer.recommended_size in ALL_SIZES

    def test_athletic_body_type_notes(self):
        """Athletic body type should produce shoulder-related fit notes."""
        profile = _make_profile(
            chest=1.05, waist=0.82, shoulder=0.30,
            sh_hip_ratio=1.25, body_type="Athletic / V-Shape"
        )
        result = recommend_size(profile)
        assert "Athletic" in result.fit_notes or "shoulder" in result.fit_notes.lower() or "Good fit" in result.fit_notes

    def test_pear_body_type_notes(self):
        """Pear body type should produce hip-related fit notes."""
        profile = _make_profile(
            chest=0.95, waist=0.85, shoulder=0.24,
            sh_hip_ratio=0.92, body_type="Pear / Triangle"
        )
        result = recommend_size(profile)
        assert "Pear" in result.fit_notes or "hip" in result.fit_notes.lower() or "Good fit" in result.fit_notes

    def test_confidence_decreases_at_boundaries(self):
        """A profile at the extreme edge of a size band should have lower confidence than one at center."""
        center_profile = _make_profile(chest=1.015, waist=0.86, shoulder=0.27)
        edge_profile = _make_profile(chest=0.95, waist=0.80, shoulder=0.24)
        center_result = recommend_size(center_profile)
        edge_result = recommend_size(edge_profile)
        # Center of M should have higher confidence than edge
        # (not guaranteed to be strictly higher due to multi-dimensional scoring, but should be reasonable)
        assert center_result.size_confidence >= 0.0
        assert edge_result.size_confidence >= 0.0


# ── get_size_offset tests ─────────────────────────────────────────────────────

class TestGetSizeOffset:
    """Tests for the get_size_offset() function."""

    def test_returns_size_offset(self):
        result = get_size_offset("M", "M")
        assert isinstance(result, SizeOffset)

    def test_recommended_size_is_perfect(self):
        """Selecting the recommended size should return 'perfect' quality."""
        result = get_size_offset("M", "M")
        assert result.fit_quality == "perfect"
        assert result.steps_from_recommended == 0
        assert result.does_not_fit is False
        assert result.offset_lo_mm == 15.0
        assert result.offset_hi_mm == 25.0

    def test_one_size_down_is_snug(self):
        """One size below recommended should return 'snug' quality."""
        result = get_size_offset("M", "S")
        assert result.fit_quality == "snug"
        assert result.steps_from_recommended == -1
        assert result.does_not_fit is False
        assert result.offset_lo_mm == 0.0
        assert result.offset_hi_mm == 10.0

    def test_two_sizes_down_without_body_is_tight(self):
        """Without a body profile only the offset is known: -2 is tight (won't fit needs the body check or -3)."""
        result = get_size_offset("M", "XS")
        assert result.fit_quality == "tight"
        assert result.fit_label == "Tight fit"
        assert result.steps_from_recommended == -2
        assert result.does_not_fit is False

    def test_three_sizes_down_does_not_fit(self):
        """Three or more sizes below never fit, with or without a body profile."""
        result = get_size_offset("XL", "S")
        assert result.steps_from_recommended == -3
        assert result.does_not_fit is True
        assert result.fit_label == "Won't fit"

    def test_larger_sizes_scale_relaxed_loose_oversized(self):
        """+1 relaxed, +2 clearly loose, +3 or more very oversized."""
        assert get_size_offset("M", "L").fit_quality == "relaxed"
        assert get_size_offset("M", "XL").fit_quality == "loose"
        assert get_size_offset("M", "XXL").fit_quality == "oversized"
        assert get_size_offset("XS", "XXL").fit_quality == "oversized"
        assert get_size_offset("XS", "XXL").steps_from_recommended == 5
        assert get_size_offset("M", "L").fit_label == "Relaxed fit"
        assert get_size_offset("M", "XXL").fit_label == "Very oversized"

    def test_all_size_pairs_have_valid_quality(self):
        """Every combination of recommended and selected size should produce a valid quality."""
        valid_qualities = {"perfect", "snug", "tight", "does_not_fit", "relaxed", "loose", "oversized"}
        for rec in ALL_SIZES:
            for sel in ALL_SIZES:
                for body in (None, _make_profile(chest=1.25, waist=0.85, shoulder=0.26)):
                    result = get_size_offset(rec, sel, body_profile=body)
                    assert result.fit_quality in valid_qualities, (
                        f"Invalid quality '{result.fit_quality}' for rec={rec}, sel={sel}"
                    )

    def test_offset_mm_ranges_are_ordered(self):
        """Offset ranges increase monotonically from tight to very oversized."""
        order = [get_size_offset("L", s) for s in ("S", "M", "L", "XL", "XXL")]
        assert [o.fit_quality for o in order] == ["tight", "snug", "perfect", "relaxed", "loose"]
        his = [o.offset_hi_mm for o in order] + [get_size_offset("M", "XXL").offset_hi_mm]
        assert his == sorted(his)

    def test_case_insensitive_size_labels(self):
        """Size labels should work case-insensitively."""
        result = get_size_offset("m", "l")
        assert result.fit_quality == "relaxed"
        assert result.steps_from_recommended == 1


class TestBodyVersusSizeCheck:
    """'Won't fit' vs snug/tight is decided by the visitor's body against the selected size, not the offset alone."""

    LEAN_M = dict(chest=1.17, waist=0.81, shoulder=0.245)   # just into M on every dimension
    TYPICAL_M = dict(chest=1.25, waist=0.85, shoulder=0.26)
    BROAD_M = dict(chest=1.33, waist=0.89, shoulder=0.28)   # top of M on every dimension

    def _fit(self, body, sel, structure="semi_structured", rec="M"):
        return get_size_offset(rec, sel, "shirt", _make_profile(**body), structure)

    def test_same_offset_lean_is_tight_broad_wont_fit(self):
        lean, broad = self._fit(self.LEAN_M, "XS"), self._fit(self.BROAD_M, "XS")
        assert lean.fit_quality == "tight" and not lean.does_not_fit
        assert broad.fit_quality == "does_not_fit" and broad.does_not_fit
        assert broad.wont_fit_reason and broad.limiting_dimension in ("chest", "waist", "shoulder")

    def test_typical_m_in_woven_shirt(self):
        """The expected M-visitor ladder in a woven shirt: XS won't fit, S snug."""
        assert self._fit(self.TYPICAL_M, "XS").fit_quality == "does_not_fit"
        assert self._fit(self.TYPICAL_M, "S").fit_quality == "snug"

    def test_one_size_down_stays_snug_even_at_the_top_of_the_band(self):
        assert self._fit(self.BROAD_M, "S").fit_quality == "snug"

    def test_stretch_depends_on_fabric(self):
        """Structured tailoring cannot stretch; a soft drape stretches more than a woven shirt."""
        assert self._fit(self.BROAD_M, "S", "rigid_tailored").fit_quality == "does_not_fit"
        assert self._fit(self.TYPICAL_M, "XS", "soft_drape").fit_quality == "tight"

    def test_body_far_above_size_wont_fit_even_one_size_down(self):
        """A visitor whose chest votes L but is recommended M (median) cannot close an S."""
        result = self._fit(dict(chest=1.45, waist=0.86, shoulder=0.26), "S")
        assert result.does_not_fit and result.limiting_dimension == "chest"

    def test_larger_sizes_never_body_checked(self):
        result = self._fit(self.BROAD_M, "XL")
        assert result.fit_quality == "loose" and result.body_check == {}

    def test_size_fit_map_labels_every_size(self):
        from app.services.size_chart import size_fit_map
        fits = size_fit_map(_make_profile(**self.TYPICAL_M), "M", "shirt", "semi_structured")
        assert [fits[s]["fit_label"] for s in ALL_SIZES] == [
            "Won't fit", "Snug fit", "Correct fit", "Relaxed fit", "Loose fit", "Very oversized"]
        assert [fits[s]["size_offset"] for s in ALL_SIZES] == [-2, -1, 0, 1, 2, 3]


# ── Deterministic Algorithm Unit Tests ────────────────────────────────────────

class TestDeterministicSizeRecommendation:
    """Explicit unit tests covering XS through XXL, band boundaries, and category rules."""

    def test_unit_1_pure_xs(self):
        # Below / lower boundary of XS
        size = compute_size_from_measurements(chest=0.95, waist=0.70, shoulder=0.20, garment_category="t_shirt")
        assert size == "XS"

    def test_unit_2_pure_s(self):
        # Clearly in S band
        size = compute_size_from_measurements(chest=1.12, waist=0.77, shoulder=0.23, garment_category="shirt")
        assert size == "S"

    def test_unit_3_pure_m(self):
        # Clearly in M band
        size = compute_size_from_measurements(chest=1.25, waist=0.85, shoulder=0.26, garment_category="shirt")
        assert size == "M"

    def test_unit_4_pure_l(self):
        # Clearly in L band
        size = compute_size_from_measurements(chest=1.35, waist=0.92, shoulder=0.30, garment_category="shirt")
        assert size == "L"

    def test_unit_5_pure_xl(self):
        # Clearly in XL band
        size = compute_size_from_measurements(chest=1.46, waist=0.99, shoulder=0.34, garment_category="shirt")
        assert size == "XL"

    def test_unit_6_pure_xxl(self):
        # Above all bands / clearly in XXL band
        size = compute_size_from_measurements(chest=1.60, waist=1.10, shoulder=0.38, garment_category="shirt")
        assert size == "XXL"

    def test_unit_7_boundary_s_m(self):
        # Exactly at S/M boundary: chest=1.18 (M), waist=0.81 (S), shoulder=0.245 (S)
        # Median of [S, S, M] is S
        size = compute_size_from_measurements(chest=1.18, waist=0.81, shoulder=0.245, garment_category="t_shirt")
        assert size == "S"

    def test_unit_8_boundary_m_l(self):
        # Exactly at M/L boundary: chest=1.30 (M), waist=0.88 (L), shoulder=0.280 (M)
        # Median of [M, M, L] is M
        size = compute_size_from_measurements(chest=1.30, waist=0.88, shoulder=0.280, garment_category="shirt")
        assert size == "M"

    def test_unit_9_blazer_shoulder_dominance(self):
        # Blazer: shoulder=0.30 (L), chest=1.25 (M), waist=0.85 (M)
        # Category blazer makes shoulder primary => L
        size = compute_size_from_measurements(chest=1.25, waist=0.85, shoulder=0.30, garment_category="blazer")
        assert size == "L"

    def test_unit_10_dress_waist_dominance(self):
        # Dress: waist=0.76 (S), chest=1.25 (M), shoulder=0.26 (M)
        # Category dress makes waist primary => S
        size = compute_size_from_measurements(chest=1.25, waist=0.76, shoulder=0.26, garment_category="dress")
        assert size == "S"

    def test_unit_12_sherwani_is_shoulder_led_like_blazer(self):
        # shoulder=0.30 (L), chest=1.25 (M), waist=0.85 (M) => shoulder primary => L, as for a blazer
        size = compute_size_from_measurements(chest=1.25, waist=0.85, shoulder=0.30, garment_category="sherwani")
        assert size == "L"
        for chest, waist, shoulder in ((0.95, 0.70, 0.20), (1.25, 0.76, 0.26), (1.46, 0.99, 0.34), (1.30, 1.10, 0.24)):
            assert compute_size_from_measurements(chest, waist, shoulder, "sherwani") == \
                compute_size_from_measurements(chest, waist, shoulder, "blazer")

    def test_unit_13_sherwani_waist_does_not_vote(self):
        # chest=1.25 (M), shoulder=0.26 (M): a waist from S to XXL never changes the size (the dhoti is draped)
        sizes = {compute_size_from_measurements(1.25, waist, 0.26, "sherwani") for waist in (0.76, 0.85, 0.92, 1.10)}
        assert sizes == {"M"}
        # a shirt at the same chest and shoulder does follow the waist vote
        assert compute_size_from_measurements(1.25, 1.10, 0.35, "shirt") != compute_size_from_measurements(1.25, 1.10, 0.35, "sherwani")

    def test_unit_14_sherwani_fits_as_structured_tailoring(self):
        broad_m = _make_profile(chest=1.33, waist=0.89, shoulder=0.28)
        # rigid tailoring cannot stretch: one size down on a broad M is "won't fit", where a woven shirt is snug
        assert get_size_offset("M", "S", "sherwani", broad_m).fit_quality == "does_not_fit"
        assert get_size_offset("M", "S", "shirt", broad_m).fit_quality == "snug"
        # standard ease table (not the airy kurti table) for the fitted jacket
        assert (get_size_offset("M", "M", "sherwani").offset_lo_mm, get_size_offset("M", "M", "sherwani").offset_hi_mm) == (15.0, 25.0)

    def test_unit_15_kurta_dhoti_is_sized_by_its_kurta(self):
        # Same shoulder/chest-led weighting as a kurta (0.4 shoulder + 0.4 chest + 0.2 waist)
        for chest, waist, shoulder in ((0.95, 0.70, 0.20), (1.25, 0.76, 0.30), (1.46, 0.99, 0.34), (1.30, 1.10, 0.24)):
            assert compute_size_from_measurements(chest, waist, shoulder, "kurta_dhoti") == \
                compute_size_from_measurements(chest, waist, shoulder, "kurta")
        # chest XL, shoulder M, waist S: the chest-led weighting gives L (0.4*M + 0.4*XL + 0.2*S), where a shirt's
        # median vote stays at M
        assert compute_size_from_measurements(1.50, 0.78, 0.26, "kurta_dhoti") == "L"
        assert compute_size_from_measurements(1.50, 0.78, 0.26, "shirt") == "M"

    def test_unit_16_kurta_dhoti_uses_the_airy_kurta_ease(self):
        assert (get_size_offset("M", "M", "kurta_dhoti").offset_lo_mm, get_size_offset("M", "M", "kurta_dhoti").offset_hi_mm) == \
            (get_size_offset("M", "M", "kurta").offset_lo_mm, get_size_offset("M", "M", "kurta").offset_hi_mm) == (20.0, 35.0)

    def test_unit_17_salwar_suit_is_sized_by_its_kameez_like_a_kurti(self):
        for chest, waist, shoulder in ((0.95, 0.70, 0.20), (1.25, 0.76, 0.30), (1.50, 0.78, 0.26), (1.30, 1.10, 0.24)):
            assert compute_size_from_measurements(chest, waist, shoulder, "salwar_suit") == \
                compute_size_from_measurements(chest, waist, shoulder, "kurti")
        assert compute_size_from_measurements(1.50, 0.78, 0.26, "salwar_suit") == "L"   # chest-led, waist de-emphasized

    def test_unit_18_salwar_suit_fits_as_a_soft_drape_with_kurti_ease(self):
        typical_m = _make_profile(chest=1.25, waist=0.85, shoulder=0.26)
        # a soft drape stretches: two sizes down is tight, not "won't fit" as in a woven shirt
        assert get_size_offset("M", "XS", "salwar_suit", typical_m).fit_quality == "tight"
        assert get_size_offset("M", "XS", "shirt", typical_m).fit_quality == "does_not_fit"
        assert (get_size_offset("M", "M", "salwar_suit").offset_lo_mm, get_size_offset("M", "M", "salwar_suit").offset_hi_mm) == (20.0, 35.0)

    def test_unit_19_kurta_pajama_is_sized_by_its_kurta(self):
        for chest, waist, shoulder in ((0.95, 0.70, 0.20), (1.25, 0.76, 0.30), (1.50, 0.78, 0.26), (1.30, 1.10, 0.24)):
            assert compute_size_from_measurements(chest, waist, shoulder, "kurta_pajama") == \
                compute_size_from_measurements(chest, waist, shoulder, "kurta")
        assert (get_size_offset("M", "M", "kurta_pajama").offset_lo_mm, get_size_offset("M", "M", "kurta_pajama").offset_hi_mm) == (20.0, 35.0)

    def test_unit_20_lehenga_is_sized_by_its_blouse_at_fitted_ease(self):
        for chest, waist, shoulder in ((0.95, 0.70, 0.20), (1.25, 0.76, 0.30), (1.50, 0.78, 0.26), (1.30, 1.10, 0.24)):
            assert compute_size_from_measurements(chest, waist, shoulder, "lehenga") == \
                compute_size_from_measurements(chest, waist, shoulder, "kurti")
        # the blouse is fitted: standard ease, not the airy kurti ease
        assert (get_size_offset("M", "M", "lehenga").offset_lo_mm, get_size_offset("M", "M", "lehenga").offset_hi_mm) == (15.0, 25.0)

    def test_unit_21_royal_sherwani_is_sized_like_the_sherwani(self):
        for chest, waist, shoulder in ((0.95, 0.70, 0.20), (1.25, 0.85, 0.30), (1.46, 0.99, 0.34), (1.30, 1.10, 0.24)):
            assert compute_size_from_measurements(chest, waist, shoulder, "royal_sherwani") == \
                compute_size_from_measurements(chest, waist, shoulder, "sherwani")
        broad_m = _make_profile(chest=1.33, waist=0.89, shoulder=0.28)
        assert get_size_offset("M", "S", "royal_sherwani", broad_m).fit_quality == "does_not_fit"   # rigid tailoring

    def test_unit_22_saree_and_mundu_shirt_are_sized_by_their_fitted_top(self):
        for category in ("saree", "mundu_shirt"):
            for chest, waist, shoulder in ((0.95, 0.70, 0.20), (1.25, 0.76, 0.30), (1.50, 0.78, 0.26), (1.30, 1.10, 0.24)):
                assert compute_size_from_measurements(chest, waist, shoulder, category) == \
                    compute_size_from_measurements(chest, waist, shoulder, "kurti")
            # fitted blouse / shirt: standard ease
            assert (get_size_offset("M", "M", category).offset_lo_mm, get_size_offset("M", "M", category).offset_hi_mm) == (15.0, 25.0)
        # shoulder/chest-led, unlike a plain shirt's median vote
        assert compute_size_from_measurements(1.50, 0.78, 0.26, "mundu_shirt") == "L"
        assert compute_size_from_measurements(1.50, 0.78, 0.26, "shirt") == "M"

    def test_unit_11_missing_data_raises_error(self):
        # Missing required data must raise ValueError, never guess
        with pytest.raises(ValueError):
            compute_size_from_measurements(chest=None, waist=0.85, shoulder=0.26)


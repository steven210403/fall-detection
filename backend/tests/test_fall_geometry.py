"""Aspect ratio, torso angle, and hip centre (THESIS_PLAN §4.1–§4.2)."""

from __future__ import annotations

import math

import pytest
from tests.helpers_fall import invisible, lying, standing

from fall_detection.fall.geometry import aspect_ratio, compute_geometry, midpoint, torso_angle_deg


class TestAspectRatio:
    def test_upright_box_is_below_one(self) -> None:
        assert aspect_ratio((0.0, 0.0, 50.0, 170.0)) == pytest.approx(50 / 170)

    def test_lying_box_is_above_one(self) -> None:
        assert aspect_ratio((0.0, 0.0, 170.0, 50.0)) == pytest.approx(3.4)

    def test_degenerate_box_is_unmeasurable(self) -> None:
        assert aspect_ratio((10.0, 10.0, 60.0, 10.0)) is None


class TestTorsoAngle:
    def test_vertical_torso_is_ninety_degrees(self) -> None:
        assert torso_angle_deg((100.0, 50.0), (100.0, 110.0)) == pytest.approx(90.0)

    def test_horizontal_torso_is_zero_degrees(self) -> None:
        assert torso_angle_deg((100.0, 50.0), (160.0, 50.0)) == pytest.approx(0.0)

    def test_diagonal_torso_is_forty_five_degrees(self) -> None:
        assert torso_angle_deg((100.0, 100.0), (160.0, 160.0)) == pytest.approx(45.0)

    def test_direction_does_not_matter(self) -> None:
        # Falling to the left or the right, or head-down, is the same angle.
        assert torso_angle_deg((160.0, 160.0), (100.0, 100.0)) == pytest.approx(45.0)
        assert torso_angle_deg((100.0, 160.0), (160.0, 100.0)) == pytest.approx(45.0)

    def test_zero_length_torso_has_no_angle(self) -> None:
        assert torso_angle_deg((100.0, 100.0), (100.0, 100.0)) is None


class TestMidpoint:
    def test_both_sides_average(self) -> None:
        points = [(0.0, 0.0, 0.0)] * 5 + [(10.0, 20.0, 0.9), (30.0, 40.0, 0.9)]
        assert midpoint(points, 5, 6, min_conf=0.3) == ((20.0, 30.0), True)

    def test_one_visible_side_stands_in(self) -> None:
        points = [(0.0, 0.0, 0.0)] * 5 + [(10.0, 20.0, 0.9), (30.0, 40.0, 0.1)]
        assert midpoint(points, 5, 6, min_conf=0.3) == ((10.0, 20.0), False)

    def test_no_visible_side_is_none(self) -> None:
        points = [(0.0, 0.0, 0.0)] * 7
        assert midpoint(points, 5, 6, min_conf=0.3) == (None, False)


class TestComputeGeometry:
    def test_standing_person(self) -> None:
        box, keypoints = standing()
        geometry = compute_geometry(box, keypoints, min_conf=0.3)
        assert geometry.aspect_ratio is not None and geometry.aspect_ratio < 1.0
        assert geometry.torso_angle_deg is not None and geometry.torso_angle_deg > 70.0
        assert geometry.torso_side == "both"
        assert geometry.hip_center is not None
        assert geometry.visible_keypoints == 17

    def test_lying_person(self) -> None:
        box, keypoints = lying()
        geometry = compute_geometry(box, keypoints, min_conf=0.3)
        assert geometry.aspect_ratio is not None and geometry.aspect_ratio >= 1.0
        assert geometry.torso_angle_deg is not None and geometry.torso_angle_deg < 30.0

    def test_body_length_does_not_depend_on_posture(self) -> None:
        # max(W, H) is the same whether the person stands or lies, which is what lets
        # the velocity be normalised during a fall.
        stand = compute_geometry(*standing(), min_conf=0.3)
        lie = compute_geometry(*lying(), min_conf=0.3)
        assert stand.body_length_px == lie.body_length_px

    def test_invisible_keypoints_give_no_angle_but_keep_the_box(self) -> None:
        box, keypoints = invisible((0.0, 0.0, 50.0, 170.0))
        geometry = compute_geometry(box, keypoints, min_conf=0.3)
        assert geometry.torso_angle_deg is None
        assert geometry.hip_center is None
        assert geometry.aspect_ratio is not None
        assert geometry.visible_keypoints == 0

    def test_a_box_only_model_measures_the_box(self) -> None:
        geometry = compute_geometry((0.0, 0.0, 50.0, 170.0), None, min_conf=0.3)
        assert geometry.torso_angle_deg is None
        assert math.isclose(geometry.aspect_ratio or 0.0, 50 / 170)

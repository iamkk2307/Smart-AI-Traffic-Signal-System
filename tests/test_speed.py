"""
Speed monitoring tests.

The claim is that speeds are *measured* in road coordinates, not inferred
from queue length and not computed in pixels. Each test below pins one part
of that claim, including the parts where the right answer is "no number
yet".
"""

import math

import pytest

from app.sensors.speed import (
    ApproachSpeedTracker,
    percentile,
    project,
)

#: Image-to-ground homography for a pinhole camera mounted ``CAM_H`` metres
#: above the road, looking along the approach with its horizon at image row
#: ``V0``. Ground distance goes as 1/(v - V0), so the far end of the
#: approach occupies very few pixels — which is exactly why a scale factor
#: cannot stand in for a homography.
CAM_H, FOCAL, U0, V0 = 8.0, 900.0, 640.0, 200.0
H = [
    [CAM_H, 0.0, -CAM_H * U0],
    [0.0, 0.0, CAM_H * FOCAL],
    [0.0, 1.0, -V0],
]


def road_point(x_m: float, y_m: float):
    """Invert the homography so a test can place a car at a known distance."""
    v = V0 + CAM_H * FOCAL / y_m
    u = U0 + x_m * FOCAL / y_m
    return u, v


def test_projection_is_not_a_scale_factor():
    """Equal pixel steps map to unequal distances, by design."""
    near = project(H, *road_point(0.0, 10.0))
    mid = project(H, *road_point(0.0, 40.0))
    assert near[1] == pytest.approx(10.0, abs=0.2)
    assert mid[1] == pytest.approx(40.0, abs=0.2)

    # Ten pixels of image movement is worth far more metres further away.
    px_near, py_near = road_point(0.0, 10.0)
    px_far, py_far = road_point(0.0, 60.0)
    step_near = abs(project(H, px_near, py_near - 10)[1] - 10.0)
    step_far = abs(project(H, px_far, py_far - 10)[1] - 60.0)
    assert step_far > step_near * 2


def test_single_frame_yields_no_speed():
    """One observation of a car says nothing about how fast it is going."""
    tracker = ApproachSpeedTracker(H)
    stats = tracker.update([road_point(0.0, 50.0)], dt=1.0)
    assert stats.samples == 0
    assert stats.mean == 0.0
    assert tracker.active_tracks == 1


def test_measures_a_known_speed_in_road_coordinates():
    """A car closing 10 m/s must read as 36 km/h, wherever it is on screen."""
    tracker = ApproachSpeedTracker(H)
    distance = 80.0
    for _ in range(8):
        tracker.update([road_point(0.0, distance)], dt=1.0)
        distance -= 10.0
    stats = tracker.stats()
    assert stats.samples >= 5
    assert stats.mean == pytest.approx(36.0, rel=0.08)


def test_speed_is_independent_of_screen_position():
    """The same physical speed near and far must not report differently.

    This is the test that fails for any pixels-per-frame implementation.
    """
    results = []
    for start in (90.0, 30.0):
        tracker = ApproachSpeedTracker(H)
        distance = start
        for _ in range(5):
            tracker.update([road_point(0.0, distance)], dt=0.5)
            distance -= 5.0  # 10 m/s again
        results.append(tracker.stats().mean)
    assert results[0] == pytest.approx(results[1], rel=0.1)


def test_implausible_displacement_is_rejected_not_averaged():
    """A track swap must not drag the mean up; it must be counted."""
    tracker = ApproachSpeedTracker(H, max_plausible_kmh=120.0)
    tracker.update([road_point(0.0, 70.0)], dt=1.0)
    tracker.update([road_point(0.0, 60.0)], dt=1.0)  # 36 km/h
    before = tracker.stats().mean
    # Teleport the detection 50 m in one second: 180 km/h.
    tracker.update([road_point(0.0, 10.0)], dt=1.0)
    after = tracker.stats()
    assert after.mean == pytest.approx(before, rel=0.01)
    assert tracker.rejected_associations == 1


def test_speeding_is_counted_against_the_posted_limit():
    tracker = ApproachSpeedTracker(H, speed_limit_kmh=50.0)
    distance = 100.0
    for _ in range(6):
        tracker.update([road_point(0.0, distance)], dt=1.0)
        distance -= 18.0  # 64.8 km/h
    stats = tracker.stats()
    assert stats.mean > 50.0
    assert stats.speeding == 1

    slow = ApproachSpeedTracker(H, speed_limit_kmh=50.0)
    distance = 100.0
    for _ in range(6):
        slow.update([road_point(0.0, distance)], dt=1.0)
        distance -= 8.0  # 28.8 km/h
    assert slow.stats().speeding == 0


def test_dropped_tracks_are_forgotten():
    tracker = ApproachSpeedTracker(H, drop_after=1.0)
    tracker.update([road_point(0.0, 50.0)], dt=1.0)
    assert tracker.active_tracks == 1
    tracker.update([], dt=1.5)
    assert tracker.active_tracks == 0


def test_percentile_matches_linear_interpolation():
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 0.0) == 10.0
    assert percentile(values, 1.0) == 40.0
    assert percentile(values, 0.5) == pytest.approx(25.0)
    assert percentile([], 0.85) == 0.0


def test_p85_exceeds_the_mean_on_a_dispersed_sample():
    """The 85th percentile is reported because it is not the mean."""
    tracker = ApproachSpeedTracker(H)
    distance = 150.0
    steps = [10.0, 12.0, 9.0, 14.0, 8.0, 11.0, 15.0, 10.0]
    for step in steps:
        tracker.update([road_point(0.0, distance)], dt=1.0)
        distance -= step
    stats = tracker.stats()
    assert stats.samples == len(steps) - 1
    assert stats.p85 > stats.mean


def test_degenerate_homography_point_is_skipped_not_raised():
    tracker = ApproachSpeedTracker([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    stats = tracker.update([(100.0, 200.0)], dt=1.0)
    assert stats.samples == 0
    assert tracker.active_tracks == 0

# Camera calibration

The camera backend refuses to start without a calibration file. This is
deliberate. Detection gives boxes in image space; queue length, occupancy
and speed are quantities on the road surface. Without a mapping between the
two, every number the sensor produces is a pixel count wearing a traffic
engineering label.

A worked example lives in `data/camera.example.json`. Point
`CAMERA_CALIBRATION` at your own copy.

## What each approach needs

| Key | Meaning |
|---|---|
| `stop_line_polygon` | Image-space polygon covering the detection zone for that approach. Vehicles are counted by their ground-contact point, so draw the polygon over the road surface, not over the vehicle bodies. |
| `homography` | 3×3 matrix mapping image points to road-plane metres: `[x, y, w] = H · [u, v, 1]`, road point `(x/w, y/w)`, with `y` measured along the approach away from the camera. |
| `zone_area_m2` | Road-plane area of the polygon. Occupancy is a fraction of usable space, so it needs an area in metres, not in pixels. |
| `lanes` | Lane count on the approach. |
| `speed_limit_kmh` | Posted limit. Speeding counts are measured against this, so an invented value produces invented enforcement data. |

## Obtaining the homography

Four correspondences are enough, and they do not require surveying
equipment:

1. Pick four points on the road surface that you can identify in the image
   and measure on the ground — lane-line ends, stop-line corners, drain
   covers, or temporary cones laid out before commissioning.
2. Measure their positions in metres relative to one origin, by convention
   the centre of the stop line, with `y` increasing away from the camera.
3. Read off their pixel coordinates in a still frame.
4. Solve for `H`:

```python
import cv2, numpy as np
image_pts = np.float32([[640, 980], [980, 980], [735, 430], [860, 430]])
road_pts  = np.float32([[-3.5, 0.0], [3.5, 0.0], [-3.5, 45.0], [3.5, 45.0]])
H, _ = cv2.findHomography(image_pts, road_pts)
print(H.tolist())
```

Spread the points over the full depth of the zone. Four points clustered
near the stop line fit the near field well and extrapolate badly into the
far field, which is exactly where the perspective compression is worst and
where a speed error would be largest.

## Checking it

Project a known distance and see whether you get it back:

```python
from app.sensors.speed import project
print(project(H, 735, 430))   # expect roughly (-3.5, 45.0)
```

Then watch a vehicle of known speed — a service vehicle driven at a
speedometer-held 30 km/h is enough — and compare against
`GET /api/sensor`. If the measured speed drifts with distance from the
camera rather than staying flat, the homography is wrong, not the tracker.

## A note on tilt and wobble

A pole-mounted camera moves in wind, and a homography fitted on a calm day
degrades on a windy one. The error shows up as a speed that oscillates with
the sway period. If the site is exposed, re-fit from a frame captured in
typical conditions rather than the calmest one, and treat a rising
`rejected_associations` count in `GET /api/sensor` as the symptom to
investigate.

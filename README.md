# Smart Traffic Signal Control

Adaptive signal control for a four-leg intersection, with a safety interlock
that verifies every command, phase decisions that explain themselves, and
arrival forecasting that is only used when it beats a baseline.

Everything described below is in the repository and covered by tests. There
is no roadmap section listing capabilities that do not exist.

```
detectors ─▶ controller ─▶ interlock ─▶ signal heads
   │             │             │
   │             │             └─▶ fault: flashing red + audit record
   │             └─▶ decision log (reason + the numbers behind it)
   └─▶ forecaster ─▶ pressure term      ─▶ incident rules ─▶ alerts
```

## What it does

| Area | Implementation |
|---|---|
| Phasing | Protected-only ring: north–south, east–west, exclusive pedestrian. Exhaustive conflict matrix over 4 vehicle movements and 4 crosswalks (`app/core/phases.py`) |
| Intervals | Every phase service runs GREEN → YELLOW → ALL\_RED. No code path skips either, including emergency preemption |
| Control | Actuated: minimum green commitment, gap-out with hysteresis, max-out, rest-in-green when nothing conflicts, hard red ceiling per approach |
| Safety | Independent interlock verifies conflicts, change intervals, minimum green/yellow, and clearance; trips to flashing red on any finding (`app/core/safety.py`) |
| Preemption | Emergency priority bounded by a hold limit *and* a budget over a trailing window; an exhausted budget is logged and alarmed, never silently dropped |
| Pedestrians | Walk plus flashing clearance, served concurrently with the parallel vehicle phase, or exclusively when a call has waited too long |
| Forecasting | Per-approach arrival model on lag and cyclic-time features, promoted only if it beats persistence on a chronological split (`app/ml/`) |
| Incidents | Deterministic rules for blockage, saturation and excessive delay, plus an isolation forest labelled as an anomaly rather than a diagnosis |
| Speed | Road-plane speed from multi-frame tracking through a per-approach homography; mean, 85th percentile and a speeding count against the posted limit (`app/sensors/speed.py`) |
| Conditions | UV/ambient-light sensor with real solar geometry. Low-sun glare is identified per approach and lengthens clearance, raises minimum green and derates discharge — it is kept out of the pressure score (`app/sensors/environment.py`) |
| Field I/O | MQTT backend for roadside loops, pushbuttons, emergency beacons and the UV sensor. A station that goes silent is reported degraded and its approach is placed on constant recall, never read as empty (`app/sensors/iot.py`) |
| Console | Live junction plan, ring position, detector table with speeds, conditions line, decision log, queue trace, operator controls (`web/index.html`) |

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env            # set OPERATOR_TOKEN to enable operator actions
uvicorn app.main:app --reload
```

Open <http://localhost:8000>. API reference at `/docs`.

```bash
docker compose up --build       # or
make test                       # 116 tests
make bench                      # adaptive vs fixed-time on identical demand
```

## Dataset and demo

```bash
python scripts/generate_data.py --days 7            # random; --seed to reproduce
python scripts/build_demo.py --data data/demo       # self-contained HTML viewer
```

`generate_data.py` writes a detector log, phase-decision log, incident log,
replayable MQTT message stream and a run summary under `data/synthetic/`.
Arrivals are random; everything else is the controller's response to them.
Demand, weather, and detector dropouts vary per day, and the message stream
contains deliberate faults — dropouts, malformed bodies, out-of-range values
— because a clean feed does not exercise the validation or staleness paths.

`build_demo.py` embeds one recorded day in a single HTML file with no server
and no external assets: junction plan, signal heads, queues, the sun's
position with the glare legs marked, the detector table, the decision log as
it happened, and the dropout window. It is playback and says so on the page.

The delay reduction printed by `generate_data.py` is not the figure below and
should not be quoted in its place. It randomises demand, so some days run
close to saturation where a fixed-time plan degrades much faster than an
actuated one; `benchmark.py` holds demand at a level both controllers can
serve, which is the fair comparison.

The console also runs with no backend at all — open `web/index.html`
directly. It then drives the same control logic against an in-page arrival
simulation, and the badge in the header says so. A demo that implied it was
connected to a live controller would be worse than no demo.

## Measured results

`python scripts/benchmark.py --hours 2 --seed 42` — both controllers see an
identical seeded arrival stream, so the only difference is timing policy:

```
mode        mean delay   vs fixed    served   cycles   safety
fixed_time      31.94s      +0.0%      7670       81        0
adaptive        21.68s     -32.1%      7695      107        0
```

Rerun it after any change to the pressure model. A change that does not move
these numbers is not an improvement.

## Design decisions worth knowing about

**The simulation is closed-loop.** Arrivals are exogenous; discharge depends
on what the signal is displaying. A simulator that generates queue numbers
independently of the controller cannot show whether the controller is any
good, and a test suite built on one cannot fail.

**Weather changes timing, not ranking.** Adverse conditions lengthen the
yellow and all-red intervals, raise the minimum green, and derate the
saturation discharge rate. They are deliberately *not* a multiplier on each
approach's score: a factor applied equally to every candidate cannot change
an argmax, so scoring code that does that is dead weight presented as
intelligence. `test_a_uniform_multiplier_cannot_change_the_ranking` pins
this down.

**Accumulated red time only counts where somebody is waiting.** Without that
gate an empty approach builds pressure out of nothing and the ring serves a
street with no vehicles on it. With it, the controller rests in green on a
one-sided demand pattern, which is both correct and cheaper.

**The interlock does not repair bad commands.** A controller that emitted
conflicting greens has unknown internal state; substituting a plausible
green hides the defect and keeps a broken controller in service. Flashing
red is the right failure mode — drivers know what to do with it.

**Glare is a sensor-integrity problem, not a dashboard reading.** A UV index
on its own is decoration. What makes it a control input is the mechanism:
UV irradiance is dominated by the direct beam, so a high reading means the
sun is unobscured, and an unobscured sun at low elevation washes out the
signal head for drivers facing it and saturates the camera the detector
depends on. The magnitude test is illuminance rather than UV, because UV is
near zero at exactly the elevations where glare occurs; UV is used as a
*ratio* to clear-sky at the same elevation, which is what separates a low
sun from a low sun behind cloud. `test_environment.py` pins both.

**A silent detector is not an empty road.** The two produce identical
readings and only one of them is safe to assume. A station past its
staleness window keeps its last values, is flagged, and its approach goes
on constant recall and cannot gap out. The failure mode is a green for a
street that may be empty, which is merely wasteful. The alternative failure
mode is never serving a street that is full.

**Forecasts are not assumed to help.** The model is measured against "predict
last interval" on a chronological split and discarded if it loses. The
split is chronological because the target for interval *t* appears as
`lag_1` in the row for *t+1*, so a shuffled validation set sits on both
sides of the fence and reports a meaningless error.

**Operator endpoints are closed by default.** With no `OPERATOR_TOKEN` set
they return 503 rather than running unauthenticated. CORS origins are
explicit.

## Project layout

```
app/
  config.py              environment-driven settings and the timing plan
  core/
    phases.py            movements, phases, conflict matrix  ← safety foundation
    controller.py        interval state machine, actuated logic, preemption
    safety.py            independent interlock
    types.py             detector readings, commands, decision records
  sim/intersection.py    closed-loop point-queue microsimulation
  sensors/
    base.py              one interface; the controller never learns the source
    simulation.py        closed-loop microsimulation
    camera.py            YOLO detection, zone counting, road-plane speed
    speed.py             homography projection and multi-frame tracker
    environment.py       solar position, UV model, per-approach glare
    iot.py               MQTT roadside feed with staleness handling
  ml/
    features.py          lags, rolling stats, cyclic time encodings
    predictor.py         per-approach forecaster with baseline gate
    incidents.py         rule layer, then an anomaly layer
  db/                    SQLAlchemy schema: detector log, decision audit, incidents
  services/runtime.py    the control loop
  api/                   state, operator, analytics, forecasting routes
  main.py                app, lifespan-managed loop, bounded WebSocket hub
web/index.html           operations console (self-contained, works offline)
scripts/benchmark.py     adaptive vs fixed-time
scripts/generate_data.py synthetic dataset from real controller runs
scripts/build_demo.py    self-contained playback viewer
web/demo_template.html   viewer template; data is injected at build time
tests/                   116 tests
```

## Tests

The suite is organised around the claims, not around coverage percentage.

```bash
pytest tests/test_phases.py           # conflict matrix is symmetric and exhaustive
pytest tests/test_safety_interlock.py # the interlock catches each injected defect
pytest tests/test_controller.py       # timing and preemption properties over long runs
pytest tests/test_ml.py               # no leakage, no promotion without a win
pytest tests/test_simulation.py       # red grows queues, green clears them
pytest tests/test_api.py              # operator surface is actually closed
pytest tests/test_speed.py            # speed is road-plane, not pixels-per-frame
pytest tests/test_environment.py      # solar geometry, and glare is selective
pytest tests/test_iot.py              # a silent station is not an empty road
```

`test_safety_interlock.py` matters most: a verifier that never fires has not
been shown to work, so each test hand-builds a command stream containing one
specific defect and asserts the interlock names it.

## Using a camera instead of the simulation

```bash
pip install "ultralytics>=8.1" "opencv-python-headless>=4.9"
# .env
SENSOR_MODE=camera
CAMERA_SOURCE=rtsp://…
CAMERA_CALIBRATION=./data/camera.json
```

Detection gives counts in image space. Queue length, occupancy and arrival
gap are road-plane quantities, so each approach needs a stop-line polygon
and a homography. Without a calibration file the sensor refuses to start
rather than returning pixel counts dressed up as vehicle queues.

## Limits

Worth stating plainly, because these are the gaps a reviewer should ask
about:

- **One intersection.** There is no coordination, no offset optimisation, no
  green wave. Multi-intersection control is a different problem and the
  schema has the `intersections` table but not the logic.
- **Point queues.** The simulation is not a car-following model. That is the
  right fidelity for evaluating signal timing, but it will not tell you
  about spillback into an upstream junction.
- **Protected-only phasing.** No permissive turns, which is what makes the
  conflict matrix exhaustive. A real deployment with permissive lefts needs
  the matrix extended and the phases re-derived.
- **Mean delay is the only objective.** No emissions term, no transit
  priority beyond emergency preemption, no equity weighting between
  approaches.
- **Speeds are measured only on the camera and MQTT backends.** The
  simulation models them from occupancy and free-flow speed. Nothing in the
  shipped dataset or the demo viewer is a tracked measurement.
- **Glare thresholds are defaults, not site calibration.** Elevation band,
  azimuth tolerance and beam fraction are plausible starting values. A real
  site has buildings, a shaded approach and a signal-head visor, all of
  which move those numbers.
- **One UV sensor, no per-approach optics.** Glare is inferred from solar
  geometry plus a single sky reading. A tree line on one leg would need its
  own measurement.
- **Not safety-certified.** Real signal hardware runs a conflict monitor in
  separate silicon for a reason. The interlock here is the software analogue
  of that idea, not a substitute for it.

## Licence

MIT.

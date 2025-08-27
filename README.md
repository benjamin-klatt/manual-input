# Hand Mouse (MediaPipe + pynput)

Turn your **hand motion & micro‑gestures** into **mouse & keyboard** control with a webcam.  
Cross‑platform (Windows / macOS / Linux). No special drivers or virtual HID devices.

- Hand translation → **cursor movement**
- Finger twitches → **left/right click**
- **Closed hand** → clutch (pause movement)
- **Calibration** (F9) and **camera switching** hotkeys
- Output‑centric **YAML config** that’s simple, explicit, and extensible

---

## Purpose

Hands‑free pointing/clicking for accessibility, ergonomic experiments, presentations, and touch‑free kiosks.  
Under the hood: **MediaPipe Hand Landmarker** for 21 landmarks and **pynput** for software input events.

---

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install mediapipe==0.10.* opencv-python pynput pyyaml
```

Place `hand_landmarker.task` **next to** the app (or in `assets/models/`).

**OS notes**
- **Windows**: works out of the box.
- **macOS**: grant Accessibility permissions to your terminal/IDE (System Settings → Privacy & Security → Accessibility).
- **Linux**: works best on **X11**. Wayland may block synthetic mouse events (cursor might not move).

---

## Run

```bash
python -m manualInput.main  # if using the repo layout below
# or
python hand_mouse_pynput_calibrated.py  # if using the single-file version
```

Hotkeys in the preview window
- **F9** — enter/advance **Calibration** (cycles steps)
- **]** or **TAB** — next **camera**
- **[** — previous **camera**
- **r** — rescan cameras
- **ESC** — quit (or cancel calibration if in progress)

---

## How it works (quick)

- Tracks **left/right hands**.
- Projects hand motion onto two learned axes (**Up/Down**, **Left/Right**) to move the cursor.
- Computes **finger curvatures** & **differences** for clicks.
- Uses **gates** (e.g., “only move while hand not closed”). Gates have hysteresis to avoid flicker.
- Saves & reloads a **YAML config**. Missing keys are auto‑filled with sensible defaults.
- Remembers the **last camera by name/ID** (not just index), so it’s stable across device order changes.

---

# Calibration

Press **F9** to step through:

1) **Up/Down motion axis**  
   Move hand **up & down** repeatedly. Learns axis direction + **range** (full sweep → screen height).

2) **Left/Right motion axis**  
   Move **left & right** repeatedly. Learns axis + **range** (full sweep → screen width).

3) **Closed hand range**  
   Slowly close/open; we record **min/max** of `gesture.closed` (avg finger curvature). Default pause trigger: **80%**, release: **60%**.

4) **Left click range**  
   Quick bends of **index** finger. Metric: `curv.diff.index_minus_middle`. Range becomes 0–100%.

5) **Right click range**  
   Quick bends of **middle** finger. Metric: `curv.diff.middle_minus_avg_index_ring`. Range becomes 0–100%.

**Absolute position (perspective)**  
Absolute hand position uses a **four‑corner quad** per hand to account for perspective. Default quad is the **viewport**:
```yaml
quad: [[0,0],[1,0],[1,1],[0,1]]  # TL, TR, BR, BL in camera-normalized coords
```
At runtime, a homography maps the quad → the unit square; `*_hand.pos.x/y` are then 0..1.

After the last step, settings are saved to YAML.

---

# Configuration (YAML)

The app is **output‑centric**: each mapping defines **what you manipulate**, **which input** drives it, and an optional **gate**.

- **No pipelines**: normalization comes from **calibration** (min/max, axes, quad).
- **Hands** are explicit in input names: `left_hand.*` and `right_hand.*`. Cross‑hand inputs live under `hands.*`.
- **We never rewrite your `kind:` strings.** If you write `kind: mouse.click.left`, it stays that way in the file (internally it behaves as press+release).

## File overview

```yaml
version: 1

last_camera:
  backend: any   # "dshow" | "mf" | "avfoundation" | "v4l2" | "any"
  name: "USB2.0 HD UVC WebCam"
  id: "v4l2:/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920-video-index0"
  index: 1       # fallback if name/id lookup fails

smoothing:
  position_ms: 120
  movement_ms: 120
  curvature_ms: 80
  gesture_ms: 80

calibration:
  right_hand.motion.up:   { axis: [0.0, -1.0], range_norm: 0.20 }
  right_hand.motion.left: { axis: [1.0,  0.0], range_norm: 0.20 }

  # Always a quad (TL, TR, BR, BL), perspective-correct
  right_hand.pos:
    quad: [[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]]

  right_hand.gesture.closed: { min: 0.30, max: 0.95 }
  right_hand.curv.diff.index_minus_middle: { min: -0.20, max: 0.50 }
  right_hand.curv.diff.middle_minus_avg_index_ring: { min: -0.20, max: 0.50 }

  # Cross-hand example
  hands.distance: { min: 0.10, max: 0.80 }

outputs:
  # Relative cursor movement (split axes). Negative sensitivity inverts.
  - id: move_x
    kind: mouse.move.x
    input: right_hand.motion.left
    sensitivity: screen.width
    gate: { input: right_hand.gesture.closed, op: "<", trigger_pct: 0.50, release_pct: 0.45, refractory_ms: 120, lost_hand_policy: release }
    lost_hand_policy: zero   # zero|min|max|center|hold|<value>

  - id: move_y
    kind: mouse.move.y
    input: right_hand.motion.up
    sensitivity: -screen.height
    gate: { input: right_hand.gesture.closed, op: "<", trigger_pct: 0.50, release_pct: 0.45 }
    lost_hand_policy: zero

  # Clicks (stateful, linked edges). We keep kind as written in YAML.
  - id: left_click
    kind: mouse.click.left
    input: right_hand.curv.diff.index_minus_middle
    op: ">"                # press when value rises above trigger_pct (default is ">")
    trigger_pct: 0.80
    release_pct: 0.60
    refractory_ms: 250
    gate: { input: right_hand.gesture.closed, op: "<", trigger_pct: 0.5, release_pct: 0.45, refractory_ms: 120, lost_hand_policy: release }
    lost_hand_policy: release   # release|hold|true|toggle

  # Scroll when pinching; engine uses delta of input internally.
  - id: scroll_y
    kind: mouse.scroll.y
    input: hands.distance
    sensitivity: -180
    gate: { input: right_hand.gesture.pinch, op: ">", trigger_pct: 0.6, release_pct: 0.5 }
    lost_hand_policy: zero
```

### Output kinds

- **Delta axes**: `mouse.move.x`, `mouse.move.y`, `mouse.scroll.y`, `mouse.scroll.x`  
  Params: `input`, `sensitivity` (signed), optional `gate`, `lost_hand_policy` (`zero|min|max|center|hold|<value>`).

- **Absolute axes**: `mouse.pos.x`, `mouse.pos.y`  
  Params: `input`, `min`, `max` (screen edges or pixels), optional `gate`, `lost_hand_policy` (same set; default `hold`).

- **Stateful (press+release in one)**: `mouse.click.left|right|middle` (and optionally `key.NAME`)  
  Params: `input`, `op` (`">"`/`"<"`), `trigger_pct`, `release_pct`, `refractory_ms`, optional `gate`, `lost_hand_policy` (`release|hold|true|toggle`).

> **Explicit edges** (optional): If you need custom actions, you may use:
> ```yaml
> kind: { trigger: mouse.click.right.down, release: mouse.click.right.up }
> ```
> The YAML won’t be rewritten; the engine expands internally.


### Input Features

#### Implemented Now

**Per-hand (`left_hand.*` / `right_hand.*`)**

- `*.motion.up`  
  Projection of palm-center movement along the calibrated Up axis.  
  - **Use:** `mouse.move.y` (delta)  
  - **Normalized:** Internal value in 0..1; deltas are taken frame-to-frame  
  - **Calibration:** `{ axis: [ux, uy], range_norm }` in `*.motion.up`  
  - **Smoothing:** `movement_ms`

- `*.motion.left`  
  Projection along the calibrated Left axis.  
  - **Use:** `mouse.move.x` (delta)  
  - **Calibration/Smoothing:** Same as above

- `*.pos.x`, `*.pos.y`  
  Absolute hand position inside a quad (perspective-correct homography → unit square).  
  - **Use:** `mouse.pos.x/y` (absolute)  
  - **Calibration:** `*.pos.quad: [[TL], [TR], [BR], [BL]]`  
  - **Smoothing:** `position_ms`

- `*.gesture.closed`  
  Average finger curvature (index + middle + ring + pinky), normalized to 0..1.  
  - **Use:** Gates (e.g., “only move while closed < 50%”), clutch  
  - **Calibration:** `{ min, max }` in `*.gesture.closed`  
  - **Smoothing:** `gesture_ms`

- `*.curv.diff.index_minus_middle`  
  Index curvature minus middle curvature (for left-click “twitch”).  
  - **Use:** Stateful click thresholds  
  - **Calibration:** `{ min, max }`  
  - **Smoothing:** `curvature_ms`

- `*.curv.diff.middle_minus_avg_index_ring`  
  Middle curvature minus mean(index, ring) (for right-click “twitch”).  
  - **Use/Calibration/Smoothing:** As above

**Cross-hand (`hands.*`)**

- `hands.distance`  
  Distance between palm centers, normalized by palm width (reduces depth effects).  
  - **Use:** Scroll while pinching both hands closer/farther (as a delta)  
  - **Calibration:** `{ min, max }`  
  - **Smoothing:** `movement_ms` (or `gesture_ms` if you prefer)

---

#### Planned / Easy Additions

These are straightforward to expose; list them here so your schema is future-proof.

**Per-hand**

- `*.gesture.pinch`, `*.gesture.spread`, `*.gesture.neutral`  
  - *Pinch* can be proxied by `*.dist.index_thumb` (low distance → high pinch)  
  - *Spread* could be average fingertip pair distance vs palm width  
  - *Neutral* can be a composite score staying near a calibrated “rest” pose

- `*.curv.index|middle|ring|pinky`  
  Raw per-finger curvature (0..~1); useful for custom gestures/gates

- `*.dist.<fingerA>_<fingerB>`  
  Any fingertip distance normalized by palm width (e.g., `dist.index_thumb`, `dist.index_middle`, …)

- `*.tip.index.x|y` (and other fingers)  
  Absolute fingertip position in the pos-quad (0..1)  
  - `*.tip.index.motion.x|y` (delta of the above) is also an option

- `*.rot.roll|pitch|yaw`  
  Approximate hand orientation (via plane fit or PnP). Good for advanced mappings

**Cross-hand**

- `hands.angle`  
  Angle between wrist-to-wrist vector and camera axes

- `hands.symmetricity`  
  How mirrored finger states are

- `hands.pinch_distance`  
  Minimum fingertip-to-fingertip cross-hand

Each of these would follow the same pattern: normalized 0..1 via calibration (`{ min, max }` or a quad/axis for positions/motion), and use category smoothing.

---

#### Semantics & Calibration Quick-Reference

- **Normalization:** Every input you reference is normalized to 0..1 using its calibration block:
  - **Motion axes:** `axis: [x, y]`, `range_norm` (full sweep → 1.0). Deltas for movement come from frame-to-frame changes of the normalized value.
  - **Position:** `pos.quad` homography → unit square → x/y in 0..1.
  - **Metrics (curvatures, diffs, distances, gestures):** `{ min, max }` → 0..1.

- **Smoothing categories:**  
  `position_ms`, `movement_ms`, `curvature_ms`, `gesture_ms`


### Gates (with hysteresis)

Any output can include `gate` (single) or `gate_all` (AND list). Gates are **stateful** with hysteresis & refractory to prevent flicker:

```yaml
gate:
  input: right_hand.gesture.closed
  op: "<"
  trigger_pct: 0.50
  release_pct: 0.45
  refractory_ms: 120
  lost_hand_policy: release   # release|hold|true|toggle for the gate's boolean state
```

If the **gate becomes false**, stateful outputs **release** immediately (unless their own `lost_hand_policy` says otherwise).

---

## First launch & autofill

On first run (and when keys are missing), the app **adds defaults**:

- `last_camera` (by **name/ID**; index only as fallback)
- `smoothing` (category windows)
- **Calibration** stubs for each referenced input:
  - `*_hand.motion.up/left`: `axis [0,-1]/[1,0]`, `range_norm 0.20`
  - `*_hand.pos.quad`: `[[0,0],[1,0],[1,1],[0,1]]`
  - `gesture.closed`: `min 0.30`, `max 0.95`
  - curvature diffs: `min -0.20`, `max 0.50`
  - distances: `min 0.10`, `max 0.80`
- **Outputs**:
  - stateful: `trigger_pct 0.80`, `release_pct 0.60`, `refractory_ms 250`, `op ">”`, `lost_hand_policy release`
  - move.x/y: `sensitivity screen.width/height`, `lost_hand_policy zero`
  - scroll: `sensitivity 120`, `lost_hand_policy zero`
  - pos.x/y: `min/max` = screen edges, `lost_hand_policy hold`

---

## Project structure (recommended)

```text
src/
├─ src/
│  ├─ __init__.py
│  ├─ main.py                 # app entry: camera loop, hotkeys, calibration, runner
│  ├─ io/
│  │  ├─ camera.py            # enumerate/open camera by name/ID; remember last_camera
│  │  └─ screen.py            # screen size helpers
│  ├─ input/
│  │  ├─ tracker.py           # MediaPipe hand landmarker wrapper
│  │  ├─ features.py          # compute inputs: pos, motion, curvature, distances, gestures
│  │  ├─ calibration.py       # load/save; axis PCA; quad homography; defaults/autofill
│  │  └─ smoothing.py         # time-based EMA per category
│  ├─ runtime/
│  │  ├─ engine.py            # evaluate gates + run outputs per frame
│  │  └─ state.py             # per-output and per-gate state (hysteresis, refractory, loss timers)
│  ├─ outputs/
│  │  ├─ mouse.py             # mouse.move/pos/scroll + buttons (pynput)
│  │  └─ keyboard.py          # optional: key down/up
│  ├─ ui/
│  │  ├─ overlay.py           # on-screen overlay (values, gate states, hints)
│  │  └─ hotkeys.py           # F9 calibration, [, ], TAB, r, ESC
│  ├─ config/
│  │  ├─ loader.py            # read YAML, validate, autofill, expand kinds internally
│  │  └─ schema.md            # (doc) keys, enums, defaults
│  └─ assets/
│     └─ models/
│        └─ hand_landmarker.task   # place the model here
├─ configs/
│  └─ example_config.yaml
├─ README.md
├─ requirements.txt
├─ LICENSE
└─ .gitignore
```

You can start with `README.md` and `configs/example_config.yaml`, then migrate your current single-file logic into `src/main.py` and the folders above in small steps.

---

## Troubleshooting

- **No hand detected**: ensure good lighting; keep the hand in frame; try another camera (TAB / `[` / `]`).
- **Cursor doesn’t move on Linux**: likely Wayland; try an X11 session.
- **Clicks too sensitive / not enough**: adjust `trigger_pct` / `release_pct` or recalibrate steps 4–5.
- **Cursor too fast/slow**: tune `sensitivity` for `mouse.move.x/y`, or recalibrate steps 1–2.
- **Jitter**: increase `smoothing.*_ms`.
- **Hand lost behavior**: set `lost_hand_policy` appropriately (e.g., `release` for clicks, `zero` for deltas).

---

## License

Add your preferred license (e.g., MIT).

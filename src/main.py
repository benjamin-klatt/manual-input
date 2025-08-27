# handmouse/main.py
"""
Starter app loop for Hand Mouse.
- Loads YAML config (adds sensible defaults on missing keys)
- Opens last camera by name/ID (fallback to index)
- Tracks hands via MediaPipe Hand Landmarker
- Computes inputs (motion, position via quad, gesture.closed, curvature diffs, hands.distance)
- Evaluates gates (with hysteresis + lost_hand_policy)
- Emits outputs: mouse.move.x/y, mouse.scroll.x/y, mouse.pos.x/y, mouse.click.*
- Minimal overlay + camera hotkeys

Dependencies:
  pip install mediapipe==0.10.* opencv-python pynput pyyaml numpy
"""


import os, sys, math, time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

import cv2
import numpy as np
from src.input.HandState import HandState
from src.input.tracker import HandTracker
from src.io.camera import CameraSwitcher
from src.config import loader as config_loader
from src.ui.debug_overlay import debug_overlay
from src.input.geometry import L, add, sub, dot, norm, palm_center, palm_width


# -------------------- Paths --------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DEFAULT_CONFIG_PATH = config_loader.DEFAULT_CONFIG_PATH

# Try both assets locations
MODEL_CANDIDATES = [
    os.path.join(HERE, "assets", "models", "hand_landmarker.task"),
    os.path.join(ROOT, "handmouse", "assets", "models", "hand_landmarker.task"),
    os.path.join(ROOT, "assets", "models", "hand_landmarker.task"),
    os.path.join(HERE, "hand_landmarker.task"),
]

def download_hand_landmarker_task(dest_path: str) -> bool:
    """
    Download the hand_landmarker.task model file from MediaPipe if not present.
    Returns True if download succeeded, False otherwise.
    """
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        print(f"[info] Downloading hand_landmarker.task to {dest_path} ...")
        urllib.request.urlretrieve(url, dest_path)
        print("[info] Download complete.")
        return True
    except Exception as e:
        print(f"[error] Failed to download hand_landmarker.task: {e}")
        return False

# -------------------- Utilities --------------------



def parse_sensitivity(val, screen_w, screen_h) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        s = val.strip()
        neg = s.startswith("-")
        if neg:
            s = s[1:].strip()
        out = None
        if s == "screen.width":  out = float(screen_w)
        elif s == "screen.height": out = float(screen_h)
        else:
            try:
                out = float(s)
            except:
                out = 0.0
        return -out if neg else out
    return 0.0

# -------------------- Smoothing --------------------
class TimeEMA:
    def __init__(self, tau_ms=120.0, init=None):
        self.tau = max(1.0, float(tau_ms))
        self.y = init
        self.t_prev = None
    def set_tau(self, tau_ms):
        self.tau = max(1.0, float(tau_ms))
    def reset(self, init=None):
        self.y = init; self.t_prev = None
    def update(self, x, now_ms):
        if self.y is None:
            self.y = x
            self.t_prev = now_ms
            return self.y
        dt = max(0.0, now_ms - (self.t_prev or now_ms))
        self.t_prev = now_ms
        a = 1.0 - math.exp(-dt / self.tau)
        if isinstance(x, tuple):
            self.y = ((1-a)*self.y[0] + a*x[0], (1-a)*self.y[1] + a*x[1])
        else:
            self.y = (1-a)*self.y + a*x
        return self.y




# -------------------- Main loop --------------------

def main():
    # Allow passing a custom config path
    cfg_path = DEFAULT_CONFIG_PATH
    if len(sys.argv) > 1:
        cfg_path = sys.argv[1]
    cfg = config_loader.load_yaml(cfg_path)
    cfg = config_loader.ensure_defaults(cfg)


    # Find model, try download if not found
    model_path = None
    for c in MODEL_CANDIDATES:
        if os.path.exists(c):
            model_path = c
            break
    if not model_path:
        # Try to download to the first candidate
        dest = MODEL_CANDIDATES[0]
        if download_hand_landmarker_task(dest) and os.path.exists(dest):
            model_path = dest
        else:
            print("[error] hand_landmarker.task not found and download failed. Put it in handmouse/assets/models/")
            sys.exit(1)

    # Camera
    switcher = CameraSwitcher(width=640, height=480, fps=30)
    preferred_index = int(cfg.get("last_camera", {}).get("index", 0))
    cap, cam_idx = switcher.open(preferred_index)
    if cam_idx is None:
        print("[error] No camera could be opened.")
        sys.exit(1)
    # TODO: enumerate device name/ID per platform; for now save index
    cfg["last_camera"]["index"] = cam_idx
    config_loader.write_yaml(cfg_path, cfg)

    # Tracker
    tracker = HandTracker(model_path)


    # --- Modular pipeline: FeatureIndex, ActuatorBuilder, GateBuilder, BindingIndex ---
    from src.input.features import FeatureIndex
    from src.outputs.actuators import ActuatorBuilder
    from src.gate.gate import GateBuilder
    from src.binding.binding import BindingIndex

    # Feature index
    feature_index = FeatureIndex(cfg.get('calibration', {}))

    # Actuator builder
    actuator_builder = ActuatorBuilder()

    # Gate builder (simple wrapper for now)


    gate_builder = GateBuilder(feature_index)

    # Build bindings from config
    binding_index = BindingIndex(cfg, feature_index, actuator_builder, gate_builder)


    font = cv2.FONT_HERSHEY_SIMPLEX
    last_seen_ms = int(time.time()*1000)

    while True:
        ok, frame = cap.read()
        if not ok:
            cap, cam_idx = switcher.next()
            cfg["last_camera"]["index"] = cam_idx
            config_loader.write_yaml(cfg_path, cfg)
            continue

        ts = int(time.time()*1000)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_list = tracker.detect(rgb, ts)
        hands_map: Dict[str, HandState] = {h.label: h for h in hands_list}
        any_tracked = len(hands_map) > 0
        if any_tracked: last_seen_ms = ts

        # Get left/right hand for bindings
        left_hand = hands_map.get('Left')
        right_hand = hands_map.get('Right')

        # Update all bindings
        binding_index.update(left_hand, right_hand)

        # ---- Overlay ----
        overlay_lines = []
        overlay_lines.append(f"Cam [{cam_idx}]  ( [ / ] / TAB switch, r rescan )")
        overlay_lines.append("ESC: quit   F9: calibrate (stub)")

        # Debug overlay for each binding (per-binding debug key)
        for i, binding in enumerate(binding_index.bindings):
            # Try to get debug keys for this binding from config
            binding_cfg = None
            if hasattr(binding, 'id'):
                # Find the config for this binding by id
                for cfg_b in cfg.get('bindings', []):
                    if cfg_b.get('id') == getattr(binding, 'id', None):
                        binding_cfg = cfg_b
                        break
            debug_keys = set()
            if binding_cfg and 'debug' in binding_cfg and isinstance(binding_cfg['debug'], list):
                debug_keys = set(binding_cfg['debug'])
            if debug_keys:
                probe = binding.probe_last()
                line = f"{getattr(binding, 'id', f'binding_{i}')}: "
                parts = []
                def fmt(v):
                    if isinstance(v, bool):
                        return "1" if v else "0"
                    if isinstance(v, float):
                        sign = "+" if v >= 0 else ""
                        return f"{sign}{v:.3f}"
                    if isinstance(v, dict):
                        return '{' + ', '.join(f"{kk}: {fmt(vv)}" for kk, vv in v.items()) + '}'
                    if isinstance(v, list):
                        return '[' + ', '.join(fmt(x) for x in v) + ']'
                    return str(v)
                for k in ['feature', 'gate', 'actuator', 'binding_state', 'binding_value', 'binding_time']:
                    if k in debug_keys:
                        val = probe.get(k, None)
                        parts.append(f"{k}={fmt(val)}")
                line += ", ".join(parts)
                overlay_lines.append(line)


        y0 = 24
        for i, line in enumerate(overlay_lines):
            cv2.putText(frame, line, (10, y0 + 22*i), font, 0.6, (0,255,255), 2, cv2.LINE_AA)


        # ---- DebugOverlay rendering (3D points/lines/vectors) ----
        debug_overlay.addHand(right_hand)
        debug_overlay.render(frame)


        cv2.imshow("Hand Mouse", frame)
        key = cv2.waitKeyEx(1)
        if key == 27:  # ESC
            break
        if key == 0x78:  # F9 (calibration stub)
            # TODO: wire calibration steps to update cfg["calibration"] and write_yaml(cfg_path, cfg)
            print("[info] Calibration UI not yet wired in this starter.")
        # camera hotkeys
        cap, cam_idx = switcher.handle_key(key & 0xFF)
        if cam_idx is not None and cfg["last_camera"].get("index") != cam_idx:
            cfg["last_camera"]["index"] = cam_idx
            config_loader.write_yaml(cfg_path, cfg)

        # Clear debug overlay after rendering
        debug_overlay.clear()

    if cap: cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

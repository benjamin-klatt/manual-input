
import os, sys, json
from typing import Dict, Any

try:
	import yaml
except Exception:
	yaml = None

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_CONFIG_PATH = os.path.join(ROOT, "configs", "config.yaml")

def minimal_default_config() -> Dict[str, Any]:
	return {
		"version": 1,
		"last_camera": {"backend": "any", "name": "", "id": "", "index": 0},
		"smoothing": {"position_ms": 120, "movement_ms": 120, "curvature_ms": 80, "gesture_ms": 80},
		"calibration": {
			"right_hand.motion.up":   {"axis": [0.0, -1.0], "range_norm": 0.20},
			"right_hand.motion.left": {"axis": [1.0,  0.0], "range_norm": 0.20},
			"right_hand.pos":         {"quad": [[0,0],[1,0],[1,1],[0,1]]},
			"right_hand.gesture.closed": {"min": 0.30, "max": 0.95},
			"right_hand.curv.diff.index_minus_middle": {"min": -0.20, "max": 0.50},
			"right_hand.curv.diff.middle_minus_avg_index_ring": {"min": -0.20, "max": 0.50},
			"hands.distance": {"min": 0.10, "max": 0.80},
		},
		"bindings": [
			{
				"id": "move_x",
				"actuator": "mouse.move.x",
				"input": "right_hand.motion.left",
				"sensitivity": "screen.width",
				"gate": {"input": "right_hand.gesture.closed", "op": "<", "trigger_pct": 0.5, "release_pct": 0.45, "refractory_ms": 120, "lost_hand_policy": "release"},
				"lost_hand_policy": "zero"
			},
			{
				"id": "move_y",
				"actuator": "mouse.move.y",
				"input": "right_hand.motion.up",
				"sensitivity": "-screen.height",
				"gate": {"input": "right_hand.gesture.closed", "op": "<", "trigger_pct": 0.5, "release_pct": 0.45, "refractory_ms": 120, "lost_hand_policy": "release"},
				"lost_hand_policy": "zero"
			},
			{
				"id": "left_click",
				"actuator": "mouse.click.left",
				"input": "right_hand.curv.index.rel",
				"op": ">",
				"trigger_pct": 0.8,
				"release_pct": 0.6,
				"refractory_ms": 250,
				"gate": {"input": "right_hand.gesture.closed", "op": "<", "trigger_pct": 0.5, "release_pct": 0.45, "refractory_ms": 120, "lost_hand_policy": "release"},
				"lost_hand_policy": "release"
			},
			{
				"id": "scroll_y",
				"actuator": "mouse.scroll.y",
				"input": "hands.distance.thumb.index",
				"sensitivity": -180,
				"gate": {"input": "right_hand.gesture.pinch", "op": ">", "trigger_pct": 0.6, "release_pct": 0.5, "refractory_ms": 120, "lost_hand_policy": "release"},
				"lost_hand_policy": "zero"
			}
		]
	}

def write_yaml(path: str, data: Dict[str, Any]) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		if yaml:
			yaml.safe_dump(data, f, sort_keys=False)
		else:
			f.write(json.dumps(data, indent=2))

def load_yaml(path: str) -> Dict[str, Any]:
	if not os.path.exists(path):
		print(f"[info] Config not found at {path}. Creating a minimal default.")
		os.makedirs(os.path.dirname(path), exist_ok=True)
		data = minimal_default_config()
		write_yaml(path, data)
		return data
	with open(path, "r", encoding="utf-8") as f:
		if yaml:
			return yaml.safe_load(f) or {}
		return json.loads(f.read())


def ensure_defaults(cfg: Dict[str, Any]):
	# last_camera block
	cfg.setdefault("last_camera", {"backend": "any", "name": "", "id": "", "index": 0})
	# smoothing
	sm = cfg.setdefault("smoothing", {})
	# Get screen size
	def get_screen_size():
		try:
			import platform
			if platform.system() == "Windows":
				import ctypes
				user32 = ctypes.windll.user32
				return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
			import tkinter as tk
			root = tk.Tk(); root.withdraw()
			w, h = root.winfo_screenwidth(), root.winfo_screenheight()
			root.destroy()
			return int(w), int(h)
		except Exception:
			return 1920, 1080

	screen_w, screen_h = get_screen_size()

	# last_camera block
	cfg.setdefault("last_camera", {"backend": "any", "name": "", "id": "", "index": 0})
	# smoothing
	sm = cfg.setdefault("smoothing", {})
	sm.setdefault("position_ms", 120)
	sm.setdefault("movement_ms", 120)
	sm.setdefault("curvature_ms", 80)
	sm.setdefault("gesture_ms", 80)
	# calibration: only add for features used in a binding
	calib = cfg.setdefault("calibration", {})
	# Gather all feature names used in bindings (input and gate.input)
	outs = cfg.setdefault("bindings", [])
	used_features = set()
	for o in outs:
		if 'input' in o:
			used_features.add(o['input'])
		if 'gate' in o and isinstance(o['gate'], dict) and 'input' in o['gate']:
			used_features.add(o['gate']['input'])

	def get_calib_default(feat: str) -> dict:
		# Heuristic: use suffixes and patterns
		if ".motion.up" in feat:
			return {"axis": [0.0, -1.0], "range_norm": 0.20}
		if ".motion.left" in feat:
			return {"axis": [1.0, 0.0], "range_norm": 0.20}
		if ".pos" in feat:
			return {"quad": [[0,0],[1,0],[1,1],[0,1]]}
		if ".gesture.closed" in feat or ".gesture.pinch" in feat:
			return {"min": 0.30, "max": 0.95}
		if ".curv." in feat:
			# All curvatures (per-finger, rel, diff) use same min/max
			return {"min": -0.20, "max": 0.50}
		if ".distance." in feat or feat.endswith(".distance"):
			return {"min": 0.10, "max": 0.80}
		# fallback
		return {"min": 0.0, "max": 1.0}

	for feat in used_features:
		# If .pos.x or .pos.y, add .pos calibration instead
		if feat.endswith('.pos.x') or feat.endswith('.pos.y'):
			base = feat.rsplit('.', 1)[0]  # e.g. right_hand.pos
			if base not in calib:
				calib[base] = get_calib_default(base)
		elif feat not in calib:
			calib[feat] = get_calib_default(feat)

	# Replace 'screen.width' and 'screen.height' in sensitivity with actual values
	for o in outs:
		if 'sensitivity' in o:
			val = o['sensitivity']
			if val == 'screen.width':
				o['sensitivity'] = float(screen_w)
			elif val == '-screen.width':
				o['sensitivity'] = -float(screen_w)
			elif val == 'screen.height':
				o['sensitivity'] = float(screen_h)
			elif val == '-screen.height':
				o['sensitivity'] = -float(screen_h)

	# Set other binding/gate defaults as before
	for o in outs:
		actuator = o.get("actuator", "")
		if isinstance(actuator, dict):
			o.setdefault("op", ">")
			o.setdefault("trigger_pct", 0.80)
			o.setdefault("release_pct", 0.60)
			o.setdefault("refractory_ms", 250)
			o.setdefault("lost_hand_policy", "release")
		elif isinstance(actuator, str) and actuator.startswith("mouse.move."):
			o.setdefault("lost_hand_policy", "zero")
		elif isinstance(actuator, str) and actuator.startswith("mouse.scroll."):
			o.setdefault("sensitivity", 120)
			o.setdefault("lost_hand_policy", "zero")
		elif isinstance(actuator, str) and actuator.startswith("mouse.pos."):
			o.setdefault("min", 0.0)
			o.setdefault("max", 1.0)
			o.setdefault("lost_hand_policy", "hold")
		elif isinstance(actuator, str) and (actuator.startswith("mouse.click.") or actuator.startswith("key.")):
			o.setdefault("op", ">")
			o.setdefault("trigger_pct", 0.80)
			o.setdefault("release_pct", 0.60)
			o.setdefault("refractory_ms", 250)
			o.setdefault("lost_hand_policy", "release")
		# gate defaults
		if "gate" in o and isinstance(o["gate"], dict):
			g = o["gate"]
			g.setdefault("op", ">")
			g.setdefault("trigger_pct", 0.5)
			g.setdefault("release_pct", 0.45)
			g.setdefault("refractory_ms", 120)
			g.setdefault("lost_hand_policy", "release")
	return cfg

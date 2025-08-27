# Feature extraction framework for hand tracking
from typing import Optional, Dict, Any
from src.input.HandState import HandState
from src.input.tracker import HandTracker
import numpy as np
import cv2
from src.input.geometry import finger_curvature_3d, finger_bend_plane_angle


class Feature:
	def normalize_value(self, raw: float) -> float:
		# Linear normalization: 0 at self.min, 1 at self.max, linear in between, not clamped
		min_v = getattr(self, 'min', 0.0)
		max_v = getattr(self, 'max', 1.0)
		rng = max_v - min_v
		if abs(rng) < 1e-6:
			return 0.0
		# If max < min, flip the scale
		if rng > 0:
			return (raw - min_v) / rng
		else:
			return (min_v - raw) / (-rng)
	"""
	Base class for all features. Subclasses must implement getValue.
	"""
	def __init__(self, hand: str, calibration: Optional[Dict[str, Any]] = None):
		self.hand = hand  # 'left' or 'right'
		self.calibration = calibration or {}
		self._last_value = None
		self._last_raw_value = None

	def probe_last_value(self):
		return {'value': self._last_value, 'raw': self._last_raw_value}

	def getValue(self, left_hand: Optional[HandState], right_hand: Optional[HandState]) -> Optional[float]:
		"""
		Returns normalized value (0..1 in calibrated range, can exceed if out of range), or None if hand not detected.
		"""
		raise NotImplementedError



class PositionFeature(Feature):
	"""
	Absolute hand position (x or y) in the calibrated quad.
	"""
	def __init__(self, hand: str, axis: str, calibration: Dict[str, Any]):
		super().__init__(hand, calibration)
		self.axis = axis
		self.quad = np.array(calibration.get("quad", [[0,0],[1,0],[1,1],[0,1]]), dtype=np.float32)
		self.H = self._compute_homography(self.quad)

	def _compute_homography(self, quad):
		dst = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float32)
		return cv2.getPerspectiveTransform(quad, dst)

	def getValue(self, left_hand: Optional[HandState], right_hand: Optional[HandState]) -> Optional[float]:
		hand = left_hand if self.hand == 'left' else right_hand
		if hand is None:
			self._last_value = None
			self._last_raw_value = None
			return None
		pc = np.array(hand.palm_center, dtype=np.float32)
		v = np.array([pc[0], pc[1], 1.0], dtype=np.float32)
		w = self.H @ v
		if w[2] == 0:
			self._last_value = None
			self._last_raw_value = None
			return None
		u, v = float(w[0]/w[2]), float(w[1]/w[2])
		val = u if self.axis == "x" else v
		self._last_raw_value = val
		self._last_value = self.normalize_value(val)
		return self._last_value



class MovementFeature(Feature):
	"""
	Hand movement along a calibrated axis (e.g., up/down, left/right).
	"""
	def __init__(self, hand: str, axis: str, calibration: Dict[str, Any]):
		super().__init__(hand, calibration)
		self.axis = axis
		self.axis_vec = np.array(calibration.get("axis", [0, -1] if axis=="up" else [1, 0]), dtype=np.float32)
		self.range_norm = float(calibration.get("range_norm", 20))
		self.prev_palm = None
		self.prev_hand = None

	def getValue(self, left_hand: Optional[HandState], right_hand: Optional[HandState]) -> Optional[float]:
		hand = left_hand if self.hand == 'left' else right_hand
		# prevent repeated calls for the same hand, each subsequent call would return 0
		if(hand and self.prev_hand == hand):
			return self._last_value
		self.prev_hand = hand

		if hand is None:
			self.prev_palm = None
			self.prev_hand = None
			self._last_value = None
			self._last_raw_value = None
			return None
		pc = np.array(hand.palm_center, dtype=np.float32)
		if self.prev_palm is None:
			self.prev_palm = np.copy(pc)
			self._last_value = 0.0
			self._last_raw_value = 0.0
			return 0.0  # neutral
		d = pc - self.prev_palm
		self.prev_palm[:] = pc
		proj = float(np.dot(d, self.axis_vec))
		val = proj / self.range_norm
		self._last_raw_value = val
		# Clamp to -1..1 for safety
		out = max(-1.0, min(1.0, val))
		self._last_value = out
		return out



class CurvatureFeature(Feature):
	"""
	Computes the curvature of a finger given landmark ids for mcp, pip, dip, tip.
	"""
	def __init__(self, hand: str, ids, calibration: Dict[str, Any]):
		super().__init__(hand, calibration)
		self.ids = ids
		self.min = calibration.get("min", 0.0)
		self.max = calibration.get("max", 4.0)

	def getValue(self, left_hand: Optional[HandState], right_hand: Optional[HandState]) -> Optional[float]:
		hand = left_hand if self.hand == 'left' else right_hand
		if hand is None or not hasattr(hand, "landmarks"):
			self._last_value = None
			return None
		lms = hand.landmarks
		curv = finger_curvature_3d(lms, self.ids)
		self._last_raw_value = curv
		val = self.normalize_value(curv)
		self._last_value = val
		return val




class RelativeCurvatureFeature(Feature):
	"""
	Computes the difference between a main CurvatureFeature and the mean of up to two reference CurvatureFeatures.
	"""
	def __init__(self, hand: str, main: CurvatureFeature, ref1: Optional[CurvatureFeature] = None, ref2: Optional[CurvatureFeature] = None, calibration: Dict[str, Any] = None):
		super().__init__(hand, calibration or {})
		self.main = main
		self.ref1 = ref1
		self.ref2 = ref2
		self.min = (calibration or {}).get("min", -0.2)
		self.max = (calibration or {}).get("max", 0.5)

	def getValue(self, left_hand: Optional[HandState], right_hand: Optional[HandState]) -> Optional[float]:
		hand = left_hand if self.hand == 'left' else right_hand
		if hand is None:
			self._last_value = None
			return None
		main_val = finger_curvature_3d(hand.landmarks, self.main.ids) if hand and hasattr(hand, "landmarks") else None
		ref_vals = []
		if self.ref1:
			ref_val1 = finger_curvature_3d(hand.landmarks, self.ref1.ids)
			ref_vals.append(ref_val1)
		if self.ref2:
			ref_val2 = finger_curvature_3d(hand.landmarks, self.ref2.ids)
			ref_vals.append(ref_val2)
		if main_val is None or not ref_vals:
			self._last_value = None
			return None
		mean_ref = sum(ref_vals) / len(ref_vals)
		diff = main_val - mean_ref
		self._last_raw_value = diff
		val = self.normalize_value(diff)
		self._last_value = val
		return val
	
class BendFeature(Feature):
	def __init__(self, hand, mcp_id, pip_id, calibration):
		super().__init__(hand, calibration)
		self.mcp_id = mcp_id
		self.pip_id = pip_id
		self.min = calibration.get("min", 0.0)
		self.max = calibration.get("max", np.pi/2)
	def getValue(self, left_hand, right_hand):
		hand = left_hand if self.hand == 'left' else right_hand
		if hand is None:
			self._last_value = None
			self._last_raw_value = None
			return None
		angle = finger_bend_plane_angle(hand, self.mcp_id, self.pip_id)
		self._last_raw_value = angle
		val = self.normalize_value(angle)
		self._last_value = val
		return val
	
class RelativeBendFeature(Feature):
	"""
	Computes the difference between a main BendFeature and the mean of up to two reference BendFeatures.
	"""
	def __init__(self, hand: str, main: BendFeature, ref1: Optional[BendFeature] = None, ref2: Optional[BendFeature] = None, calibration: Dict[str, Any] = None):
		super().__init__(hand, calibration or {})
		self.main = main
		self.ref1 = ref1
		self.ref2 = ref2
		self.min = (calibration or {}).get("min", -np.pi/2)
		self.max = (calibration or {}).get("max", np.pi/2)

	def getValue(self, left_hand: Optional[HandState], right_hand: Optional[HandState]) -> Optional[float]:
		hand = left_hand if self.hand == 'left' else right_hand
		if hand is None:
			self._last_value = None
			return None
		self.main.getValue(left_hand, right_hand)
		main_val = self.main._last_raw_value if self.main else None

		ref_vals = []
		if self.ref1:
			self.ref1.getValue(left_hand, right_hand)
			ref_val1 = self.ref1._last_raw_value
			ref_vals.append(ref_val1)
		if self.ref2:
			self.ref2.getValue(left_hand, right_hand)
			ref_val2 = self.ref2._last_raw_value
			ref_vals.append(ref_val2)
		if main_val is None or not ref_vals:
			self._last_value = None
			return None
		# Use raw values for difference
		main_raw = self.main._last_raw_value
		ref_raws = [r._last_raw_value for r in [self.ref1, self.ref2] if r is not None]
		mean_ref = sum(ref_raws) / len(ref_raws)
		diff = main_raw - mean_ref
		self._last_raw_value = diff
		val = self.normalize_value(diff)
		self._last_value = val
		return val

class GestureFeature(Feature):
	"""
	Gesture features, e.g., closed hand (average finger curvature).
	"""
	def __init__(self, hand: str, kind: str, calibration: Dict[str, Any]):
		super().__init__(hand, calibration)
		self.kind = kind
		self.min = calibration.get("min", 0.3)
		self.max = calibration.get("max", 0.95)

	def getValue(self, left_hand: Optional[HandState], right_hand: Optional[HandState]) -> Optional[float]:
		hand = left_hand if self.hand == 'left' else right_hand
		if hand is None or not hasattr(hand, "landmarks"):
			self._last_value = None
			return None
		lms = hand.landmarks
		# Example: closed gesture = avg curvature of index, middle, ring, pinky
		if self.kind == "closed":
			idx = HandState.INDEX_FINGER_MCP
			mid = HandState.MIDDLE_FINGER_MCP
			ring = HandState.RING_FINGER_MCP
			pinky = HandState.PINKY_MCP
			idx_curv = finger_curvature_3d(lms, idx, idx+1, idx+2, idx+3)
			mid_curv = finger_curvature_3d(lms, mid, mid+1, mid+2, mid+3)
			ring_curv = finger_curvature_3d(lms, ring, ring+1, ring+2, ring+3)
			pinky_curv = finger_curvature_3d(lms, pinky, pinky+1, pinky+2, pinky+3)
			curvs = [idx_curv, mid_curv, ring_curv, pinky_curv]
			avg_curv = sum(curvs) / 4.0
			val = self.normalize_value(avg_curv)
			self._last_value = val
			return val
		# Add more gestures as needed
		self._last_value = None
		return None




class DistanceFeature(Feature):
	"""
	Distance between two landmarks (e.g., fingertips), normalized by palm width.
	"""
	def __init__(self, hand: str, id1: int, id2: int, calibration: Dict[str, Any]):
		super().__init__(hand, calibration)
		self.id1 = id1
		self.id2 = id2
		self.min = calibration.get("min", 0.1)
		self.max = calibration.get("max", 0.8)

	def getValue(self, left_hand: Optional[HandState], right_hand: Optional[HandState]) -> Optional[float]:
		hand = left_hand if self.hand == 'left' else right_hand
		if hand is None or not hasattr(hand, "landmarks"):
			self._last_value = None
			return None
		lms = hand.landmarks
		p1 = lms[self.id1]
		p2 = lms[self.id2]
		d = np.hypot(p1.x - p2.x, p1.y - p2.y)
		# Normalize by palm width
		scale = hand.palm_width
		val = d / max(1e-6, scale)
		self._last_raw_value = val
		out = self.normalize_value(val)
		self._last_value = out
		return out

class FeatureIndex:
	"""
	Manages all features and provides lookup by name.
	"""


	def __init__(self, calibration: Dict[str, Any]):
		self.features: Dict[str, Feature] = {}

		for hand in ("right_hand", "left_hand"):
			# Position features
			for axis in ("x", "y"):
				key = f"{hand}.pos.{axis}"
				self.features[key] = PositionFeature(hand, axis, calibration.get(f"{hand}.pos", {}))
			# Movement features
			for axis in ("up", "left"):
				key = f"{hand}.motion.{axis}"
				self.features[key] = MovementFeature(hand, axis, calibration.get(f"{hand}.motion.{axis}", {}))

			# Curvature features for each finger (full finger)
			finger_names = [
				("index", [HandState.INDEX_FINGER_MCP, HandState.INDEX_FINGER_PIP, HandState.INDEX_FINGER_DIP, HandState.INDEX_FINGER_TIP]),
				("middle", [HandState.MIDDLE_FINGER_MCP, HandState.MIDDLE_FINGER_PIP, HandState.MIDDLE_FINGER_DIP, HandState.MIDDLE_FINGER_TIP]),
				("ring", [HandState.RING_FINGER_MCP, HandState.RING_FINGER_PIP, HandState.RING_FINGER_DIP, HandState.RING_FINGER_TIP]),
				("pinky", [HandState.PINKY_MCP, HandState.PINKY_PIP, HandState.PINKY_DIP, HandState.PINKY_TIP]),
			]
			for name, ids in finger_names:
				key = f"{hand}.curv.{name}"
				self.features[key] = CurvatureFeature(hand, ids, calibration.get(f"{hand}.curv.{name}", {}))

			# Relative curvature features for each finger (difference to mean of adjacent fingers)
			rel_refs = {
				"index":  ["middle"],
				"middle": ["index", "ring"],
				"ring":   ["middle", "pinky"],
				"pinky":  ["ring"],
			}
			for name, ids in finger_names:
				main = self.features[f"{hand}.curv.{name}"]
				ref_names = rel_refs[name]
				ref1 = self.features.get(f"{hand}.curv.{ref_names[0]}")
				ref2 = self.features.get(f"{hand}.curv.{ref_names[1]}") if len(ref_names) > 1 else None
				key = f"{hand}.curv.{name}.rel"
				self.features[key] = RelativeCurvatureFeature(hand, main, ref1, ref2, calibration.get(key, {}))

			# Bend features: angle to palm plane (MCP->PIP vs palm plane)

			for name, ids in finger_names:
				key = f"{hand}.bend.{name}"

				self.features[key] = BendFeature(hand, ids[0], ids[3], calibration.get(key, {}))

			# Relative bend features
			for name, ids in finger_names:
				main = self.features[f"{hand}.bend.{name}"]
				ref_names = rel_refs[name]
				ref1 = self.features.get(f"{hand}.bend.{ref_names[0]}")
				ref2 = self.features.get(f"{hand}.bend.{ref_names[1]}") if len(ref_names) > 1 else None
				key = f"{hand}.bend.{name}.rel"
				self.features[key] = RelativeBendFeature(hand, main, ref1, ref2, calibration.get(key, {}))

			# Gesture features
			self.features[f"{hand}.gesture.closed"] = GestureFeature(hand, "closed", calibration.get(f"{hand}.gesture.closed", {}))

			# Distance features: all combinations between fingertips (thumb, index, middle, ring, pinky)
			fingertip_names = [
				("thumb", HandState.THUMB_TIP),
				("index", HandState.INDEX_FINGER_TIP),
				("middle", HandState.MIDDLE_FINGER_TIP),
				("ring", HandState.RING_FINGER_TIP),
				("pinky", HandState.PINKY_TIP),
			]
			for i, (name1, id1) in enumerate(fingertip_names):
				for j, (name2, id2) in enumerate(fingertip_names):
					if i >= j:
						continue  # avoid duplicates and self
					key1 = f"{hand}.dist.{name1}.{name2}"
					key2 = f"{hand}.dist.{name2}.{name1}"
					self.features[key1] = self.features[key2] = DistanceFeature(hand, id1, id2, calibration.get(key1, {}))

	def getFeature(self, name: str) -> Optional[Feature]:
		return self.features.get(name)

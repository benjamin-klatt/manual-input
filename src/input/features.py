# Feature extraction framework for hand tracking
from typing import Optional, Dict, Any
from src.input.HandState import HandState
import numpy as np
import math
import cv2
from src.input.geometry import finger_curvature_3d, finger_bend_plane_angle
from src.ui.debug_overlay import debug_overlay


class Feature:
	def normalize_value(self, raw: float) -> float:
		# Linear normalization: 0 at self.min, 1 at self.max, linear in between, not clamped
		min_v = getattr(self, 'min', 0.0)
		max_v = getattr(self, 'max', 1.0)
		rng = max_v - min_v
		if abs(rng) < 1e-6:
			return 0.0
		return (raw - min_v) / rng

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
		pc = hand.landmarks[HandState.PALM_CENTER]
		v = np.array([pc.x, pc.y, 1.0], dtype=np.float32)
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
		self.min = 0
		self.max = self.range_norm

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
		pc = hand.landmarks[HandState.PALM_CENTER]
		if self.prev_palm is None:
			self.prev_palm = pc.copy()
			self._last_value = 0.0
			self._last_raw_value = 0.0
			return 0.0  # neutral
		d = pc - self.prev_palm
		self.prev_palm.assign(pc)
		val = float(np.dot(d.sArray2(), self.axis_vec))
		self._last_raw_value = val
		out = self.normalize_value(val)
		# Clamp to -1..1 for safety
		out = max(-1.0, min(1.0, out))
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
			idx_curv = finger_curvature_3d(lms, [idx, idx+1, idx+2, idx+3])
			mid_curv = finger_curvature_3d(lms, [mid, mid+1, mid+2, mid+3])
			ring_curv = finger_curvature_3d(lms, [ring, ring+1, ring+2, ring+3])
			pinky_curv = finger_curvature_3d(lms, [pinky, pinky+1, pinky+2, pinky+3])
			curvs = [idx_curv, mid_curv, ring_curv, pinky_curv]
			avg_curv = sum(curvs) / 4.0
			self._last_raw_value = avg_curv
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
		debug_overlay.addLine(p1, p2)
		diff = p2 - p1
		val = math.hypot(diff.wx, diff.wy, diff.wz)
		# # Normalize by palm width
		# val /= max(1e-6, hand.palm_width)
		self._last_raw_value = val
		out = self.normalize_value(val)
		self._last_value = out
		return out
	


class RotationFeature(Feature):
	"""
	Measures the rotation of the hand around an axis defined by two landmarks, relative to a reference point.
	Returns the signed angle (in radians) between the vector from ref to axis1 and the vector from ref to axis2, projected onto the axis.
	"""
	def __init__(self, hand: str, ref_id: int, axis1_id: int, axis2_id: int, calibration: Dict[str, Any]):
		super().__init__(hand, calibration)
		self.ref_id = ref_id
		self.axis1_id = axis1_id
		self.axis2_id = axis2_id
		self.min = calibration.get("min", -np.pi)
		self.max = calibration.get("max", np.pi)

	def getValue(self, left_hand: Optional[HandState], right_hand: Optional[HandState]) -> Optional[float]:
		hand = left_hand if self.hand == 'left' else right_hand
		if hand is None or not hasattr(hand, "landmarks"):
			self._last_value = None
			self._last_raw_value = None
			return None
		lms = hand.landmarks
		ref = lms[self.ref_id]
		a1 = lms[self.axis1_id]
		a2 = lms[self.axis2_id]
		v1 = a1 - ref
		v2 = a2 - ref
		# Project onto the plane orthogonal to the axis (a2 - a1)
		axis = a2 - a1
		axis_vec = axis.sArray()
		axis_norm = np.linalg.norm(axis_vec)
		if axis_norm < 1e-6:
			self._last_value = None
			self._last_raw_value = None
			return None
		axis_unit = axis_vec / axis_norm
		# Remove axis component from v1 and v2
		def project_onto_plane(v, axis_unit):
			v_vec = v.sArray()
			return v_vec - np.dot(v_vec, axis_unit) * axis_unit
		v1_proj = project_onto_plane(v1, axis_unit)
		v2_proj = project_onto_plane(v2, axis_unit)
		# Angle between projections
		dot = np.dot(v1_proj, v2_proj)
		cross = np.cross(v1_proj, v2_proj)
		angle = math.atan2(np.dot(cross, axis_unit), dot)
		self._last_raw_value = angle
		val = self.normalize_value(angle)
		self._last_value = val
		return val


class RollRotationFeature(Feature):
	"""
	Hand roll around the camera/view vertical: compute palm normal (from WRIST, INDEX_MCP, PINKY_MCP),
	force it to point upwards, project onto the plane spanned by the index-pinky vector and the up vector,
	then measure the signed angle to the up vector. Positive sign is consistent across hands.
	"""
	def __init__(self, hand: str, calibration: Dict[str, Any]):
		super().__init__(hand, calibration)
		# Default roll range about +/- 90 degrees
		self.min = calibration.get("min", -np.pi/2)
		self.max = calibration.get("max", np.pi/2)

	def getValue(self, left_hand: Optional[HandState], right_hand: Optional[HandState]) -> Optional[float]:
		hand = left_hand if self.hand == 'left' else right_hand
		if hand is None or not hasattr(hand, "landmarks"):
			self._last_value = None
			self._last_raw_value = None
			return None

		lms = hand.landmarks
		w = lms[HandState.WRIST].sArray()
		idx = lms[HandState.INDEX_FINGER_MCP].sArray()
		pky = lms[HandState.PINKY_MCP].sArray()

		# Up vector in screen coords (y up is negative in image space)
		up = np.array([0.0, 0.0, -1.0], dtype=np.float32)
		up_norm = np.linalg.norm(up)
		if up_norm < 1e-6:
			return None
		up_u = up / up_norm

		# Palm normal from triangle (wrist, index, pinky)
		v1 = idx - w
		v2 = pky - w
		n = np.cross(v1, v2)
		n_norm = np.linalg.norm(n)
		if n_norm < 1e-6:
			self._last_value = None
			self._last_raw_value = None
			return None
		n_u = n / n_norm

		# Ensure normal points "up" (flip if pointing down)
		if not getattr(hand, 'label', 'Right').lower().startswith('left'):
			n_u = -n_u

		# Index-Pinky direction (from pinky to index)
		ip = idx - pky
		ip_norm = np.linalg.norm(ip)
		if ip_norm < 1e-6:
			self._last_value = None
			self._last_raw_value = None
			return None
		ip_u = ip / ip_norm

		# Plane normal for plane spanned by ip_u and up_u
		plane_n = np.cross(ip_u, up_u)
		plane_n_norm = np.linalg.norm(plane_n)
		if plane_n_norm < 1e-6:
			self._last_value = None
			self._last_raw_value = None
			return None
		plane_n_u = plane_n / plane_n_norm

		# Project palm normal onto that plane
		n_proj = n_u - np.dot(n_u, plane_n_u) * plane_n_u
		n_proj_norm = np.linalg.norm(n_proj)
		if n_proj_norm < 1e-6:
			self._last_value = None
			self._last_raw_value = None
			return None
		n_proj_u = n_proj / n_proj_norm

		# Signed angle from up to projected normal within the plane
		num = np.dot(np.cross(up_u, n_proj_u), plane_n_u)
		den = float(np.clip(np.dot(up_u, n_proj_u), -1.0, 1.0))
		angle = math.atan2(num, den)

		# Normalize sign across hands: make left-hand sign match right-hand
		if getattr(hand, 'label', 'Right').lower().startswith('left'):
			angle = -angle

		self._last_raw_value = angle
		val = self.normalize_value(angle)
		self._last_value = val
		# Optional debug vectors
		try:
			debug_overlay.addVector(lms[HandState.PALM_CENTER], n_u * 0.1, color=(255, 255, 0))
			debug_overlay.addVector(lms[HandState.PALM_CENTER], n_proj_u * 0.1, color=(0, 255, 255))
			# Draw up and IP directions for reference
			debug_overlay.addVector(lms[HandState.PALM_CENTER], up_u * 0.1, color=(0, 255, 0))
			debug_overlay.addVector(lms[HandState.PALM_CENTER], ip_u * 0.1, color=(255, 0, 255))
		except Exception:
			pass
		return val

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
			self.features[f"{hand}.curv.thumb"] = CurvatureFeature(hand, [HandState.THUMB_CMC, HandState.THUMB_MCP, HandState.THUMB_IP, HandState.THUMB_TIP], calibration.get(f"{hand}.curv.thumb", {}))

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
			self.features[f"{hand}.dist.thumb.hand"] = self.features[f"{hand}.dist.hand.thumb"] = DistanceFeature(hand, HandState.THUMB_IP, HandState.INDEX_FINGER_MCP, calibration.get(f"{hand}.dist.thumb.hand", {}))

			# Rotation feature example (customize ids as needed)
			rot_key = f"{hand}.rotation"
			# Roll: palm-normal based roll around vertical
			roll_feature = RollRotationFeature(hand, calibration.get(f"{hand}.rotation.roll", {}))
			self.features[f"{hand}.rotation.roll"] = roll_feature
			# Keep legacy alias to roll
			self.features[rot_key] = roll_feature

			rot_key = f"{hand}.rotation.pitch"
			self.features[rot_key] = RotationFeature(
				hand,
				HandState.PINKY_MCP,
				HandState.WRIST,
				HandState.PALM_CENTER,
				calibration.get(rot_key, {})
			)

	def getFeature(self, name: str) -> Optional[Feature]:
		return self.features.get(name)

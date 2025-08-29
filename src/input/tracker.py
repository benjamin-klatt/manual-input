# MediaPipe hand landmarker wrapper (placeholder)
from __future__ import annotations
import numpy as np
from src.input.HandState import HandState
from src.input.geometry import palm_center, palm_width


import mediapipe as mp
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark, Landmark


from typing import List





vision = mp.tasks.vision

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
RunningMode = vision.RunningMode
MPImage = mp.Image


Z_AMPLIFICATION = 1.5

class MultiLandmark(NormalizedLandmark):
    def __init__(self, lm: NormalizedLandmark, wlm: Landmark,  frame_shape):
        super().__init__(lm.x, lm.y, lm.z, lm.visibility, lm.presence)
        self.frame_shape = frame_shape
        self.sx = lm.x * frame_shape[1]
        self.sy = lm.y * frame_shape[0]
        self.sz = lm.z * frame_shape[0]
        self.wlm = wlm
        self.wx = wlm.x
        self.wy = wlm.y
        self.wz = wlm.z
    def nTuple(self):
        return (self.x, self.y, self.z)
    def sTuple(self):
        return (self.sx, self.sy, self.sz)
    def wTuple(self):
        return (self.wlm.x, self.wlm.y, self.wlm.z)
    def nArray(self):
        return np.array([self.x, self.y, self.z])
    def sArray(self):
        return np.array([self.sx, self.sy, self.sz])
    def wArray(self):
        return np.array([self.wx, self.wy, self.wz])
    
    def nTuple2(self):
        return (self.x, self.y)
    def sTuple2(self):
        return (self.sx, self.sy)
    def nArray2(self):
        return np.array([self.x, self.y])
    def sArray2(self):
        return np.array([self.sx, self.sy])

    def assign(self, other: MultiLandmark):
        self.x = other.x
        self.y = other.y
        self.z = other.z
        self.sx = other.sx
        self.sy = other.sy
        self.sz = other.sz
        self.wlm = other.wlm
        self.wx = self.wlm.x
        self.wy = self.wlm.y
        self.wz = self.wlm.z

    def copy(self):
        return MultiLandmark(
                NormalizedLandmark(self.x, self.y, self.z, self.visibility, self.presence),
                Landmark(self.wlm.x, self.wlm.y, self.wlm.z, self.wlm.visibility, self.wlm.presence ),
                self.frame_shape
        )
    
    def __add__(self, o):
        # MultiLandmark + MultiLandmark
        if isinstance(o, MultiLandmark):
            return MultiLandmark(
                NormalizedLandmark(self.x + o.x, self.y + o.y, self.z + o.z, self.visibility, self.presence),
                Landmark(self.wlm.x + o.wlm.x, self.wlm.y + o.wlm.y, self.wlm.z + o.wlm.z, self.wlm.visibility, self.wlm.presence ),
                self.frame_shape
            )
        # MultiLandmark + np.array or tuple/list
        arr = None
        if isinstance(o, np.ndarray) and o.shape == (3,):
            arr = o
        elif isinstance(o, (tuple, list)) and len(o) == 3:
            arr = np.array(o)
        if arr is not None:
            return MultiLandmark(
                NormalizedLandmark(self.x + arr[0], self.y + arr[1], self.z + arr[2], self.visibility, self.presence),
                Landmark(self.wlm.x + arr[0], self.wlm.y + arr[1], self.wlm.z + arr[2], self.wlm.visibility, self.wlm.presence ),
                self.frame_shape
            )
        return NotImplemented
    def __sub__(self, o):
        # MultiLandmark - MultiLandmark
        if isinstance(o, MultiLandmark):
            return MultiLandmark(
                NormalizedLandmark(self.x - o.x, self.y - o.y, self.z - o.z, self.visibility, self.presence),
                Landmark(self.wlm.x - o.wlm.x, self.wlm.y - o.wlm.y, self.wlm.z - o.wlm.z, self.wlm.visibility, self.wlm.presence ),
                self.frame_shape
            )
        # MultiLandmark - np.array or tuple/list
        arr = None
        if isinstance(o, np.ndarray) and o.shape == (3,):
            arr = o
        elif isinstance(o, (tuple, list)) and len(o) == 3:
            arr = np.array(o)
        if arr is not None:
            return MultiLandmark(
                NormalizedLandmark(self.x - arr[0], self.y - arr[1], self.z - arr[2], self.visibility, self.presence),
                Landmark(self.wlm.x - arr[0], self.wlm.y - arr[1], self.wlm.z - arr[2], self.wlm.visibility, self.wlm.presence ),
                self.frame_shape
            )
        return NotImplemented
    def __mul__(self, o):
        # check if number
        try:
            val = float(o)
            return MultiLandmark(
                NormalizedLandmark(self.x * val, self.y * val, self.z * val, self.visibility, self.presence),
                Landmark(self.wlm.x * val, self.wlm.y * val, self.wlm.z * val, self.wlm.visibility, self.wlm.presence ),
                self.frame_shape
            )
        except ValueError:
            pass
    
    def __truediv__(self, o):
        # check if number
        try:
            val = float(o)
            return MultiLandmark(
                NormalizedLandmark(self.x / val, self.y / val, self.z / val, self.visibility, self.presence),
                Landmark(self.wlm.x / val, self.wlm.y / val, self.wlm.z / val, self.wlm.visibility, self.wlm.presence ),
                self.frame_shape
            )
        except ValueError:
            pass

    def __iadd__(self, o):
        if isinstance(o, MultiLandmark):
            self.x += o.x
            self.y += o.y
            self.z += o.z
            self.sx += o.sx
            self.sy += o.sy
            self.sz += o.sz
            self.wlm.x += o.wlm.x
            self.wlm.y += o.wlm.y
            self.wlm.z += o.wlm.z
            self.wx = self.wlm.x
            self.wy = self.wlm.y
            self.wz = self.wlm.z
            return self
        arr = None
        if isinstance(o, np.ndarray) and o.shape == (3,):
            arr = o
        elif isinstance(o, (tuple, list)) and len(o) == 3:
            arr = np.array(o)
        if arr is not None:
            self.x += arr[0]
            self.y += arr[1]
            self.z += arr[2]
            self.sx += arr[0]
            self.sy += arr[1]
            self.sz += arr[2]
            self.wlm.x += arr[0]
            self.wlm.y += arr[1]
            self.wlm.z += arr[2]
            self.wx = self.wlm.x
            self.wy = self.wlm.y
            self.wz = self.wlm.z
            return self
        return NotImplemented
    def __isub__(self, o):
        if isinstance(o, MultiLandmark):
            self.x -= o.x
            self.y -= o.y
            self.z -= o.z
            self.sx -= o.sx
            self.sy -= o.sy
            self.sz -= o.sz
            self.wlm.x -= o.wlm.x
            self.wlm.y -= o.wlm.y
            self.wlm.z -= o.wlm.z
            self.wx = self.wlm.x
            self.wy = self.wlm.y
            self.wz = self.wlm.z
            return self
        arr = None
        if isinstance(o, np.ndarray) and o.shape == (3,):
            arr = o
        elif isinstance(o, (tuple, list)) and len(o) == 3:
            arr = np.array(o)
        if arr is not None:
            self.x -= arr[0]
            self.y -= arr[1]
            self.z -= arr[2]
            self.sx -= arr[0]
            self.sy -= arr[1]
            self.sz -= arr[2]
            self.wlm.x -= arr[0]
            self.wlm.y -= arr[1]
            self.wlm.z -= arr[2]
            self.wx = self.wlm.x
            self.wy = self.wlm.y
            self.wz = self.wlm.z
            return self
        return NotImplemented
    def __imul__(self, o):
        # check if number
        try:
            val = float(o)
            self.x *= val
            self.y *= val
            self.z *= val
            self.sx *= val
            self.sy *= val
            self.sz *= val
            self.wlm.x *= val
            self.wlm.y *= val
            self.wlm.z *= val
            self.wx = self.wlm.x
            self.wy = self.wlm.y
            self.wz = self.wlm.z
            return self
        except ValueError:
            pass
    
    def __idiv__(self, o):
        # check if number
        try:
            val = float(o)
            self.x /= val
            self.y /= val
            self.z /= val
            self.sx /= val
            self.sy /= val
            self.sz /= val
            self.wlm.x /= val
            self.wlm.y /= val
            self.wlm.z /= val
            self.wx = self.wlm.x
            self.wy = self.wlm.y
            self.wz = self.wlm.z
            return self
        except ValueError:
            pass

class HandTracker:
    def __init__(self, model_path: str):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_hands=2, running_mode=RunningMode.VIDEO)
        self.landmarker = HandLandmarker.create_from_options(options)

    def detect(self, rgb_frame, ts_ms: int) -> List[HandState]:
        mp_img = MPImage(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        res = self.landmarker.detect_for_video(mp_img, ts_ms)
        out = []
        if not res.hand_landmarks:
            return out
        # handedness length matches landmarks list
        for i, nlms in enumerate(res.hand_landmarks):
            wlms = res.hand_world_landmarks[i]
            label = "Right"
            if res.handedness and i < len(res.handedness) and len(res.handedness[i]) > 0:
                label = res.handedness[i][0].category_name  # "Left"/"Right"

            # Transform landmarks to screen space coordinates
            lms = []
            for i, lm in enumerate(nlms):
                wlm = wlms[i]
                lm.z *= Z_AMPLIFICATION
                wlm.z *= Z_AMPLIFICATION
                lms.append(MultiLandmark(lm, wlm, rgb_frame.shape))
            lms.append(palm_center(lms))  # add palm center as extra landmark
            pw = palm_width(lms)
            out.append(HandState(label=label, landmarks=lms, palm_width=pw))
        return out


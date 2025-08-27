# MediaPipe hand landmarker wrapper (placeholder)
import numpy as np
from src.input.HandState import HandState
from src.input.geometry import palm_center, palm_width


import mediapipe as mp
from mediapipe.tasks.python.components.containers.landmark  import NormalizedLandmark


from typing import List





vision = mp.tasks.vision

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
RunningMode = vision.RunningMode
MPImage = mp.Image



class Landmark(NormalizedLandmark):

    def __init__(self, lm: NormalizedLandmark, frame_shape):
        super().__init__(lm.x, lm.y, lm.z)
        self.sx = lm.x * frame_shape[1]
        self.sy = lm.y * frame_shape[0]
        self.sz = lm.z * frame_shape[0]
    def nTuple(self):
        return (self.x, self.y, self.z)
    def sTuple(self):
        return (self.sx, self.sy, self.sz)
    def nArray(self):
        return np.array([self.x, self.y, self.z])
    def sArray(self):
        return np.array([self.sx, self.sy, self.sz])

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
            for lm in nlms:
                lms.append(Landmark(lm, rgb_frame.shape))
            pc = palm_center(lms)
            pw = palm_width(lms)
            out.append(HandState(label=label, landmarks=lms, world_landmarks=wlms, palm_center=pc, palm_width=pw))
        return out


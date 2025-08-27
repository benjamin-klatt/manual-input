import cv2
import numpy as np

from src.input.HandState import HandState

class DebugOverlay:
    """
    Collects 3D points, lines, and vectors for debug rendering. Use addPoint, addLine, addVector from anywhere.
    Call clear() after rendering to reset.
    """
    def __init__(self):
        self.points = []  # List of (x, y, z, color)
        self.lines = []   # List of ((x1, y1, z1), (x2, y2, z2), color)
        self.vectors = [] # List of ((x, y, z), (dx, dy, dz), color)

    def _to_xyz(self, v):
        # Accepts (x, y, z) tuple or NormalizedLandmark
        if hasattr(v, 'x') and hasattr(v, 'y') and hasattr(v, 'z'):
            return (float(v.x), float(v.y), float(v.z))
        if isinstance(v, (tuple, list)) and len(v) == 3:
            return (float(v[0]), float(v[1]), float(v[2]))
        if isinstance(v, (np.ndarray)) and v.shape == (3,):
            return (float(v[0]), float(v[1]), float(v[2]))
        raise ValueError(f"Cannot convert {v} to (x, y, z)")

    def addPoint(self, x, y=None, z=None, color=(0,255,0)):
        # Accepts (x, y, z) or NormalizedLandmark as first arg
        if y is None and z is None:
            x, y, z = self._to_xyz(x)
        self.points.append((x, y, z, color))

    def addLine(self, p1, p2, color=(255,0,0)):
        p1 = self._to_xyz(p1)
        p2 = self._to_xyz(p2)
        self.lines.append((p1, p2, color))

    def addVector(self, origin, direction, color=(0,0,255)):
        """origin: (x, y, z), direction: (dx, dy, dz)"""
        
        origin = self._to_xyz(origin)
        direction = self._to_xyz(direction)
        self.vectors.append((origin, direction, color))

    def addHand(self, hand: HandState, color=(0,255,255)):
        if hand is None:
            return
        lms = hand.landmarks
        for lm in lms:
            self.addPoint(lm.x, lm.y, lm.z, color)

        self.addLine(lms[HandState.WRIST], lms[HandState.INDEX_FINGER_MCP], color)
        self.addLine(lms[HandState.INDEX_FINGER_MCP], lms[HandState.MIDDLE_FINGER_MCP], color)
        self.addLine(lms[HandState.MIDDLE_FINGER_MCP], lms[HandState.RING_FINGER_MCP], color)
        self.addLine(lms[HandState.RING_FINGER_MCP], lms[HandState.PINKY_MCP], color)
        self.addLine(lms[HandState.PINKY_MCP], lms[HandState.WRIST], color)
        self.addLine(lms[HandState.INDEX_FINGER_MCP], lms[HandState.PINKY_MCP], color)

        self.addLine(lms[HandState.WRIST], lms[HandState.THUMB_CMC], color)
        self.addLine(lms[HandState.THUMB_CMC], lms[HandState.THUMB_MCP], color)
        self.addLine(lms[HandState.THUMB_MCP], lms[HandState.THUMB_IP], color)
        self.addLine(lms[HandState.THUMB_IP], lms[HandState.THUMB_TIP], color)

        self.addLine(lms[HandState.INDEX_FINGER_MCP], lms[HandState.INDEX_FINGER_PIP], color)
        self.addLine(lms[HandState.INDEX_FINGER_PIP], lms[HandState.INDEX_FINGER_DIP], color)
        self.addLine(lms[HandState.INDEX_FINGER_DIP], lms[HandState.INDEX_FINGER_TIP], color)

        self.addLine(lms[HandState.MIDDLE_FINGER_MCP], lms[HandState.MIDDLE_FINGER_PIP], color)
        self.addLine(lms[HandState.MIDDLE_FINGER_PIP], lms[HandState.MIDDLE_FINGER_DIP], color)
        self.addLine(lms[HandState.MIDDLE_FINGER_DIP], lms[HandState.MIDDLE_FINGER_TIP], color)

        self.addLine(lms[HandState.RING_FINGER_MCP], lms[HandState.RING_FINGER_PIP], color)
        self.addLine(lms[HandState.RING_FINGER_PIP], lms[HandState.RING_FINGER_DIP], color)
        self.addLine(lms[HandState.RING_FINGER_DIP], lms[HandState.RING_FINGER_TIP], color)

        self.addLine(lms[HandState.PINKY_MCP], lms[HandState.PINKY_PIP], color)
        self.addLine(lms[HandState.PINKY_PIP], lms[HandState.PINKY_DIP], color)
        self.addLine(lms[HandState.PINKY_DIP], lms[HandState.PINKY_TIP], color)

    def clear(self):
        self.points.clear()
        self.lines.clear()
        self.vectors.clear()

    def render(self, frame):
        # Example: project 3D to 2D (requires camera intrinsics, here just drop z for demo)
        for (x, y, z, color) in self.points:
            cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 6, color, -1)
        for (p1, p2, color) in self.lines:
            x1, y1, z1 = p1
            x2, y2, z2 = p2
            cv2.line(frame, (int(x1 * frame.shape[1]), int(y1 * frame.shape[0])), (int(x2 * frame.shape[1]), int(y2 * frame.shape[0])), color, 2)
        for (origin, direction, color) in self.vectors:
            x, y, z = origin
            dx, dy, dz = direction
            tip_x = x+dx 
            tip_y = y+dy
            cv2.arrowedLine(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), (int(tip_x * frame.shape[1]), int(tip_y * frame.shape[0])), color, 2, tipLength=0.2)

# Global instance for universal access
debug_overlay = DebugOverlay()
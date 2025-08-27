
import numpy as np

from src.input.HandState import HandState
from src.ui.debug_overlay import debug_overlay


# Calculate the angle between a finger segment and the palm plane (INDEX_FINGER_MCP, PINKY_MCP, WRIST)
def finger_bend_plane_angle(hand, mcp_id, pip_id):
    """
    Returns the angle (in radians) between the MCP->PIP vector and the palm plane defined by INDEX_FINGER_MCP, PINKY_MCP, WRIST.
    0 = finger is in the plane, pi/2 = finger is perpendicular to the plane.
    """
    def L3(p): return np.array([p.x, p.y, p.z], dtype=np.float32)
    lms = hand.landmarks
    # Palm plane
    mcp1 = lms[HandState.INDEX_FINGER_MCP].sArray()
    mcp2 = lms[HandState.MIDDLE_FINGER_MCP].sArray()
    wrist = lms[HandState.WRIST].sArray()
    v1 = mcp1 - wrist
    v2 = mcp2 - wrist
    normal = np.cross(v1, v2)
    normal = normal / (np.linalg.norm(normal) + 1e-9)

    # Finger vector
    mcp = lms[mcp_id].sArray()
    pip = lms[pip_id].sArray()
    finger_vec = pip - mcp

    finger_vec = finger_vec / (np.linalg.norm(finger_vec) + 1e-9)



    debug_overlay.addVector(L3(lms[mcp_id]), normal * 0.1, color=(255, 255, 0))  # Palm normal

    debug_overlay.addVector(L3(lms[mcp_id]), finger_vec * 0.1)  # Palm normal

    # Angle between finger_vec and palm plane (0 = in plane, pi/2 = orthogonal)
    dotprod = np.dot(finger_vec, normal)
    dotprod = np.clip(dotprod, -1.0, 1.0)
    angle = np.arcsin(abs(dotprod))  # abs: treat up/down as same
    return angle

def finger_curvature_3d(lms, ids):
    """
    Compute finger curvature in 3D for a sequence of landmark ids (at least 3).
    Returns sum of (pi - angle) at each interior joint (higher = more bent, 0 = straight).
    """
    if len(ids) < 3:
        return 0.0
    def L3(p): return np.array([p.x, p.y, p.z], dtype=np.float32)
    pts = [L3(lms[i]) for i in ids]
    total = 0.0
    for i in range(1, len(pts)-1):
        a, b, c = pts[i-1], pts[i], pts[i+1]
        v1 = a - b
        v2 = c - b
        n1 = np.linalg.norm(v1) + 1e-9
        n2 = np.linalg.norm(v2) + 1e-9
        cosang = np.dot(v1, v2) / (n1 * n2)
        cosang = np.clip(cosang, -1.0, 1.0)
        angle = np.arccos(cosang)
        total += (np.pi - angle)
    return max(0.0, total)
# Geometry helpers for hand tracking and palm/finger analysis
import math

def L(p):
    return (p.x, p.y)

def add(a, b):
    return (a[0] + b[0], a[1] + b[1])

def sub(a, b):
    return (a[0] - b[0], a[1] - b[1])

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]

def norm(a):
    return math.hypot(a[0], a[1]) + 1e-9

def palm_center(lms):
    pts = [0, 5, 9, 13, 17]
    s = (0.0, 0.0)
    for i in pts:
        s = add(s, L(lms[i]))
    return (s[0] / len(pts), s[1] / len(pts))

def palm_width(lms):
    return norm(sub(L(lms[5]), L(lms[17]))) + 1e-9

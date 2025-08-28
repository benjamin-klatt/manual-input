from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Tuple

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.input.tracker import MultiLandmark

@dataclass
class HandState:
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20   
    PALM_CENTER = 21  # Custom extra landmark for palm center

    label: str                     # "Left" or "Right"
    landmarks: List[MultiLandmark]  # List of 21 landmarks + palm center
    palm_width: float
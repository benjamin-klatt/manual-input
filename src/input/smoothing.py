# HandSmoother: smooths HandState landmarks over a time window
import time
from collections import deque
from typing import Optional
from src.input.HandState import HandState
from src.input.geometry import palm_width


class HandSmootherIndex:
	def __init__(self, smoothing_time):
		self.smoothers = {}
		self.smoothing_time = smoothing_time

	def smoothe(self, label, hand):
		if label not in self.smoothers:
			self.smoothers[label] = HandSmoother(smoothing_time=self.smoothing_time)
		return self.smoothers[label].smooth(hand)

	def smoothe_dict(self, hands: dict[str, HandState]):
		smoothed = {}
		for label, hand in hands.items():
			smoothed[label] = self.smoothe(label, hand)
		return smoothed

class HandSmoother:
	def __init__(self, smoothing_time: float):
		"""
		smoothing_time: window in seconds (e.g., 0.12 for 120ms)
		"""
		self.smoothing_time = smoothing_time
		self._window = deque()  # Each entry: (timestamp, HandState)

	def add(self, hand: HandState, timestamp: Optional[float] = None):
		"""
		Add a HandState with an optional timestamp (defaults to time.time()).
		"""
		if timestamp is None:
			timestamp = time.time()
		self._window.append((timestamp, hand))

	def _prune(self, now: float):
		cutoff = now - self.smoothing_time
		while self._window and self._window[0][0] < cutoff:
			self._window.popleft()

	def get_smoothed(self) -> Optional[HandState]:
		"""
		Return a HandState with averaged landmarks over the window, or None if window is empty.
		"""
		if not self._window:
			return None
		

		avg_lms = []
		# Compute the average for each landmark
		for i in range(len(self._window[-1][1].landmarks)):
			avg_lm = None
			for t, hand in reversed(self._window):
				if avg_lm is None:
					avg_lm = hand.landmarks[i]*3.0
				else:
					avg_lm += hand.landmarks[i]
			avg_lm /= len(self._window) + 2
			avg_lms.append(avg_lm)
			
		# Return a new HandState with the averaged landmarks
		return HandState(
			label=self._window[-1][1].label,
			landmarks=avg_lms,
			palm_width=palm_width(avg_lms)
		)
	def smooth(self, hand: HandState, timestamp: Optional[float] = None) -> Optional[HandState]:
		"""
		Add a new HandState and return the smoothed version.
		"""
		now = time.time() if timestamp is None else timestamp
		self.add(hand, now)
		self._prune(now)
		return self.get_smoothed()

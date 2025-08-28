import time

class Gate:
    def __init__(self, input_feature, op=">", trigger_pct=0.5, release_pct=0.45, refractory_ms=120, lost_hand_policy="release"):
        self.input_feature = input_feature  # Feature instance
        self.op = op
        self.trigger_pct = trigger_pct
        self.release_pct = release_pct
        self.refractory_ms = refractory_ms
        self.lost_hand_policy = lost_hand_policy  # 'release'|'hold'|'true'|'toggle'
        self.state = False
        self.t_last = 0
        self._last_value = None
        self._last_tracked = False
        self._last_time = 0

    def probe_last_state(self):
        return {
            'state': self.state,
            'feature': self.input_feature.probe_last_value(),
        }

    def getState(self, hand_left, hand_right):
        """
        hand_left, hand_right: HandState or None
        Returns: bool (gate open/closed)
        """
        now_ms = int(time.time() * 1000)
        # Try to get value from input_feature (may use left, right, or both)
        val = None
        try:
            if hasattr(self.input_feature, 'getValue'):
                val = self.input_feature.getValue(hand_left, hand_right)
        except Exception as e:
            val = None

        tracked = val is not None
        self._last_value = val
        self._last_tracked = tracked
        self._last_time = now_ms
        # Lost hand policy
        if not tracked:
            if self.lost_hand_policy == "hold":
                return self.state
            elif self.lost_hand_policy == "true":
                self.state = True
                return True
            elif self.lost_hand_policy == "toggle":
                if now_ms - self.t_last > self.refractory_ms:
                    self.state = not self.state
                    self.t_last = now_ms
                return self.state
            else:  # release
                self.state = False
                return False

        v = float(val)
        desired = (v > self.trigger_pct) if self.op == ">" else (v < self.trigger_pct)
        release_ok = (v < self.release_pct) if self.op == ">" else (v > self.release_pct)

        if not self.state:
            if desired and (now_ms - self.t_last) > self.refractory_ms:
                self.state = True
                self.t_last = now_ms
        else:
            if release_ok and (now_ms - self.t_last) > self.refractory_ms:
                self.state = False
                self.t_last = now_ms
        return self.state
    
class GateBuilder:
    def __init__(self, feature_index):
        self.feature_index = feature_index

    def build(self, gate_cfg):
        # gate_cfg must contain 'input' key specifying the feature name
        if not gate_cfg:
            return None
        feature_name = gate_cfg.get('input')
        feature = self.feature_index.getFeature(feature_name)
        return Gate(
            input_feature=feature,
            op=gate_cfg.get('op', '>'),
            trigger_pct=float(gate_cfg.get('trigger_pct', 0.5)),
            release_pct=float(gate_cfg.get('release_pct', 0.45)),
            refractory_ms=int(gate_cfg.get('refractory_ms', 120)),
            lost_hand_policy=gate_cfg.get('lost_hand_policy', 'release'),
        )
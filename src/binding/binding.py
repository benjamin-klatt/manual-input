# --- BindingIndex ---
class BindingIndex:
	def __init__(self, config, feature_index, actuator_builder, gate_builder):
		self.bindings = []
		bindings_cfg = config.get('bindings', [])
		for binding_cfg in bindings_cfg:
			binding = BindingBuilder.build(binding_cfg, feature_index, actuator_builder, gate_builder)
			self.bindings.append(binding)

	def update(self, left_hand, right_hand):
		for binding in self.bindings:
			binding.update(left_hand, right_hand)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from src.outputs.actuators import Actuator, EventActuator, DeltaActuator, AbsActuator
	from src.gate.gate import Gate
	from src.input.features import Feature

# --- Binding base classes ---
class Binding:
	def __init__(self, feature: 'Feature', gate: 'Gate', actuator: 'Actuator', id=None):
		self.feature = feature
		self.gate = gate
		self.actuator = actuator
		self.id = id
		self._last_state = None
		self._last_value = None
		self._last_time = None

	def probe_last(self):
		return {
			'feature': self.feature.probe_last_value() if self.feature else None,
			'gate': self.gate.probe_last_state() if self.gate else None,
			'actuator': self.actuator.probe_last() if self.actuator else None,
			'binding_state': self._last_state,
			'binding_value': self._last_value,
			'binding_time': self._last_time
		}

	def update(self, left_hand, right_hand):
		raise NotImplementedError

class EventBinding(Binding):
	def __init__(self, feature: 'Feature', gate: 'Gate', actuator, event_type=None, id=None,
				 trigger_pct=None, release_pct=None, refractory_ms=None, op=None):
		super().__init__(feature, gate, actuator, id=id)
		self.event_type = event_type  # e.g., 'down', 'up', etc.
		self.prev_state = False
		self._last_transition_time = 0
		self._custom_trigger_pct = trigger_pct
		self._custom_release_pct = release_pct
		self._custom_refractory_ms = refractory_ms
		self._custom_op = op

	def update(self, left_hand, right_hand):
		import time
		now_ms = int(time.time() * 1000)
		value = self.feature.getValue(left_hand, right_hand) if self.feature else None
		gate_state = self.gate.getState(left_hand, right_hand) if self.gate else True
		# Use only config or defaults for thresholds and op
		trigger_pct = self._custom_trigger_pct if self._custom_trigger_pct is not None else 0.5
		release_pct = self._custom_release_pct if self._custom_release_pct is not None else 0.45
		refractory_ms = self._custom_refractory_ms if self._custom_refractory_ms is not None else 120
		op = self._custom_op if self._custom_op is not None else '>'
		now_state = False
		if gate_state:
			if value is not None:
				v = float(value)
				if not self.prev_state:
					desired = (v > trigger_pct) if op == '>' else (v < trigger_pct)
					if desired and (now_ms - self._last_transition_time) > refractory_ms:
						now_state = True
						self._last_transition_time = now_ms
				else:
					release_ok = (v < release_pct) if op == '>' else (v > release_pct)
					if not release_ok:
						now_state = True
					else:
						if (now_ms - self._last_transition_time) > refractory_ms:
							now_state = False
							self._last_transition_time = now_ms
			# else: feature lost, treat as released
		# else: gate is False, treat as lost hand (released)
		self._last_state = now_state
		self._last_value = value
		self._last_time = None
		if now_state and not self.prev_state:
			if hasattr(self.actuator, 'trigger_actuator') or hasattr(self.actuator, 'release_actuator'):
				self.actuator.trigger('down')
			else:
				self.actuator.trigger()
			self._last_time = 'down'
		elif not now_state and self.prev_state:
			if hasattr(self.actuator, 'trigger_actuator') or hasattr(self.actuator, 'release_actuator'):
				self.actuator.trigger('up')
			self._last_time = 'up'
		self.prev_state = now_state

class DeltaBinding(Binding):
	def __init__(self, feature: 'Feature', gate: 'Gate', actuator: 'DeltaActuator', scale=1.0, deadzone=0.0, id=None):
		super().__init__(feature, gate, actuator, id=id)
		self.scale = scale
		self.deadzone = deadzone

	def update(self, left_hand, right_hand):
		if not self.gate.getState(left_hand, right_hand):
			self._last_state = False
			self._last_value = None
			return
		value = self.feature.getValue(left_hand, right_hand)
		if value is None:
			self._last_state = True
			self._last_value = None
			return
		if abs(value) > self.deadzone:
			delta = value * self.scale
			self.actuator.trigger(delta)
			self._last_value = delta
		self._last_state = True

class AbsBinding(Binding):
	def __init__(self, feature: 'Feature', gate: 'Gate', actuator: 'AbsActuator', min_value=0.0, max_value=1.0, id=None):
		super().__init__(feature, gate, actuator, id=id)
		self.min_value = min_value
		self.max_value = max_value

	def update(self, left_hand, right_hand):
		if not self.gate.getState(left_hand, right_hand):
			self._last_state = False
			self._last_value = None
			return
		value = self.feature.getValue(left_hand, right_hand)
		if value is None:
			self._last_state = True
			self._last_value = None
			return
		scaled = self.min_value + (self.max_value - self.min_value) * value
		self.actuator.trigger(scaled)
		self._last_state = True
		self._last_value = scaled

# --- BindingBuilder ---
class BindingBuilder:
	@staticmethod
	def build(config: dict, feature_index, actuator_builder, gate_builder):
		# Determine binding type from actuator or config
		actuator_cfg = config.get('actuator')
		actuator = actuator_builder.build(actuator_cfg)

		# Type check for kind
		from src.outputs.actuators import EventActuator, DeltaActuator, AbsActuator, ActuatorPair
		kind = config.get('type')
		if kind is None:
			if isinstance(actuator, ActuatorPair):
				kind = 'event'
			elif isinstance(actuator, EventActuator):
				kind = 'event'
			elif isinstance(actuator, DeltaActuator):
				kind = 'delta'
			elif isinstance(actuator, AbsActuator):
				kind = 'abs'
			else:
				kind = 'event'

		# Feature/input
		feature_name = config.get('input') or config.get('feature')
		feature = feature_index.getFeature(feature_name)

		# Gate (optional)
		gate_cfg = config.get('gate')
		gate = gate_builder.build(gate_cfg)

		# Parameters for delta/abs
		scale = config.get('sensitivity', config.get('scale', 1.0))
		deadzone = config.get('deadzone', 0.0)
		min_value = config.get('min', 0.0)
		max_value = config.get('max', 1.0)

		id_val = config.get('id')
		if kind == 'event':
			return EventBinding(
				feature, gate, actuator,
				event_type=config.get('event_type'),
				id=id_val,
				trigger_pct=config.get('trigger_pct'),
				release_pct=config.get('release_pct'),
				refractory_ms=config.get('refractory_ms'),
				op=config.get('op')
			)
		elif kind == 'delta':
			return DeltaBinding(
				feature, gate, actuator,
				scale=scale,
				deadzone=deadzone,
				id=id_val
			)
		elif kind == 'abs':
			return AbsBinding(
				feature, gate, actuator,
				min_value=min_value,
				max_value=max_value,
				id=id_val
			)
		else:
			raise ValueError(f"Unknown binding type: {kind}")



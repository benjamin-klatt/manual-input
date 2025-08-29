
from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyboardController, Key
from .mouse import mouse_out

# --- ActuatorPair for event actuators ---
class ActuatorPair:
    def __init__(self, trigger_actuator, release_actuator=None):
        self.trigger_actuator = trigger_actuator
        self.release_actuator = release_actuator
    def probe_last(self):
        return {
            'trigger': self.trigger_actuator.probe_last() if self.trigger_actuator else None,
            'release': self.release_actuator.probe_last() if self.release_actuator else None
        }
    def trigger(self, event_type=None):
        if event_type == 'down' or event_type == 'trigger' or event_type is None:
            if self.trigger_actuator:
                self.trigger_actuator.trigger()
        elif event_type == 'up' or event_type == 'release':
            if self.release_actuator:
                self.release_actuator.trigger()

# --- Actuator base classes ---
class Actuator:
    def __init__(self, target):
        self.target = target
        self._last_trigger_time = None
        self._last_value = None
    def probe_last(self):
        return {
            'last_trigger_time': self._last_trigger_time,
            'last_value': self._last_value
        }
    def trigger(self, *args, **kwargs):
        raise NotImplementedError

class EventActuator(Actuator):
    pass

class DeltaActuator(Actuator):
    def trigger(self, value):
        import time
        self._last_trigger_time = time.time()
        self._last_value = value
        raise NotImplementedError

class AbsActuator(Actuator):
    def trigger(self, value):
        import time
        self._last_trigger_time = time.time()
        self._last_value = value
        raise NotImplementedError

# --- Mouse actuators ---
class MouseMoveDeltaActuator(DeltaActuator):
    def __init__(self, axis):
        super().__init__(axis)
        self.mouse = mouse_out
        self.axis = axis  # 'x' or 'y'
    def trigger(self, value):
        import time
        self._last_trigger_time = time.time()
        self._last_value = value
        if self.axis == 'x':
            self.mouse.move_dx(value)
        elif self.axis == 'y':
            self.mouse.move_dy(value)

class MouseScrollDeltaActuator(DeltaActuator):
    def __init__(self, axis):
        super().__init__(axis)
        self.mouse = mouse_out
        self.axis = axis  # 'x' or 'y'
    def trigger(self, value):
        import time
        self._last_trigger_time = time.time()
        self._last_value = value
        if self.axis == 'x':
            self.mouse.scroll(value, 0)
        elif self.axis == 'y':
            self.mouse.scroll(0, value)

class MouseAbsActuator(AbsActuator):
    def __init__(self, axis):
        super().__init__(axis)
        self.mouse = mouse_out
        self.axis = axis  # 'x' or 'y'
    def trigger(self, value):
        import time
        self._last_trigger_time = time.time()
        self._last_value = value
        # NOTE: MouseOut does not support absolute positioning; this is a placeholder
        pass

class MouseButtonEventActuator(EventActuator):
    def __init__(self, button, event):
        super().__init__((button, event))
        self.mouse = mouse_out
        self.button = button  # Button.left, Button.right, etc.
        self.event = event    # 'down' or 'up'
    def trigger(self):
        if self.event == 'down':
            self.mouse.down(self.button)
        elif self.event == 'up':
            self.mouse.up(self.button)

# --- Keyboard actuators ---
class KeyboardKeyEventActuator(EventActuator):
    def __init__(self, key, event):
        super().__init__((key, event))
        self.keyboard = KeyboardController()
        self.key = key  # Key or str
        self.event = event  # 'down' or 'up'
    def trigger(self):
        if self.event == 'down':
            self.keyboard.press(self.key)
        elif self.event == 'up':
            self.keyboard.release(self.key)

class KeyboardDeltaActuator(DeltaActuator):
    def __init__(self, key):
        super().__init__(key)
        self.keyboard = KeyboardController()
        self.key = key
    def trigger(self, value):
        import time
        self._last_trigger_time = time.time()
        self._last_value = value
        # Example: could be used for volume up/down, etc.
        pass

class KeyboardAbsActuator(AbsActuator):
    def __init__(self, key):
        super().__init__(key)
        self.keyboard = KeyboardController()
        self.key = key
    def trigger(self, value):
        import time
        self._last_trigger_time = time.time()
        self._last_value = value
        # Example: could be used for setting a value, e.g., brightness
        pass
    def probe_last(self):
        return {
            'trigger': getattr(self.trigger_actuator, 'probe_last', lambda: None)(),
            'release': getattr(self.release_actuator, 'probe_last', lambda: None)()
        }

# --- ActuatorBuilder ---
class ActuatorBuilder:
    @staticmethod
    def build(key, **kwargs):
        # If key is a dict, build ActuatorPair
        if isinstance(key, dict):
            trigger = key.get('trigger')
            release = key.get('release')
            trigger_act = ActuatorBuilder.build(trigger) if trigger else None
            release_act = ActuatorBuilder.build(release) if release else None
            return ActuatorPair(trigger_act, release_act)

        # Mouse move delta
        if key == 'mouse.move.x':
            return MouseMoveDeltaActuator('x')
        if key == 'mouse.move.y':
            return MouseMoveDeltaActuator('y')
        # Mouse scroll delta
        if key == 'mouse.scroll.x':
            return MouseScrollDeltaActuator('x')
        if key == 'mouse.scroll.y':
            return MouseScrollDeltaActuator('y')
        # Mouse button events (down/up)
        import re
        m_event = re.match(r"mouse\.click\.(\w+)\.(down|up)$", key)
        if m_event:
            btn = m_event.group(1)
            action = m_event.group(2)
            return MouseButtonEventActuator(btn, action)

        # Mouse button ActuatorPair for generic mouse.click.*
        m_pair = re.match(r"mouse\.click\.(\w+)$", key)
        if m_pair:
            btn = m_pair.group(1)
            return ActuatorPair(
                MouseButtonEventActuator(btn, 'down'),
                MouseButtonEventActuator(btn, 'up')
            )

        # Keyboard key events (down/up)
        if isinstance(key, str) and key.startswith('key.'):
            parts = key.split('.')
            # key.<name>.down or key.<name>.up
            if len(parts) == 3:
                keyname, event = parts[1], parts[2]
                k = getattr(Key, keyname, keyname)
                return KeyboardKeyEventActuator(k, event)
            # key.<name> → ActuatorPair(key.<name>.down, key.<name>.up)
            if len(parts) == 2:
                keyname = parts[1]
                k = getattr(Key, keyname, keyname)
                return ActuatorPair(
                    KeyboardKeyEventActuator(k, 'down'),
                    KeyboardKeyEventActuator(k, 'up')
                )
        # Keyboard delta/abs (custom, placeholder)
        if isinstance(key, str) and key.startswith('key.delta.'):
            keyname = key.split('.')[-1]
            return KeyboardDeltaActuator(keyname)
        if isinstance(key, str) and key.startswith('key.abs.'):
            keyname = key.split('.')[-1]
            return KeyboardAbsActuator(keyname)
        # Mouse abs (not implemented)
        if key == 'mouse.pos.x':
            return MouseAbsActuator('x')
        if key == 'mouse.pos.y':
            return MouseAbsActuator('y')
        raise ValueError(f"Unknown actuator key: {key}")
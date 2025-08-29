# Mouse move/pos/scroll and buttons: platform-dependent backend
import sys
import os

_PLATFORM = sys.platform

# --- Backend flags ---
_USE_SENDINPUT = False
_USE_UINPUT = False
_USE_QUARTZ = False
_UInputDevice = None
_UINPUT_EVENTS = None

# --- Windows: SendInput ---
if _PLATFORM.startswith('win'):
    try:
        import ctypes
        from ctypes import wintypes
        _USE_SENDINPUT = True
    except Exception:
        _USE_SENDINPUT = False

# --- Linux: python-uinput ---
if _PLATFORM.startswith('linux'):
    try:
        import uinput
        _UInputDevice = uinput.Device
        _UINPUT_EVENTS = [uinput.REL_X, uinput.REL_Y, uinput.BTN_LEFT, uinput.BTN_RIGHT, uinput.BTN_MIDDLE, uinput.REL_WHEEL]
        _USE_UINPUT = True
    except ImportError:
        _USE_UINPUT = False

# --- macOS: PyObjC + Quartz ---
if _PLATFORM == 'darwin':
    try:
        import Quartz
        from Quartz.CoreGraphics import (
            CGEventCreateMouseEvent, CGEventPost, kCGHIDEventTap,
            kCGEventMouseMoved, kCGEventLeftMouseDown, kCGEventLeftMouseUp,
            kCGEventRightMouseDown, kCGEventRightMouseUp,
            kCGEventScrollWheel, kCGMouseButtonLeft, kCGMouseButtonRight
        )
        _USE_QUARTZ = True
    except Exception:
        _USE_QUARTZ = False

# --- Fallback: pynput ---
if not (_USE_SENDINPUT or _USE_UINPUT or _USE_QUARTZ):
    from pynput.mouse import Button, Controller as MouseController





# --- Platform-specific MouseOut subclasses ---
class SendInputMouseOut:
    def move_dx(self, dx):
        self._sendinput_move(int(dx), 0)
    def move_dy(self, dy):
        self._sendinput_move(0, int(dy))
    def scroll(self, dx_ticks, dy_ticks):
        self._sendinput_scroll(dx_ticks, dy_ticks)
    def down(self, button):
        self._sendinput_button(button, True)
    def up(self, button):
        self._sendinput_button(button, False)
    def _sendinput_move(self, dx, dy):
        import ctypes
        from ctypes import wintypes
        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]
        class INPUT(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("mi", MOUSEINPUT)]
        MOUSEEVENTF_MOVE = 0x0001
        inp = INPUT(type=0, mi=MOUSEINPUT(dx, dy, 0, MOUSEEVENTF_MOVE, 0, None))
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
    def _sendinput_button(self, button, down):
        import ctypes
        from ctypes import wintypes
        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]
        class INPUT(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("mi", MOUSEINPUT)]
        flags = 0
        if button == 'left':
            flags = 0x0002 if down else 0x0004
        elif button == 'right':
            flags = 0x0008 if down else 0x0010
        else:
            flags = 0x0020 if down else 0x0040
        inp = INPUT(type=0, mi=MOUSEINPUT(0, 0, 0, flags, 0, None))
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
    def _sendinput_scroll(self, dx, dy):
        import ctypes
        from ctypes import wintypes
        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]
        class INPUT(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("mi", MOUSEINPUT)]
        if dy:
            inp = INPUT(type=0, mi=MOUSEINPUT(0, 0, int(dy)*120, 0x0800, 0, None))
            ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
        if dx:
            inp = INPUT(type=0, mi=MOUSEINPUT(0, 0, int(dx)*120, 0x1000, 0, None))
            ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

class UInputMouseOut:
    def __init__(self):
        self.device = _UInputDevice(_UINPUT_EVENTS)
    def move_dx(self, dx):
        self.device.emit(uinput.REL_X, int(dx), syn=False)
    def move_dy(self, dy):
        self.device.emit(uinput.REL_Y, int(dy), syn=False)
    def scroll(self, dx_ticks, dy_ticks):
        if dx_ticks:
            try:
                self.device.emit(uinput.REL_HWHEEL, int(dx_ticks), syn=False)
            except AttributeError:
                pass
        if dy_ticks:
            self.device.emit(uinput.REL_WHEEL, int(dy_ticks), syn=False)
        self.device.syn()
    def down(self, button):
        if button == 'left':
            self.device.emit(uinput.BTN_LEFT, 1)
        elif button == 'right':
            self.device.emit(uinput.BTN_RIGHT, 1)
        else:
            self.device.emit(uinput.BTN_MIDDLE, 1)
    def up(self, button):
        if button == 'left':
            self.device.emit(uinput.BTN_LEFT, 0)
        elif button == 'right':
            self.device.emit(uinput.BTN_RIGHT, 0)
        else:
            self.device.emit(uinput.BTN_MIDDLE, 0)

class QuartzMouseOut:
    def move_dx(self, dx):
        self._quartz_move(int(dx), 0)
    def move_dy(self, dy):
        self._quartz_move(0, int(dy))
    def scroll(self, dx_ticks, dy_ticks):
        self._quartz_scroll(dx_ticks, dy_ticks)
    def down(self, button):
        self._quartz_button(button, True)
    def up(self, button):
        self._quartz_button(button, False)
    def _quartz_move(self, dx, dy):
        import Quartz
        loc = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
        newx, newy = loc.x + dx, loc.y + dy
        event = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventMouseMoved, (newx, newy), Quartz.kCGMouseButtonLeft)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
    def _quartz_button(self, button, down):
        import Quartz
        loc = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
        btn = Quartz.kCGMouseButtonLeft if button == 'left' else Quartz.kCGMouseButtonRight if button == 'right' else 2
        event_type = (Quartz.kCGEventLeftMouseDown if down else Quartz.kCGEventLeftMouseUp) if button == 'left' else (Quartz.kCGEventRightMouseDown if down else Quartz.kCGEventRightMouseUp)
        event = Quartz.CGEventCreateMouseEvent(None, event_type, (loc.x, loc.y), btn)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
    def _quartz_scroll(self, dx, dy):
        import Quartz
        event = Quartz.CGEventCreateScrollWheelEvent(None, Quartz.kCGScrollEventUnitLine, 2, int(dy), int(dx))
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

class PynputMouseOut:
    def __init__(self):
        self.mouse = MouseController()
        self.buttons = {
            'left': Button.left,
            'right': Button.right,
            'middle': Button.middle
        }
    def move_dx(self, dx):
        self.mouse.move(int(dx), 0)
    def move_dy(self, dy):
        self.mouse.move(0, int(dy))
    def scroll(self, dx_ticks, dy_ticks):
        self.mouse.scroll(int(dx_ticks), int(dy_ticks))
    def down(self, button):
        self.mouse.press(self.buttons.get(button))
    def up(self, button):
        self.mouse.release(self.buttons.get(button))

# --- MouseOut factory/delegator ---
class MouseOut:
    def __init__(self):
        if _USE_SENDINPUT:
            self._impl = SendInputMouseOut()
        elif _USE_UINPUT:
            self._impl = UInputMouseOut()
        elif _USE_QUARTZ:
            self._impl = QuartzMouseOut()
        else:
            self._impl = PynputMouseOut()
    def move_dx(self, dx):
        self._impl.move_dx(dx)
    def move_dy(self, dy):
        self._impl.move_dy(dy)
    def scroll(self, dx_ticks, dy_ticks):
        self._impl.scroll(dx_ticks, dy_ticks)
    def down(self, button):
        self._impl.down(button)
    def up(self, button):
        self._impl.up(button)


def button_from_kind(kind: str):
    # Always return 'left'/'right'/'middle' for all backends except pynput
    if _USE_SENDINPUT or _USE_UINPUT or _USE_QUARTZ:
        if "left" in kind: return 'left'
        if "right" in kind: return 'right'
        return 'middle'
    else:
        if "left" in kind: return Button.left
        if "right" in kind: return Button.right
        return Button.middle

mouse_out = MouseOut()
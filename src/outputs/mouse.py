# Mouse move/pos/scroll and buttons (pynput) (placeholder)
from pynput.mouse import Button, Controller as MouseController


class MouseOut:
    def __init__(self):
        self.mouse = MouseController()
    def move_dx(self, dx):
        self.mouse.move(int(dx), 0)
    def move_dy(self, dy):
        self.mouse.move(0, int(dy))
    def scroll(self, dx_ticks, dy_ticks):
        # pynput expects number of steps; dy positive is up on some OSes
        self.mouse.scroll(int(dx_ticks), int(dy_ticks))
    def down(self, button):
        self.mouse.press(button)
    def up(self, button):
        self.mouse.release(button)


def button_from_kind(kind: str) -> Button:
    if "left" in kind: return Button.left
    if "right" in kind: return Button.right
    return Button.middle
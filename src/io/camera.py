import os
import platform
from typing import Optional

import cv2


class CameraSwitcher:
    def __init__(self, width=640, height=480, fps=30, max_search=20,
                 preferred_backend: Optional[int]=None):
        self.width, self.height, self.fps = width, height, fps
        self.max_search = max_search
        self.cap = None
        self.idx = None
        self.backend = preferred_backend
        self.backends = self._default_backends_for_os()

        # Linux: map /sys/class/video4linux names and /dev/v4l/by-id symlinks
        self.linux_index_name = self._linux_video_names() if self._is_linux() else {}
        self.linux_id_to_index = self._linux_id_to_index() if self._is_linux() else {}

    def _is_linux(self): return platform.system() == "Linux"
    def _is_windows(self): return platform.system() == "Windows"
    def _is_macos(self): return platform.system() == "Darwin"

    def _default_backends_for_os(self):
        if self._is_windows():
            # Try Media Foundation first, then DirectShow
            return [getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
                    getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
                    cv2.CAP_ANY]
        if self._is_macos():
            return [getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY),
                    cv2.CAP_ANY]
        # Linux
        return [getattr(cv2, "CAP_V4L2", cv2.CAP_ANY), cv2.CAP_ANY]

    def _linux_video_names(self):
        out = {}
        try:
            base = "/sys/class/video4linux"
            if not os.path.isdir(base):
                return out
            for entry in os.listdir(base):
                if not entry.startswith("video"):
                    continue
                idx = int(entry.replace("video", ""))
                name_file = os.path.join(base, entry, "name")
                if os.path.exists(name_file):
                    with open(name_file, "r", encoding="utf-8", errors="ignore") as f:
                        out[idx] = f.read().strip()
        except Exception:
            pass
        return out

    def _linux_id_to_index(self):
        out = {}
        try:
            byid = "/dev/v4l/by-id"
            if not os.path.isdir(byid):
                return out
            for link in os.listdir(byid):
                path = os.path.join(byid, link)
                try:
                    real = os.path.realpath(path)
                    if "/video" in real:
                        idx = int(os.path.basename(real).replace("video", ""))
                        out[path] = idx  # store full by-id path
                except Exception:
                    continue
        except Exception:
            pass
        return out

    def _open_cap(self, index, backend):
        cap = cv2.VideoCapture(index, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        ok, _ = cap.read()
        if not ok:
            cap.release()
            return None
        return cap

    def enumerate(self):
        found = []
        tried = []
        backends = [self.backend] + self.backends if self.backend else self.backends
        seen_pairs = set()
        for b in backends:
            for i in range(self.max_search):
                if (i, b) in seen_pairs:
                    continue
                seen_pairs.add((i, b))
                tried.append((i, b))
                cap = self._open_cap(i, b)
                if cap:
                    name = self.linux_index_name.get(i) if self._is_linux() else None
                    found.append({"index": i, "backend": b, "name": name})
                    cap.release()
        # Diagnostics
        def bname(b):
            for k, v in vars(cv2).items():
                if k.startswith("CAP_") and v == b:
                    return k
            return str(b)
        if not found:
            print("[camera] No cameras found.")
            print("[camera] Tried:")
            for i, b in tried:
                print(f"  - index {i} via {bname(b)}")
            print("[camera] Tips:")
            if self._is_windows():
                print("  • Try CAP_DSHOW vs CAP_MSMF; check Windows Camera privacy settings.")
            elif self._is_macos():
                print("  • Ensure your terminal/IDE has Camera permission (System Settings → Privacy).")
            else:
                print("  • Check /dev/video* permissions (user in 'video' group).")
                print("  • Close apps using the camera; verify with: fuser /dev/video*")
        else:
            print("[camera] Found devices:")
            for d in found:
                print(f"  - index {d['index']} via {bname(d['backend'])}"
                      + (f" name='{d['name']}'" if d.get("name") else ""))
        return found

    def open(self, preferred_index=None):
        # Respect preferred backend first if given
        backends = [self.backend] + self.backends if self.backend else self.backends
        if preferred_index is not None:
            for b in backends:
                cap = self._open_cap(preferred_index, b)
                if cap:
                    self.cap, self.idx, self.backend = cap, preferred_index, b
                    print(f"[camera] Opened index {preferred_index} via {self._backend_name(b)}")
                    return self.cap, self.idx
        # Fallback: scan
        devices = self.enumerate()
        for d in devices:
            cap = self._open_cap(d["index"], d["backend"])
            if cap:
                self.cap, self.idx, self.backend = cap, d["index"], d["backend"]
                print(f"[camera] Opened index {self.idx} via {self._backend_name(self.backend)}")
                return self.cap, self.idx
        self.cap, self.idx = None, None
        return self.cap, self.idx

    def open_by_linux_id_or_name(self, dev_id: Optional[str], name: Optional[str]):
        if not self._is_linux():
            return self.open(None)
        # Try stable by-id path
        if dev_id and dev_id in self.linux_id_to_index:
            idx = self.linux_id_to_index[dev_id]
            return self.open(idx)
        # Try name match
        if name:
            for idx, nm in self.linux_index_name.items():
                if nm == name:
                    return self.open(idx)
        # Fallback
        return self.open(None)

    def next(self):
        devs = self.enumerate()
        if not devs: return self.open(None)
        indices = [d["index"] for d in devs]
        if self.idx is None: return self.open(indices[0])
        try:
            j = indices.index(self.idx)
        except ValueError:
            j = -1
        # Try same backend on next index, then fall back
        for k in range(1, len(indices)+1):
            i2 = indices[(j + k) % len(indices)]
            cap = self._open_cap(i2, self.backend or cv2.CAP_ANY)
            if cap:
                if self.cap: self.cap.release()
                self.cap, self.idx = cap, i2
                print(f"[camera] Switched to index {self.idx} via {self._backend_name(self.backend or cv2.CAP_ANY)}")
                return self.cap, self.idx
        return self.open(None)

    def prev(self):
        devs = self.enumerate()
        if not devs: return self.open(None)
        indices = [d["index"] for d in devs]
        if self.idx is None: return self.open(indices[0])
        try:
            j = indices.index(self.idx)
        except ValueError:
            j = -1
        for k in range(1, len(indices)+1):
            i2 = indices[(j - k) % len(indices)]
            cap = self._open_cap(i2, self.backend or cv2.CAP_ANY)
            if cap:
                if self.cap: self.cap.release()
                self.cap, self.idx = cap, i2
                print(f"[camera] Switched to index {self.idx} via {self._backend_name(self.backend or cv2.CAP_ANY)}")
                return self.cap, self.idx
        return self.open(None)

    def handle_key(self, key_ascii):
        if key_ascii in (ord(']'), 9):      # ] or TAB
            return self.next()
        elif key_ascii == ord('['):
            return self.prev()
        elif key_ascii in (ord('r'), ord('R')):
            return self.open(None)  # rescan and open first working
        return self.cap, self.idx

    def _backend_name(self, b):
        for k, v in vars(cv2).items():
            if k.startswith("CAP_") and v == b:
                return k
        return str(b)

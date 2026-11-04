"""
Regenerate architecture.svg and pin_diagram.svg.

    python scripts/draw_diagrams.py

Standard library only. To refresh the PNGs, open each SVG in a browser and screenshot it at its full size.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

FONT = "'Segoe UI', Inter, Arial, sans-serif"
ROOT = Path(__file__).resolve().parents[1]

AMBER = ("#fdf3e1", "#c8902e", "#6b4a0e")
GREEN = ("#e3f4ea", "#2e8b57", "#1b4d33")
BLUE = ("#e8f0fb", "#2f6db5", "#1d3f6e")
PURPLE = ("#eeebfb", "#6a55c9", "#3b2f7a")
GRAY = ("#efeee9", "#8a8a84", "#333333")
RED, BLACK, MUTED = "#d64545", "#2b2b2b", "#6b6b6b"


class Svg:
    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.parts = [f'<rect width="{width}" height="{height}" fill="white"/>',
                      '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">'
                      '<path d="M0,0 L9,4.5 L0,9 Z" fill="#555"/></marker></defs>']

    def text(self, x, y, value, size=20, fill="#222", weight="normal", anchor="middle", family=FONT):
        self.parts.append(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" fill="{fill}" '
                          f'font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>')

    def rect(self, x, y, w, h, fill="none", stroke="#555", width=2.5, rx=14, dash=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="{width}"{extra}/>')

    def line(self, x1, y1, x2, y2, stroke="#555", width=3, arrow=False, dash=None):
        extra = (' marker-end="url(#arrow)"' if arrow else "") + (f' stroke-dasharray="{dash}"' if dash else "")
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
                          f'stroke-width="{width}"{extra}/>')

    def path(self, d, stroke, width=4):
        self.parts.append(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round"/>')

    def circle(self, x, y, r, fill, stroke="#8a6a1a", width=1.5):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def box(self, x, y, w, h, title, lines, palette, title_size=30):
        fill, stroke, ink = palette
        self.rect(x, y, w, h, fill, stroke)
        total = title_size + 8 + 30 * len(lines)
        top = y + (h - total) / 2 + title_size
        self.text(x + w / 2, top, title, title_size, ink, "600")
        for i, value in enumerate(lines):
            self.text(x + w / 2, top + 38 + i * 30, value, 21, stroke if palette is not GRAY else "#666")

    def write(self, path: Path) -> None:
        head = f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" ' \
               f'viewBox="0 0 {self.width} {self.height}">'
        path.write_text("\n".join([head, *self.parts, "</svg>", ""]), encoding="utf-8", newline="\n")
        print("wrote", path)


def architecture() -> None:
    s = Svg(1800, 2060)
    s.text(900, 90, "GestureGrasp: system architecture", 46, weight="700")
    s.box(70, 130, 1660, 130, "User hand gesture", ["Call · point · thumbs up/down · stop · peace · OK, with nothing to wear"], GRAY)
    s.line(900, 260, 900, 330, arrow=True)

    s.rect(70, 335, 1660, 215, stroke="#999", dash="12 8")
    s.text(105, 378, "ESP32-CAM", 24, "#333", anchor="start")
    s.text(245, 378, "AI-Thinker · runs only the firmware in esp32_cam/", 22, "#888", anchor="start")
    s.box(115, 405, 765, 125, "OV2640 camera", ["640x480 JPEG, always the newest frame"], AMBER)
    s.box(920, 405, 765, 125, "HTTP servers", [":81/stream for the Pi only · :80 /status /control /capture"], AMBER)
    s.line(900, 550, 900, 655, arrow=True)
    s.text(925, 595, "MJPEG over Wi-Fi", 23, "#333", anchor="start")
    s.text(925, 628, "2.4 GHz, one client: the Pi", 21, "#666", anchor="start")

    s.rect(70, 660, 1660, 1040, stroke="#999", dash="12 8")
    s.text(105, 703, "Raspberry Pi 5", 24, "#333", anchor="start")
    s.text(290, 703, "runs everything · CPU only · python -m server", 22, "#888", anchor="start")
    s.box(115, 725, 1570, 120, "Capture thread", ["Keeps only the latest frame, reconnects by itself, frames it skips are never decoded"], GREEN)
    s.line(495, 845, 495, 885, arrow=True)
    s.line(1305, 845, 1305, 885, arrow=True)
    s.box(115, 890, 765, 120, "Hand landmarks", ["MediaPipe, 21 points per hand"], GREEN)
    s.line(495, 1010, 495, 1050, arrow=True)
    s.box(115, 1055, 765, 130, "Gesture classifier", ["126 features → MLP, 9 classes, NumPy only"], GREEN)
    s.line(495, 1185, 495, 1225, arrow=True)
    s.box(115, 1230, 765, 130, "Intent engine", ["EMA smoothing, confidence + hold time, STOP has priority"], BLUE)
    s.box(920, 890, 765, 150, "Object detection", ["YOLO11n on its own thread at 3 Hz, imgsz 320", "The gesture loop never waits for it"], GREEN)
    s.line(1305, 1040, 1305, 1105, arrow=True)
    s.text(1330, 1080, "last boxes + index-finger ray", 21, BLUE[1], anchor="start")
    s.box(920, 1110, 765, 250, "Pointing + target", ["Ray → nearest object within 15°", "stable vote over the hold"], BLUE)
    s.path("M495,1360 V1395 Q495,1405 505,1405 H1295 Q1305,1405 1305,1395 V1360", "#555", 3)
    s.line(900, 1405, 900, 1430, arrow=True)
    s.box(115, 1435, 1570, 115, "State machine", ["IDLE → LISTENING → TARGET_PROPOSED → EXECUTING → AWAIT_FEEDBACK, plus RETURNING and STOPPED"], BLUE)
    s.line(495, 1550, 495, 1585, arrow=True)
    s.line(1305, 1550, 1305, 1585, arrow=True)
    s.box(115, 1590, 765, 95, "Robot", ["BaseRobot: simulated now, SerBot driver later"], PURPLE, 26)
    s.box(920, 1590, 765, 95, "Web server + prompts", ["FastAPI, annotated video, espeak-ng"], AMBER, 26)

    s.line(1305, 1700, 1305, 1785, arrow=True)
    s.text(1330, 1740, "http://<pi>:8000/", 23, "#333", anchor="start", family="Consolas, monospace")
    s.box(70, 1790, 1660, 135, "Any browser on the network",
          ["Live annotated video, gesture, state, prompt, objects, robot, camera status, STOP button"], GRAY)
    legend = [("Device", AMBER), ("Perception", GREEN), ("Decision logic", BLUE), ("Robot action", PURPLE)]
    for i, (label, palette) in enumerate(legend):
        x = 450 + i * 240
        s.rect(x, 1975, 26, 26, palette[0], palette[1], rx=6)
        s.text(x + 38, 1996, label, 22, "#555", anchor="start")
    s.write(ROOT / "architecture.svg")


PI_PINS = [("3V3", "5V"), ("GPIO2", "5V"), ("GPIO3", "GND"), ("GPIO4", "GPIO14"), ("GND", "GPIO15"), ("GPIO17", "GPIO18"),
           ("GPIO27", "GND"), ("GPIO22", "GPIO23"), ("3V3", "GPIO24"), ("GPIO10", "GND"), ("GPIO9", "GPIO25"),
           ("GPIO11", "GPIO8"), ("GND", "GPIO7")]
ESP_LEFT = ["5V", "GND", "IO12", "IO13", "IO15", "IO14", "IO2", "IO4"]
ESP_RIGHT = ["3V3", "IO16", "IO0", "GND", "VCC", "U0R", "U0T", "GND"]
WIRE_5V, WIRE_GND, WIRE_TX, WIRE_RX, WIRE_IO0 = RED, BLACK, "#2e9e5b", "#3b6fd4", "#e08a1e"


def pin_label(s: Svg, x, y, value, color=None) -> None:
    width = 120
    if color:
        s.rect(x, y - 16, width, 32, color + "22", color, 2, rx=16)
        s.text(x + width / 2, y + 6, value, 15, "#1f1f1f", "600")
    else:
        s.rect(x, y - 16, width, 32, "#f5f4f0", "#d9d6cf", 1.2, rx=16)
        s.text(x + width / 2, y + 6, value, 14, "#9a978f")


def pin_diagram() -> None:
    s = Svg(1800, 2010)
    s.text(900, 88, "GestureGrasp: hardware pin diagram", 46, weight="700")
    s.text(900, 130, "ESP32-CAM camera  ·  Raspberry Pi 5  ·  any browser. Video travels over Wi-Fi: only power is wired.", 22, MUTED)

    # Raspberry Pi 5 header
    s.text(330, 182, "Raspberry Pi 5", 24, "#333", "600")
    s.rect(230, 198, 200, 700, AMBER[0], AMBER[1], rx=12)
    s.text(330, 236, "40-pin GPIO header", 17, "#8a5a12", "600")
    s.text(330, 260, "pins 1–26 shown", 15, "#8a5a12")
    s.rect(262, 278, 136, 600, "#2f2f2f", "#2f2f2f", 1, rx=6)
    pin_y = {}
    for i, (left, right) in enumerate(PI_PINS):
        y = 302 + i * 46
        pin_y[2 * i + 1], pin_y[2 * i + 2] = y, y
        s.circle(290, y, 15, "#e9c46a")
        s.text(290, y + 5, str(2 * i + 1), 12, "#3d2c05", "700")
        wired = {2: WIRE_5V, 6: WIRE_GND}.get(2 * i + 2)
        s.circle(370, y, 15, "#e9c46a", wired or "#8a6a1a", 3 if wired else 1.5)
        s.text(370, y + 5, str(2 * i + 2), 12, "#3d2c05", "700")
        pin_label(s, 90, y, left)
        pin_label(s, 450, y, right, wired)

    # ESP32-CAM
    s.text(1160, 182, "ESP32-CAM (AI-Thinker)", 24, "#333", "600")
    s.rect(1040, 198, 240, 590, AMBER[0], AMBER[1], rx=12)
    s.text(1160, 236, "top view, camera up", 17, "#8a5a12", "600")
    s.text(1160, 260, "OV2640 · Wi-Fi · 4 MB PSRAM", 15, "#8a5a12")
    s.rect(1076, 278, 40, 410, "#2f2f2f", "#2f2f2f", 1, rx=6)
    s.rect(1204, 278, 40, 410, "#2f2f2f", "#2f2f2f", 1, rx=6)
    s.rect(1125, 705, 70, 62, "#2f2f2f", "#2f2f2f", 1, rx=8)
    s.circle(1160, 736, 22, "#555", "#999", 3)
    s.circle(1160, 736, 10, "#1b2a4a", "#1b2a4a", 1)
    colours_left = {"5V": WIRE_5V, "GND": WIRE_GND}
    colours_right = {"IO0": WIRE_IO0, "U0R": WIRE_TX, "U0T": WIRE_RX}
    for i, name in enumerate(ESP_LEFT):
        y = 302 + i * 52
        colour = colours_left.get(name)
        s.circle(1096, y, 15, "#e9c46a", colour or "#8a6a1a", 3 if colour else 1.5)
        pin_label(s, 890, y, name, colour)
        s.line(1010, y, 1081, y, colour or "#d9d6cf", 3 if colour else 1.2)
    for i, name in enumerate(ESP_RIGHT):
        y = 302 + i * 52
        colour = colours_right.get(name)
        s.circle(1224, y, 15, "#e9c46a", colour or "#8a6a1a", 3 if colour else 1.5)
        pin_label(s, 1310, y, name, colour)
        s.line(1239, y, 1310, y, colour or "#d9d6cf", 3 if colour else 1.2)

    # Power wires, Pi to ESP32-CAM
    s.line(570, pin_y[2], 890, pin_y[2], WIRE_5V, 5)
    s.text(730, pin_y[2] - 12, "5 V", 18, WIRE_5V, "700")
    s.path(f"M570,{pin_y[6]} H720 V354 H890", WIRE_GND, 5)
    s.text(800, 344, "GND", 18, WIRE_GND, "700")

    # USB-serial adapter, flashing only
    s.rect(1500, 300, 240, 300, "#f7f7fb", "#8a8aa0", dash="10 6")
    s.text(1620, 338, "USB-serial adapter", 19, "#3a3a55", "600")
    s.text(1620, 362, "only for flashing", 15, "#3a3a55")
    pin_label(s, 1560, 406, "IO0 → GND", WIRE_IO0)
    pin_label(s, 1560, 462, "TX", WIRE_TX)
    pin_label(s, 1560, 518, "RX", WIRE_RX)
    s.path("M1430,406 H1560", WIRE_IO0)
    s.path("M1430,562 H1465 V462 H1560", WIRE_TX)
    s.path("M1430,614 H1485 V518 H1560", WIRE_RX)

    # Network
    s.rect(50, 930, 1700, 240, "none", "#2f6db5", 2.5, dash="12 8")
    s.text(80, 968, "One Wi-Fi network (2.4 GHz). A router, or the Pi as its own hotspot.", 21, "#1d3f6e", "600", "start")
    s.box(90, 1000, 400, 130, "ESP32-CAM", ["camera, streams MJPEG"], AMBER, 28)
    s.box(700, 1000, 400, 130, "Raspberry Pi 5", ["python -m server, all models"], AMBER, 28)
    s.box(1310, 1000, 400, 130, "Browser", ["phone, tablet or PC"], GREEN, 28)
    s.line(490, 1065, 690, 1065, arrow=True, dash="10 8")
    s.text(590, 1045, "Wi-Fi MJPEG", 19, "#333")
    s.text(590, 1095, ":81/stream", 16, "#666", family="Consolas, monospace")
    s.line(1100, 1065, 1300, 1065, arrow=True, dash="10 8")
    s.text(1200, 1045, "video + state", 19, "#333")
    s.text(1200, 1095, ":8000/", 16, "#666", family="Consolas, monospace")

    # Wiring table
    s.text(70, 1232, "Wiring table", 26, "#333", "600", "start")
    s.rect(60, 1248, 1680, 42, "#f2f1ec", "none", 0, rx=6)
    for x, value in ((80, "Wire"), (170, "From"), (500, "To"), (850, "Purpose")):
        s.text(x, 1276, value, 19, "#333", "600", "start")
    rows = [(WIRE_5V, "ESP32-CAM 5V", "Pi pin 2 (5V), or its own 5 V supply", "Power. Use 5 V, not 3V3: Wi-Fi current peaks brown out the 3.3 V pin."),
            (WIRE_GND, "ESP32-CAM GND", "Pi pin 6 (GND)", "Power return. Needed only when powering it from the Pi."),
            (WIRE_TX, "Adapter TX", "ESP32-CAM U0R (GPIO3)", "Flashing only."),
            (WIRE_RX, "Adapter RX", "ESP32-CAM U0T (GPIO1)", "Flashing and the serial log, which prints the IP address."),
            (WIRE_IO0, "ESP32-CAM IO0", "GND", "Flashing only: hold low during reset, then remove for normal running."),
            ("#999", "Pi USB port", "USB speaker", "Spoken prompts with --speak. The Pi 5 has no audio jack.")]
    for i, (colour, a, b, c) in enumerate(rows):
        y = 1330 + i * 44
        s.line(80, y - 6, 130, y - 6, colour, 5)
        s.text(170, y, a, 19, "#222", anchor="start")
        s.text(500, y, b, 19, "#222", anchor="start")
        s.text(850, y, c, 19, "#555", anchor="start")

    # Notes
    s.rect(60, 1612, 1680, 300, "#f6f8fc", "#c9d6ec", 2, rx=12)
    s.text(90, 1652, "Notes", 23, "#1d3f6e", "600", "start")
    notes = ["A stable 5 V supply matters most: at least 1 A, 2 A preferred. Brownouts are the main cause of dropouts.",
             "The Pi 5's 5 V pin suits this with its 5 A power supply. A separate 5 V 2 A supply is the safer choice.",
             "Short, thick power wires: thin jumpers can drop enough voltage to reset the board.",
             "The ESP32 radio is 2.4 GHz only, so all devices must share a 2.4 GHz network or the Pi's hotspot.",
             "The ESP32 stream serves one client: the Pi. Browsers watch the Pi's /video, never the ESP32 directly.",
             "SerBot's control link is not written yet: the robot is simulated (robot/simulated.py) until a BaseRobot driver exists."]
    for i, value in enumerate(notes):
        s.text(90, 1692 + i * 36, "•  " + value, 19, "#333", anchor="start")
    legend = [("Pi-side device", AMBER), ("Browser", GREEN)]
    for i, (label, palette) in enumerate(legend):
        x = 600 + i * 240
        s.rect(x, 1948, 26, 26, palette[0], palette[1], rx=6)
        s.text(x + 38, 1969, label, 21, "#555", anchor="start")
    s.line(1060, 1961, 1120, 1961, "#555", 3, dash="10 8")
    s.text(1134, 1969, "wireless", 21, "#555", anchor="start")
    s.write(ROOT / "pin_diagram.svg")


if __name__ == "__main__":
    architecture()
    pin_diagram()

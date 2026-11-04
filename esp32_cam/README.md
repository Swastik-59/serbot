# ESP32-CAM firmware

Firmware for the AI-Thinker ESP32-CAM (OV2640). It connects to your 2.4 GHz Wi-Fi and streams MJPEG to the
Raspberry Pi 5, which runs everything else.

## 1. Set your Wi-Fi details

Edit [`config.h`](config.h):

```c
#define WIFI_SSID     "YOUR_WIFI_SSID"
#define WIFI_PASSWORD "YOUR_WIFI_PASSWORD"
```

The ESP32 only sees 2.4 GHz networks. Keep your real password out of git.

## 2. Wire it up for flashing

The AI-Thinker board has no USB port, so you need a USB-serial adapter or an ESP32-CAM-MB base board.

| USB-serial adapter | ESP32-CAM |
|---|---|
| 5V | 5V |
| GND | GND |
| TX | U0R |
| RX | U0T |
| GND | **IO0** (only while flashing) |

With an ESP32-CAM-MB base board, plug the camera in and hold **IO0** while pressing **RST**.

## 3. Flash it

### Arduino IDE

1. In **File > Preferences > Additional boards manager URLs**, add
   `https://espressif.github.io/arduino-esp32/package_esp32_index.json`
2. In **Tools > Board > Boards Manager**, install **esp32 by Espressif Systems** (version 3.x).
3. Open `esp32_cam/esp32_cam.ino`.
4. Choose **Tools > Board > esp32 > AI Thinker ESP32-CAM**, check that **PSRAM** is **Enabled** if the menu
   offers it, and choose your serial port.
5. Connect IO0 to GND, press **RST**, then click **Upload**.
6. Remove the IO0 wire and press **RST**. The board will not run your code while IO0 is held low.

### PlatformIO

Create `platformio.ini` next to the `esp32_cam/` folder:

```ini
[platformio]
src_dir = esp32_cam

[env:esp32cam]
platform = https://github.com/pioarduino/platform-espressif32/releases/download/stable/platform-espressif32.zip
board = esp32cam
framework = arduino
monitor_speed = 115200
build_flags = -DBOARD_HAS_PSRAM -mfix-esp32-psram-cache-issue
```

Then run `pio run -t upload` with IO0 on GND, and `pio device monitor` after removing it. The stock PlatformIO
`espressif32` platform ships an older Arduino core, so this uses the pioarduino platform for core 3.x.

## 4. Find the board's address

Open the serial monitor at 115200 baud and press **RST**:

```
GestureGrasp ESP32-CAM
Camera ready, PID 0x26, buffers for uxga
Wi-Fi connected. Stream: http://192.168.1.50:81/stream  Status: http://192.168.1.50/status
mDNS: http://esp32cam.local
```

`esp32cam.local` works on most networks. If it does not resolve, use the IP address, and reserve that address
in your router so it stays the same.

## 5. Test it

From the Raspberry Pi (`<ip>` can also be `esp32cam.local`):

```bash
curl -s http://<ip>/status
curl -s -o test.jpg http://<ip>/capture && file test.jpg
curl -s --max-time 3 -o /dev/null -w "%{size_download} bytes in 3 s\n" http://<ip>:81/stream
```

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET :81/stream` | MJPEG stream (`multipart/x-mixed-replace`) |
| `GET :80/capture` | One JPEG |
| `GET :80/status` | JSON: framesize, quality, fps, uptime, free heap and PSRAM, RSSI, IP |
| `GET :80/control?var=<name>&val=<n>` | Change a setting while running |

`/control` accepts `framesize` (a name such as `vga` or `qvga`, or the number from `esp_camera`), `quality`,
`brightness`, `contrast`, `saturation`, `sharpness`, `hmirror`, `vflip`, `awb`, `awb_gain`, `wb_mode`, `aec`,
`aec2`, `ae_level`, `aec_value`, `agc`, `agc_gain`, `gainceiling`, `special_effect`, `lenc`, `bpc`, `wpc`,
`raw_gma`, `dcw` and `colorbar`.

```bash
curl "http://<ip>/control?var=framesize&val=qvga"
curl "http://<ip>/control?var=quality&val=15"
```

Settings live in RAM and are lost on reboot, so the Pi sends its frame size, quality and flip on every connection.

## The stream serves one client

`esp_http_server` handles one request at a time, and `/stream` never returns, so a second client waits until the
first disconnects. **The Raspberry Pi must be the only consumer.** Browsers watch the Pi's `/video`, never this
board directly. Opening the stream in a browser while the Pi is connected will leave one of them waiting.

`/capture`, `/status` and `/control` run on a separate server on port 80, so they keep working during a stream.

## Settings that matter

- **VGA 640x480, quality 12** is the default. At QVGA, a hand 1-2 m away is too small for MediaPipe.
- **`CAMERA_GRAB_LATEST`** always hands over the newest frame, never a stale queued one.
- **Buffers are sized at start-up.** The firmware starts at UXGA so any smaller size works later, and falls back to
  VGA and then QVGA if that much memory cannot be found. The serial log says which one it got, and `/control`
  refuses a frame size above it.

## Power

**Brownouts are the most common cause of dropouts.** Wi-Fi transmit peaks are large for such a small board.

- Use a stable **5 V supply**: at least 1 A, 2 A preferred.
- Power it through the **5V pin**, not 3V3.
- Keep the power wires short. Thin jumper wires drop enough voltage to cause resets.
- A `Brownout detector was triggered` line in the serial log means the supply is the problem.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Upload fails: `Failed to connect` | IO0 must be on GND and the board reset just before uploading. Check TX to U0R and RX to U0T |
| Nothing after upload | Remove the IO0 wire and press RST |
| `No PSRAM found` | Enable PSRAM in the board settings |
| `Camera init ... failed` | Reseat the ribbon cable. Check the board really is an AI-Thinker |
| `Brownout detector was triggered` | Better 5 V supply, shorter wires |
| Never joins Wi-Fi | Check `config.h`, and that the network is 2.4 GHz, not 5 GHz only |
| `esp32cam.local` does not resolve | Use the IP. On the Pi: `sudo apt install avahi-daemon libnss-mdns` |
| Reboots with `Too many camera failures` | Usually power or a loose camera ribbon |
| Stream opens but hangs | Another client holds `/stream`. Stop the Pi's service or close that browser tab |

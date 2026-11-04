// GestureGrasp ESP32-CAM settings. Fill in the Wi-Fi details, then flash.
// Do not commit real credentials to git.
#pragma once

// 2.4 GHz only: the ESP32 cannot see 5 GHz networks.
#define WIFI_SSID     "YOUR_WIFI_SSID"
#define WIFI_PASSWORD "YOUR_WIFI_PASSWORD"

// The board answers at http://esp32cam.local
#define MDNS_NAME "esp32cam"

// Optional static IP. Leave 0 for DHCP, which suits most networks.
#define USE_STATIC_IP 0
#define STATIC_IP      192, 168, 1, 50
#define STATIC_GATEWAY 192, 168, 1, 1
#define STATIC_SUBNET  255, 255, 255, 0
#define STATIC_DNS     192, 168, 1, 1

// VGA is the default because a hand at 1-2 m is too small for MediaPipe at QVGA.
#define DEFAULT_FRAMESIZE FRAMESIZE_VGA
#define DEFAULT_QUALITY   12  // 10 (best, larger frames) to 63

// Reboot after this many esp_camera_fb_get() failures in a row.
#define MAX_FB_FAILURES 20

// Restart if Wi-Fi stays down this long, in case the radio itself has hung.
#define WIFI_RESTART_MS 120000

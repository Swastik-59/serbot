// GestureGrasp firmware for the AI-Thinker ESP32-CAM (OV2640).
//
//   GET :81/stream                    MJPEG (multipart/x-mixed-replace)
//   GET :80/capture                   one JPEG
//   GET :80/status                    JSON: framesize, quality, fps, uptime, heap, PSRAM, RSSI, IP
//   GET :80/control?var=<name>&val=<n>  framesize takes a name (vga) or a number
//
// The stream runs on its own HTTP server because its handler never returns, and esp_http_server
// handles one request at a time. That also means the stream serves ONE client: a second one waits
// until the first disconnects. The Raspberry Pi must be the only consumer. Browsers watch the Pi's
// /video, never this board directly.
//
// Board: "AI Thinker ESP32-CAM", PSRAM enabled. Written for Arduino-ESP32 core 3.x.

#include <Arduino.h>
#include <ESPmDNS.h>
#include <WiFi.h>

#include "esp_camera.h"
#include "esp_http_server.h"
#include "esp_timer.h"

#include "config.h"

// AI-Thinker pin map
#define PWDN_GPIO_NUM  32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  0
#define SIOD_GPIO_NUM  26
#define SIOC_GPIO_NUM  27
#define Y9_GPIO_NUM    35
#define Y8_GPIO_NUM    34
#define Y7_GPIO_NUM    39
#define Y6_GPIO_NUM    36
#define Y5_GPIO_NUM    21
#define Y4_GPIO_NUM    19
#define Y3_GPIO_NUM    18
#define Y2_GPIO_NUM    5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM  23
#define PCLK_GPIO_NUM  22

#define PART_BOUNDARY "123456789000000000000987654321"
static const char *STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char *STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char *STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

struct FrameSizeName {
  const char *name;
  framesize_t size;
};

static const FrameSizeName FRAME_SIZES[] = {
    {"96x96", FRAMESIZE_96X96}, {"qqvga", FRAMESIZE_QQVGA}, {"qcif", FRAMESIZE_QCIF},
    {"hqvga", FRAMESIZE_HQVGA}, {"240x240", FRAMESIZE_240X240}, {"qvga", FRAMESIZE_QVGA},
    {"cif", FRAMESIZE_CIF},     {"hvga", FRAMESIZE_HVGA},   {"vga", FRAMESIZE_VGA},
    {"svga", FRAMESIZE_SVGA},   {"xga", FRAMESIZE_XGA},     {"hd", FRAMESIZE_HD},
    {"sxga", FRAMESIZE_SXGA},   {"uxga", FRAMESIZE_UXGA},
};
static const size_t FRAME_SIZE_COUNT = sizeof(FRAME_SIZES) / sizeof(FRAME_SIZES[0]);

static httpd_handle_t control_server = NULL;
static httpd_handle_t stream_server = NULL;

// Buffers are sized when the camera starts, so no later frame size may exceed the one it started with.
static framesize_t max_framesize = FRAMESIZE_QVGA;
static volatile int fb_failures = 0;
static volatile float stream_fps = 0.0f;
static unsigned long wifi_lost_at = 0;

static const char *frame_size_name(framesize_t size) {
  for (size_t i = 0; i < FRAME_SIZE_COUNT; i++) {
    if (FRAME_SIZES[i].size == size) return FRAME_SIZES[i].name;
  }
  return "unknown";
}

static bool parse_frame_size(const char *value, framesize_t *out) {
  for (size_t i = 0; i < FRAME_SIZE_COUNT; i++) {
    if (strcmp(FRAME_SIZES[i].name, value) == 0) {
      *out = FRAME_SIZES[i].size;
      return true;
    }
  }
  char *end = NULL;
  long number = strtol(value, &end, 10);
  if (end != value && *end == '\0' && number >= 0 && number < FRAMESIZE_INVALID) {
    *out = (framesize_t)number;
    return true;
  }
  return false;
}

// A camera that keeps failing rarely recovers by itself, and the Pi reconnects after a reboot.
static camera_fb_t *grab_frame() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (fb != NULL) {
    fb_failures = 0;
    return fb;
  }
  fb_failures = fb_failures + 1;
  Serial.printf("esp_camera_fb_get failed (%d)\n", fb_failures);
  if (fb_failures >= MAX_FB_FAILURES) {
    Serial.println("Too many camera failures, rebooting");
    delay(100);
    ESP.restart();
  }
  return NULL;
}

// -- HTTP handlers ------------------------------------------------------------------------------

static esp_err_t stream_handler(httpd_req_t *req) {
  char part[64];
  int64_t window_start = esp_timer_get_time();
  int window_frames = 0;

  esp_err_t res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "Cache-Control", "no-store");

  while (true) {
    camera_fb_t *fb = grab_frame();
    if (fb == NULL) {
      delay(20);
      continue;
    }
    if (fb->format != PIXFORMAT_JPEG) {
      esp_camera_fb_return(fb);
      return ESP_FAIL;
    }
    size_t header = snprintf(part, sizeof(part), STREAM_PART, (unsigned)fb->len);
    res = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, part, header);
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    if (res != ESP_OK) break;  // the client went away

    window_frames++;
    int64_t elapsed = esp_timer_get_time() - window_start;
    if (elapsed >= 2000000) {
      stream_fps = window_frames * 1000000.0f / elapsed;
      window_frames = 0;
      window_start = esp_timer_get_time();
    }
  }
  stream_fps = 0.0f;
  return res;
}

static esp_err_t capture_handler(httpd_req_t *req) {
  camera_fb_t *fb = grab_frame();
  if (fb == NULL) {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }
  httpd_resp_set_type(req, "image/jpeg");
  httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=capture.jpg");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  esp_err_t res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
  esp_camera_fb_return(fb);
  return res;
}

static esp_err_t status_handler(httpd_req_t *req) {
  sensor_t *sensor = esp_camera_sensor_get();
  char json[384];
  int length = snprintf(
      json, sizeof(json),
      "{\"framesize\":\"%s\",\"quality\":%d,\"fps\":%.1f,\"uptime_s\":%lu,\"free_heap\":%u,"
      "\"free_psram\":%u,\"rssi\":%d,\"ip\":\"%s\",\"hmirror\":%d,\"vflip\":%d}",
      frame_size_name(sensor->status.framesize), sensor->status.quality, (double)stream_fps,
      (unsigned long)(esp_timer_get_time() / 1000000), (unsigned)ESP.getFreeHeap(),
      (unsigned)ESP.getFreePsram(), (int)WiFi.RSSI(), WiFi.localIP().toString().c_str(),
      sensor->status.hmirror, sensor->status.vflip);
  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(req, json, length);
}

static int apply_control(sensor_t *s, const char *var, const char *value) {
  if (strcmp(var, "framesize") == 0) {
    framesize_t size;
    if (!parse_frame_size(value, &size) || size > max_framesize) return -1;
    return s->set_framesize(s, size);
  }
  int val = atoi(value);
  if (strcmp(var, "quality") == 0) return s->set_quality(s, val);
  if (strcmp(var, "contrast") == 0) return s->set_contrast(s, val);
  if (strcmp(var, "brightness") == 0) return s->set_brightness(s, val);
  if (strcmp(var, "saturation") == 0) return s->set_saturation(s, val);
  if (strcmp(var, "sharpness") == 0) return s->set_sharpness(s, val);
  if (strcmp(var, "hmirror") == 0) return s->set_hmirror(s, val);
  if (strcmp(var, "vflip") == 0) return s->set_vflip(s, val);
  if (strcmp(var, "awb") == 0) return s->set_whitebal(s, val);
  if (strcmp(var, "awb_gain") == 0) return s->set_awb_gain(s, val);
  if (strcmp(var, "wb_mode") == 0) return s->set_wb_mode(s, val);
  if (strcmp(var, "aec") == 0) return s->set_exposure_ctrl(s, val);
  if (strcmp(var, "aec2") == 0) return s->set_aec2(s, val);
  if (strcmp(var, "ae_level") == 0) return s->set_ae_level(s, val);
  if (strcmp(var, "aec_value") == 0) return s->set_aec_value(s, val);
  if (strcmp(var, "agc") == 0) return s->set_gain_ctrl(s, val);
  if (strcmp(var, "agc_gain") == 0) return s->set_agc_gain(s, val);
  if (strcmp(var, "gainceiling") == 0) return s->set_gainceiling(s, (gainceiling_t)val);
  if (strcmp(var, "special_effect") == 0) return s->set_special_effect(s, val);
  if (strcmp(var, "lenc") == 0) return s->set_lenc(s, val);
  if (strcmp(var, "bpc") == 0) return s->set_bpc(s, val);
  if (strcmp(var, "wpc") == 0) return s->set_wpc(s, val);
  if (strcmp(var, "raw_gma") == 0) return s->set_raw_gma(s, val);
  if (strcmp(var, "dcw") == 0) return s->set_dcw(s, val);
  if (strcmp(var, "colorbar") == 0) return s->set_colorbar(s, val);
  return -1;
}

static esp_err_t control_handler(httpd_req_t *req) {
  char query[64];
  char var[24];
  char value[16];
  if (httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK ||
      httpd_query_key_value(query, "var", var, sizeof(var)) != ESP_OK ||
      httpd_query_key_value(query, "val", value, sizeof(value)) != ESP_OK) {
    httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "use /control?var=<name>&val=<n>");
    return ESP_FAIL;
  }
  if (apply_control(esp_camera_sensor_get(), var, value) != 0) {
    httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "unknown variable or value");
    return ESP_FAIL;
  }
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(req, NULL, 0);
}

static void start_servers() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.ctrl_port = 32768;
  config.max_uri_handlers = 4;
  httpd_uri_t capture_uri = {"/capture", HTTP_GET, capture_handler, NULL};
  httpd_uri_t status_uri = {"/status", HTTP_GET, status_handler, NULL};
  httpd_uri_t control_uri = {"/control", HTTP_GET, control_handler, NULL};
  if (httpd_start(&control_server, &config) == ESP_OK) {
    httpd_register_uri_handler(control_server, &capture_uri);
    httpd_register_uri_handler(control_server, &status_uri);
    httpd_register_uri_handler(control_server, &control_uri);
  } else {
    Serial.println("Could not start the server on port 80");
  }

  httpd_config_t stream_config = HTTPD_DEFAULT_CONFIG();
  stream_config.server_port = 81;
  stream_config.ctrl_port = 32769;
  stream_config.max_uri_handlers = 1;
  stream_config.send_wait_timeout = 3;  // notice a vanished client quickly, so the Pi can reconnect
  httpd_uri_t stream_uri = {"/stream", HTTP_GET, stream_handler, NULL};
  if (httpd_start(&stream_server, &stream_config) == ESP_OK) {
    httpd_register_uri_handler(stream_server, &stream_uri);
  } else {
    Serial.println("Could not start the server on port 81");
  }
}

// -- Setup --------------------------------------------------------------------------------------

static camera_config_t base_camera_config() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.jpeg_quality = DEFAULT_QUALITY;
  config.fb_count = 2;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_LATEST;  // always the newest frame, never a stale queued one
  return config;
}

// Buffers are allocated for the start-up frame size. Starting large keeps runtime changes possible,
// and each smaller size is a retry in case that much memory cannot be found.
static bool start_camera() {
  static const framesize_t attempts[] = {FRAMESIZE_UXGA, DEFAULT_FRAMESIZE, FRAMESIZE_QVGA};
  for (size_t i = 0; i < sizeof(attempts) / sizeof(attempts[0]); i++) {
    camera_config_t config = base_camera_config();
    config.frame_size = attempts[i];
    esp_err_t err = esp_camera_init(&config);
    if (err == ESP_OK) {
      max_framesize = attempts[i];
      sensor_t *sensor = esp_camera_sensor_get();
      sensor->set_framesize(sensor, DEFAULT_FRAMESIZE <= max_framesize ? DEFAULT_FRAMESIZE : max_framesize);
      sensor->set_quality(sensor, DEFAULT_QUALITY);
      Serial.printf("Camera ready, PID 0x%02X, buffers for %s\n", sensor->id.PID, frame_size_name(max_framesize));
      return true;
    }
    Serial.printf("Camera init at %s failed: 0x%x\n", frame_size_name(attempts[i]), err);
  }
  return false;
}

static void start_mdns() {
  MDNS.end();
  if (MDNS.begin(MDNS_NAME)) {
    MDNS.addService("http", "tcp", 80);
    Serial.printf("mDNS: http://%s.local\n", MDNS_NAME);
  } else {
    Serial.println("mDNS failed. Use the IP address instead.");
  }
}

static void on_wifi_event(WiFiEvent_t event) {
  if (event == ARDUINO_EVENT_WIFI_STA_GOT_IP) {
    wifi_lost_at = 0;
    Serial.printf("Wi-Fi connected. Stream: http://%s:81/stream  Status: http://%s/status\n",
                  WiFi.localIP().toString().c_str(), WiFi.localIP().toString().c_str());
    start_mdns();
  } else if (event == ARDUINO_EVENT_WIFI_STA_DISCONNECTED && wifi_lost_at == 0) {
    wifi_lost_at = millis();
    Serial.println("Wi-Fi lost, reconnecting");
  }
}

static void start_wifi() {
  if (strcmp(WIFI_SSID, "YOUR_WIFI_SSID") == 0) {
    Serial.println("Set WIFI_SSID and WIFI_PASSWORD in config.h, then flash again.");
  }
  WiFi.mode(WIFI_STA);
  WiFi.setHostname(MDNS_NAME);
  WiFi.onEvent(on_wifi_event);
#if USE_STATIC_IP
  WiFi.config(IPAddress(STATIC_IP), IPAddress(STATIC_GATEWAY), IPAddress(STATIC_SUBNET), IPAddress(STATIC_DNS));
#endif
  WiFi.setAutoReconnect(true);
  wifi_lost_at = millis();
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  WiFi.setSleep(false);  // modem sleep adds large, uneven delays to a video stream
}

void setup() {
  Serial.begin(115200);
  Serial.println();
  Serial.println("GestureGrasp ESP32-CAM");
  if (!psramFound()) {
    Serial.println("No PSRAM found. Select PSRAM: Enabled in the board settings.");
  }
  if (!start_camera()) {
    Serial.println("Camera failed to start, rebooting in 5 s");
    delay(5000);
    ESP.restart();
  }
  start_wifi();
  start_servers();
}

void loop() {
  // setAutoReconnect handles normal drops. This covers a radio that has stopped trying.
  unsigned long lost_at = wifi_lost_at;
  if (lost_at != 0 && millis() - lost_at > WIFI_RESTART_MS) {
    Serial.println("Wi-Fi down too long, rebooting");
    delay(100);
    ESP.restart();
  }
  delay(1000);
}

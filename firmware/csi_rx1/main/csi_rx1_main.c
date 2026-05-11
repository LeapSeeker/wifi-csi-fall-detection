/**
 * ============================================================
 * 파일명: csi_rx1_main.c
 * 역할:   ESP32-S3 RX1 (수신 1번)
 * 서버:   211.227.32.31:5005
 *
 * 빌드 전 필수:
 *   idf.py menuconfig
 *   → Component config → Wi-Fi → WiFi CSI(Channel State Information) 체크
 *
 * 명령어 (모니터 실행 중):
 *   set_wifi <SSID> <PASSWORD>
 *   set_server <IP> <PORT>
 *   show_config
 *   restart
 * ============================================================
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_system.h"
#include "nvs_flash.h"
#include "nvs.h"
#include "lwip/sockets.h"
#include "lwip/netdb.h"
#include "esp_sntp.h"

/* ── 고정 설정값 ─────────────────────────────────────── */
#define DEVICE_ID           0x01
#define PACKET_MAGIC        0xAB
#define SUBCARRIER_VALID    52
#define CSI_QUEUE_SIZE      500
#define TX_MAC              {0x1c, 0xdb, 0xd4, 0x40, 0x4f, 0x90}

/* ── 기본값 ──────────────────────────────────────────── */
#define DEFAULT_WIFI_SSID    "coin"
#define DEFAULT_WIFI_PASS    "1q2w3e4r"
#define DEFAULT_SERVER_IP    "211.227.32.31"
#define DEFAULT_SERVER_PORT  5005
#define DEFAULT_WIFI_CHANNEL 6
/* ──────────────────────────────────────────────────── */

static const char *TAG = "CSI_RX1";

/* ── NVS 설정값 ───────────────────────────────────────── */
static char g_wifi_ssid[32]  = DEFAULT_WIFI_SSID;
static char g_wifi_pass[64]  = DEFAULT_WIFI_PASS;
static char g_server_ip[16]  = DEFAULT_SERVER_IP;
static int  g_server_port    = DEFAULT_SERVER_PORT;
static int  g_wifi_channel   = DEFAULT_WIFI_CHANNEL;
static const uint8_t TX_MAC_ADDR[6] = TX_MAC;

/* ── 유효 서브캐리어 인덱스 (LLTF, Null 제거) ────────── */
static const int VALID_IDX[SUBCARRIER_VALID] = {
     6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
    53, 54, 55, 56, 57, 58
};

/* ── UDP 패킷 구조체 (224 bytes) ─────────────────────── */
typedef struct {
    uint8_t  magic;                        // 1 byte  — 0xAB (패킷 유효성 검증)
    uint8_t  device_id;                    // 1 byte  — 0x01
    int8_t   rssi;                         // 1 byte  — 수신 신호 세기 (dBm)
    uint8_t  reserved;                     // 1 byte  — 향후 확장용 패딩
    uint32_t seq;                          // 4 bytes — 시퀀스 번호
    uint64_t timestamp_us;                 // 8 bytes — Unix time (마이크로초)
    float    amplitude[SUBCARRIER_VALID];  // 208 bytes — 진폭 52개
} __attribute__((packed)) csi_packet_t;

/* ── 전역 변수 ────────────────────────────────────────── */
static QueueHandle_t      csi_queue = NULL;
static int                udp_sock  = -1;
static struct sockaddr_in server_addr;
static volatile uint32_t  pkt_seq   = 0;
static EventGroupHandle_t wifi_event_group;
#define WIFI_CONNECTED_BIT BIT0

/* ═══════════════════════════════════════════════════════
 * NVS 저장 / 로드
 * ═══════════════════════════════════════════════════════ */
static void nvs_save_config(void)
{
    nvs_handle_t h;
    if (nvs_open("csi_cfg", NVS_READWRITE, &h) != ESP_OK) return;
    nvs_set_str(h, "ssid",    g_wifi_ssid);
    nvs_set_str(h, "pass",    g_wifi_pass);
    nvs_set_str(h, "ip",      g_server_ip);
    nvs_set_i32(h, "port",    g_server_port);
    nvs_set_i32(h, "channel", g_wifi_channel);
    nvs_commit(h);
    nvs_close(h);
    ESP_LOGI(TAG, "설정 저장 완료");
}

static void nvs_load_config(void)
{
    nvs_handle_t h;
    if (nvs_open("csi_cfg", NVS_READONLY, &h) != ESP_OK) {
        ESP_LOGI(TAG, "저장된 설정 없음 → 기본값 사용");
        return;
    }
    size_t len;
    len = sizeof(g_wifi_ssid);  nvs_get_str(h, "ssid", g_wifi_ssid, &len);
    len = sizeof(g_wifi_pass);  nvs_get_str(h, "pass", g_wifi_pass, &len);
    len = sizeof(g_server_ip);  nvs_get_str(h, "ip",   g_server_ip, &len);
    nvs_get_i32(h, "port",    (int32_t*)&g_server_port);
    nvs_get_i32(h, "channel", (int32_t*)&g_wifi_channel);
    nvs_close(h);
    ESP_LOGI(TAG, "설정 로드 완료");
}

/* ═══════════════════════════════════════════════════════
 * 시리얼 명령어 태스크
 * ═══════════════════════════════════════════════════════ */
static void serial_cmd_task(void *pvParameters)
{
    char buf[128];
    int  pos = 0;

    printf("\n========================================\n");
    printf("  CSI RX1 — 명령어 목록\n");
    printf("  set_wifi <SSID> <PASSWORD>\n");
    printf("  set_server <IP> <PORT>\n");
    printf("  set_channel <CH>\n");
    printf("  show_config\n");
    printf("  restart\n");
    printf("========================================\n\n");

    while (1) {
        int c = getchar();
        if (c == EOF) { vTaskDelay(pdMS_TO_TICKS(10)); continue; }

        if (c == '\n' || c == '\r') {
            buf[pos] = '\0';
            if (pos == 0) { pos = 0; continue; }

            char cmd[32], arg1[64], arg2[32];
            int  n = sscanf(buf, "%31s %63s %31s", cmd, arg1, arg2);

            if (strcmp(cmd, "set_wifi") == 0 && n >= 3) {
                strncpy(g_wifi_ssid, arg1, sizeof(g_wifi_ssid)-1);
                strncpy(g_wifi_pass, arg2, sizeof(g_wifi_pass)-1);
                nvs_save_config();
                printf("[OK] WiFi: %s / %s → restart 필요\n", g_wifi_ssid, g_wifi_pass);

            } else if (strcmp(cmd, "set_server") == 0 && n >= 3) {
                strncpy(g_server_ip, arg1, sizeof(g_server_ip)-1);
                g_server_port = atoi(arg2);
                nvs_save_config();
                printf("[OK] 서버: %s:%d → restart 필요\n", g_server_ip, g_server_port);

            } else if (strcmp(cmd, "set_channel") == 0 && n >= 2) {
                g_wifi_channel = atoi(arg1);
                nvs_save_config();
                printf("[OK] 채널: %d → restart 필요\n", g_wifi_channel);

            } else if (strcmp(cmd, "show_config") == 0) {
                printf("\n===== 현재 설정 =====\n");
                printf("  SSID    : %s\n", g_wifi_ssid);
                printf("  PW      : %s\n", g_wifi_pass);
                printf("  서버 IP : %s\n", g_server_ip);
                printf("  포트    : %d\n", g_server_port);
                printf("  채널    : %d\n", g_wifi_channel);
                printf("  DevID   : 0x%02X (RX1)\n", DEVICE_ID);
                printf("  Magic   : 0x%02X\n", PACKET_MAGIC);
                printf("  패킷    : %d bytes\n", (int)sizeof(csi_packet_t));
                printf("====================\n\n");

            } else if (strcmp(cmd, "restart") == 0) {
                printf("[INFO] 재시작...\n");
                vTaskDelay(pdMS_TO_TICKS(500));
                esp_restart();

            } else {
                printf("[ERR] 알 수 없는 명령어: %s\n", buf);
            }
            pos = 0;
        } else if (pos < (int)sizeof(buf)-1) {
            buf[pos++] = (char)c;
        }
    }
}

/* ═══════════════════════════════════════════════════════
 * CSI 콜백 — Raw IQ → 진폭 변환 → 큐 전달
 * ═══════════════════════════════════════════════════════ */
static void wifi_csi_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf) return;
    if (memcmp(info->mac, TX_MAC_ADDR, sizeof(TX_MAC_ADDR)) != 0) {
        ESP_LOGD(TAG, "CSI filtered: non-TX MAC");
        return;
    }

    csi_packet_t pkt;
    pkt.magic     = PACKET_MAGIC;
    pkt.device_id = DEVICE_ID;
    pkt.rssi      = info->rx_ctrl.rssi;
    pkt.reserved  = 0x00;
    pkt.seq       = pkt_seq++;

    /* SNTP 동기화된 Unix time (마이크로초) */
    struct timeval tv;
    gettimeofday(&tv, NULL);
    pkt.timestamp_us = (uint64_t)tv.tv_sec * 1000000ULL + tv.tv_usec;

    /* Raw IQ → 진폭 변환 + Null 서브캐리어 제거 */
    int8_t *buf = info->buf;
    for (int i = 0; i < SUBCARRIER_VALID; i++) {
        int    idx = VALID_IDX[i];
        int8_t im  = buf[2 * idx];
        int8_t re  = buf[2 * idx + 1];
        pkt.amplitude[i] = sqrtf((float)(re * re + im * im));
    }

    BaseType_t woken = pdFALSE;
    xQueueSendFromISR(csi_queue, &pkt, &woken);
    portYIELD_FROM_ISR(woken);
}

/* ── UDP 전송 태스크 ─────────────────────────────────── */
static void udp_send_task(void *pvParameters)
{
    csi_packet_t pkt;
    ESP_LOGI(TAG, "UDP 전송 태스크 시작 → %s:%d", g_server_ip, g_server_port);
    while (1) {
        if (xQueueReceive(csi_queue, &pkt, portMAX_DELAY) == pdTRUE) {
            sendto(udp_sock, &pkt, sizeof(csi_packet_t), 0,
                   (struct sockaddr *)&server_addr, sizeof(server_addr));
        }
    }
}

/* ── WiFi 이벤트 핸들러 ──────────────────────────────── */
static EventGroupHandle_t s_wifi_event_group;

static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                                int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *e = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "IP 획득: " IPSTR, IP2STR(&e->ip_info.ip));
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "WiFi 끊김 → 재연결");
        esp_wifi_connect();
        xEventGroupClearBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

/* ── WiFi 초기화 ─────────────────────────────────────── */
static void wifi_init(void)
{
    s_wifi_event_group = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                               &wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                               &wifi_event_handler, NULL));

    wifi_config_t wifi_cfg = {};
    strncpy((char*)wifi_cfg.sta.ssid,     g_wifi_ssid, sizeof(wifi_cfg.sta.ssid)-1);
    strncpy((char*)wifi_cfg.sta.password, g_wifi_pass, sizeof(wifi_cfg.sta.password)-1);
    wifi_cfg.sta.channel = g_wifi_channel;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_cfg));
    ESP_ERROR_CHECK(esp_wifi_start());

    xEventGroupWaitBits(s_wifi_event_group, WIFI_CONNECTED_BIT,
                        pdFALSE, pdTRUE, portMAX_DELAY);
    ESP_LOGI(TAG, "WiFi 연결: %s (채널 %d)", g_wifi_ssid, g_wifi_channel);
}

/* ── SNTP 초기화 ─────────────────────────────────────── */
static void sntp_sync(void)
{
    esp_sntp_setoperatingmode(SNTP_OPMODE_POLL);
    esp_sntp_setservername(0, "pool.ntp.org");
    esp_sntp_init();

    int retry = 0;
    while (sntp_get_sync_status() != SNTP_SYNC_STATUS_COMPLETED && retry < 10) {
        vTaskDelay(pdMS_TO_TICKS(500));
        retry++;
        ESP_LOGI(TAG, "NTP 동기화 대기 (%d/10)", retry);
    }
    if (retry >= 10) ESP_LOGW(TAG, "NTP 타임아웃 → 로컬 시간 사용");
    else             ESP_LOGI(TAG, "NTP 동기화 완료");
}

/* ── UDP 소켓 초기화 ─────────────────────────────────── */
static void udp_init(void)
{
    udp_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (udp_sock < 0) {
        ESP_LOGE(TAG, "소켓 생성 실패");
        return;
    }
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family      = AF_INET;
    server_addr.sin_port        = htons(g_server_port);
    server_addr.sin_addr.s_addr = inet_addr(g_server_ip);
    ESP_LOGI(TAG, "UDP 소켓 준비 → %s:%d", g_server_ip, g_server_port);
}

/* ── CSI 초기화 ──────────────────────────────────────── */
static void csi_init(void)
{
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));

    wifi_csi_config_t csi_cfg = {
        .lltf_en           = true,
        .htltf_en          = false,
        .stbc_htltf2_en    = false,
        .ltf_merge_en      = true,
        .channel_filter_en = false,
        .manu_scale        = false,
        .shift             = 0,
    };
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_cfg));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
    ESP_LOGI(TAG, "CSI 수집 시작 — LLTF %d 서브캐리어", SUBCARRIER_VALID);
}

/* ═══════════════════════════════════════════════════════
 * app_main
 * ═══════════════════════════════════════════════════════ */
void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    nvs_load_config();

    csi_queue = xQueueCreate(CSI_QUEUE_SIZE, sizeof(csi_packet_t));
    if (!csi_queue) {
        ESP_LOGE(TAG, "큐 생성 실패");
        return;
    }

    xTaskCreate(serial_cmd_task, "serial_cmd", 4096, NULL, 3, NULL);

    wifi_init();
    sntp_sync();
    udp_init();
    csi_init();

    xTaskCreate(udp_send_task, "udp_send", 8192, NULL, 5, NULL);

    ESP_LOGI(TAG, "=== RX1 준비 완료 ===");
    ESP_LOGI(TAG, "서버: %s:%d | DevID: 0x%02X | Magic: 0x%02X | 패킷: %d bytes",
             g_server_ip, g_server_port, DEVICE_ID, PACKET_MAGIC, (int)sizeof(csi_packet_t));
}

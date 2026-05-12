/**
 * ============================================================
 * 파일명: csi_tx1_main.c
 * 역할:   ESP32-S3 TX1 (송신)
 *         100Hz (10ms 간격) UDP 패킷 송출 → RX1/RX2 CSI 수집 유도
 *
 * 빌드 전 필수:
 *   idf.py menuconfig
 *   → Component config → Wi-Fi → WiFi CSI(Channel State Information) 체크
 *
 * 명령어 (모니터 실행 중):
 *   set_wifi <SSID> <PASSWORD>
 *   set_target <IP> <PORT>
 *   set_channel <CH>
 *   show_config
 *   restart
 *   set_hz <hz>
 * ============================================================
 */

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_system.h"
#include "nvs_flash.h"
#include "nvs.h"
#include "lwip/sockets.h"
#include "lwip/netdb.h"

/* ── 고정 설정값 ─────────────────────────────────────── */
#define DEVICE_ID           0x00          // TX = 0x00
#define TX_PAYLOAD_MAGIC    0xAB

/* ── 기본값 ──────────────────────────────────────────── */
#define DEFAULT_WIFI_SSID    "coin"
#define DEFAULT_WIFI_PASS    "1q2w3e4r"
#define DEFAULT_TARGET_IP    "255.255.255.255"  // 브로드캐스트
#define DEFAULT_TARGET_PORT  5000
#define DEFAULT_WIFI_CHANNEL 6
#define DEFAULT_HZ           100                // 기본 100Hz
#define DEFAULT_INTERVAL_MS  (1000 / DEFAULT_HZ)
/* ──────────────────────────────────────────────────── */

static const char *TAG = "CSI_TX1";

/* ── NVS 설정값 ───────────────────────────────────────── */
static char g_wifi_ssid[32]   = DEFAULT_WIFI_SSID;
static char g_wifi_pass[64]   = DEFAULT_WIFI_PASS;
static char g_target_ip[16]   = DEFAULT_TARGET_IP;
static int  g_target_port     = DEFAULT_TARGET_PORT;
static int  g_wifi_channel    = DEFAULT_WIFI_CHANNEL;
// 주기를 제어할 변수 (부팅 기본 10ms = 100Hz, NVS override 비활성화)
static int  g_send_interval_ms = DEFAULT_INTERVAL_MS;

/* ── TX 전송 페이로드 구조체 (8 bytes) ───────────────── */
typedef struct {
    uint8_t  magic;        // 1 byte — 0xAB
    uint8_t  device_id;    // 1 byte — 0x00 (TX)
    uint8_t  reserved[2];  // 2 bytes — 패딩
    uint32_t seq;          // 4 bytes — 시퀀스 번호
} __attribute__((packed)) tx_packet_t;

/* ── 전역 변수 ────────────────────────────────────────── */
static int                udp_sock = -1;
static struct sockaddr_in target_addr;
static volatile uint32_t  pkt_seq  = 0;
static EventGroupHandle_t s_wifi_event_group;
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
    nvs_set_str(h, "ip",      g_target_ip);
    nvs_set_i32(h, "port",    g_target_port);
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
    len = sizeof(g_target_ip);  nvs_get_str(h, "ip",   g_target_ip, &len);
    nvs_get_i32(h, "port",    (int32_t*)&g_target_port);
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
    printf("  CSI TX1 — 명령어 목록\n");
    printf("  set_wifi <SSID> <PASSWORD>\n");
    printf("  set_target <IP> <PORT>\n");
    printf("  set_channel <CH>\n");
    printf("  set_hz <Hz>\n"); // 명령어 추가
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

            } else if (strcmp(cmd, "set_target") == 0 && n >= 3) {
                strncpy(g_target_ip, arg1, sizeof(g_target_ip)-1);
                g_target_port = atoi(arg2);
                nvs_save_config();
                printf("[OK] 타겟: %s:%d → restart 필요\n", g_target_ip, g_target_port);

            } else if (strcmp(cmd, "set_channel") == 0 && n >= 2) {
                g_wifi_channel = atoi(arg1);
                nvs_save_config();
                printf("[OK] 채널: %d → restart 필요\n", g_wifi_channel);

            /* ── Hz 변경 로직 추가 ── */
            } else if (strcmp(cmd, "set_hz") == 0 && n >= 2) {
                int hz = atoi(arg1);
                if (hz > 0 && hz <= 500) {
                    g_send_interval_ms = 1000 / hz;
                    if (g_send_interval_ms == 0) g_send_interval_ms = 1; // 최소 1ms 방어코드
                    printf("[OK] 전송 주기 %dHz (%dms) 런타임 임시 변경됨 (재부팅 시 %dms)\n",
                           hz, g_send_interval_ms, DEFAULT_INTERVAL_MS);
                } else {
                    printf("[ERR] Hz 범위 초과 (1~500)\n");
                }

            } else if (strcmp(cmd, "show_config") == 0) {
                printf("\n===== 현재 설정 =====\n");
                printf("  SSID      : %s\n", g_wifi_ssid);
                printf("  PW        : %s\n", g_wifi_pass);
                printf("  타겟 IP   : %s\n", g_target_ip);
                printf("  타겟 포트 : %d\n", g_target_port);
                printf("  채널      : %d\n", g_wifi_channel);
                printf("  DevID     : 0x%02X (TX1)\n", DEVICE_ID);
                printf("  전송 주기 : %d ms (약 %d Hz)\n", g_send_interval_ms, 1000/g_send_interval_ms);
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
 * UDP 송출 태스크
 * ═══════════════════════════════════════════════════════ */
static void udp_tx_task(void *pvParameters)
{
    tx_packet_t pkt;
    pkt.magic     = TX_PAYLOAD_MAGIC;
    pkt.device_id = DEVICE_ID;
    pkt.reserved[0] = 0x00;
    pkt.reserved[1] = 0x00;

    TickType_t last_wake = xTaskGetTickCount();

    ESP_LOGI(TAG, "UDP 송출 태스크 시작 → %s:%d @ 약 %dHz",
             g_target_ip, g_target_port, 1000/g_send_interval_ms);

    while (1) {
        pkt.seq = pkt_seq++;

        int ret = sendto(udp_sock, &pkt, sizeof(tx_packet_t), 0,
                         (struct sockaddr *)&target_addr, sizeof(target_addr));
        if (ret < 0) {
            ESP_LOGW(TAG, "UDP 전송 실패 (seq=%lu)", (unsigned long)pkt.seq);
        }

        /* NVS에 저장된 주기 적용 */
        vTaskDelayUntil(&last_wake, pdMS_TO_TICKS(g_send_interval_ms));
    }
}

/* ── WiFi 이벤트 핸들러 ──────────────────────────────── */
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

    /* Power Save 비활성화 — modem sleep로 인한 burst→idle 방지 */
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_LOGI(TAG, "WiFi Power Save 비활성화 (WIFI_PS_NONE)");
}

/* ── UDP 소켓 초기화 ─────────────────────────────────── */
static void udp_init(void)
{
    udp_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (udp_sock < 0) {
        ESP_LOGE(TAG, "소켓 생성 실패");
        return;
    }

    /* 브로드캐스트 허용 */
    int broadcast = 1;
    setsockopt(udp_sock, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast));

    memset(&target_addr, 0, sizeof(target_addr));
    target_addr.sin_family      = AF_INET;
    target_addr.sin_port        = htons(g_target_port);
    target_addr.sin_addr.s_addr = inet_addr(g_target_ip);

    ESP_LOGI(TAG, "UDP 소켓 준비 → %s:%d", g_target_ip, g_target_port);
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
    g_send_interval_ms = DEFAULT_INTERVAL_MS;
    ESP_LOGW(TAG, "TX interval fixed to %d ms (NVS override disabled)", g_send_interval_ms);

    xTaskCreate(serial_cmd_task, "serial_cmd", 4096, NULL, 3, NULL);

    wifi_init();
    udp_init();

    xTaskCreate(udp_tx_task, "udp_tx", 4096, NULL, 5, NULL);

    ESP_LOGI(TAG, "=== TX1 준비 완료 ===");
    ESP_LOGI(TAG, "타겟: %s:%d | DevID: 0x%02X | 주기: %dms | 페이로드: %d bytes",
             g_target_ip, g_target_port, DEVICE_ID,
             g_send_interval_ms, (int)sizeof(tx_packet_t));
}

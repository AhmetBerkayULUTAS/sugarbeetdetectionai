# Kullanım Rehberi

## Temel Kullanım

### 1. Jetson Nano'da Publisher Başlatma

```bash
# Basit başlatma
ros2 launch sugar_beet_detection jetson_publisher.launch.py

# USB kamera
ros2 launch sugar_beet_detection jetson_publisher.launch.py \
    camera_type:=usb \
    camera_id:=0

# CSI kamera (Jetson Nano)
ros2 launch sugar_beet_detection jetson_publisher.launch.py \
    camera_type:=csi \
    camera_id:=0

# Yüksek confidence threshold
ros2 launch sugar_beet_detection jetson_publisher.launch.py \
    conf_threshold:=0.7

# Düşük FPS (bant genişliği için)
ros2 launch sugar_beet_detection jetson_publisher.launch.py \
    publish_rate:=15

# Verbose mode (debug)
ros2 launch sugar_beet_detection jetson_publisher.launch.py \
    verbose:=true
```

### 2. Host'ta UI Node Başlatma

```bash
# Sizin mevcut UI node'unuz
ros2 run your_package ui_node.py
```

## Monitoring

### Topic'leri İzleme

```bash
# Tüm topic'leri listele
ros2 topic list

# Topic bilgisi
ros2 topic info /ai_detections

# Topic'i gerçek zamanlı izle
ros2 topic echo /ai_detections

# FPS ölç
ros2 topic hz /ai_output_image

# Bant genişliği ölç
ros2 topic bw /ai_output_image
```

### Görüntüyü Görüntüleme

```bash
# rqt_image_view ile
ros2 run rqt_image_view rqt_image_view

# Dropdown'dan /ai_output_image seç
```

### Performans İzleme

```bash
# System metrics'i izle
ros2 topic echo /system_metrics

# Sürekli güncellenen metrikler:
watch -n 1 "ros2 topic echo /system_metrics --once"
```

## Örnekler

### Örnek 1: Gerçek Zamanlı Monitoring

```bash
# Terminal 1 (Jetson)
ros2 launch sugar_beet_detection jetson_publisher.launch.py \
    send_annotated:=true

# Terminal 2 (Host)
ros2 run your_package ui_node.py

# Terminal 3 (Monitoring)
ros2 run rqt_image_view rqt_image_view
```

### Örnek 2: Performans Testing

```bash
# Jetson - Yüksek FPS
ros2 launch sugar_beet_detection jetson_publisher.launch.py \
    publish_rate:=60 \
    verbose:=true

# Host - Monitoring
ros2 topic hz /ai_detections
ros2 topic hz /ai_output_image
watch -n 0.5 "ros2 topic echo /system_metrics --once | grep fps"
```

### Örnek 3: Düşük Bant Genişliği (WiFi)

```bash
# Jetson
ros2 launch sugar_beet_detection jetson_publisher.launch.py \
    publish_rate:=15 \
    send_annotated:=false

# Sadece detection'ları gönder, görüntü gönderme
```

## Parametre Tablosu

| Parametre | Varsayılan | Değerler | Açıklama |
|-----------|-----------|----------|----------|
| `camera_type` | usb | usb, csi | Kamera tipi |
| `camera_id` | 0 | 0, 1, 2... | Kamera ID |
| `engine_path` | model2.engine | path | Model yolu |
| `conf_threshold` | 0.5 | 0.0-1.0 | Tespit eşiği |
| `nms_threshold` | 0.30 | 0.0-1.0 | NMS eşiği |
| `publish_rate` | 30 | 1-60 | FPS |
| `send_annotated` | true | true, false | Bbox çiz |
| `verbose` | false | true, false | Detaylı log |

## Sorun Giderme

### Düşük FPS
- `publish_rate` parametresini düşür
- Kamera çözünürlüğünü azalt
- `send_annotated:=false` kullan

### Yüksek Gecikme
- Gigabit Ethernet kullan
- QoS BEST_EFFORT kullan
- Cyclone DDS dene

### Bağlantı Sorunu
- ROS_DOMAIN_ID kontrol et
- Multicast test yap
- Firewall kontrol et

# 🌱 Şeker Pancarı Tespit Sistemi - ROS2

Jetson Nano ve Host bilgisayar arasında ROS2 ile gerçek zamanlı pancar tespit sistemi.

## 📋 Özellikler

- ✅ TensorRT ile hızlı AI inference (Jetson Nano)
- ✅ ROS2 topic'ler üzerinden iletişim
- ✅ Custom message tipleri (DetectionData, SystemMetrics)
- ✅ USB ve CSI kamera desteği
- ✅ Gerçek zamanlı performans metrikleri
- ✅ Düşük gecikme (50-100ms)
- ✅ Kolay entegrasyon

## 🔧 Gereksinimler

### Jetson Nano
- JetPack 4.6+
- ROS2 Foxy/Humble
- TensorRT
- OpenCV
- PyCUDA

### Host Bilgisayar
- ROS2 Foxy/Humble
- cv_bridge

## 📦 Kurulum

```bash
# Workspace oluştur
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Paketi kopyala
cp -r sugar_beet_detection .

# Build
cd ~/ros2_ws
colcon build --packages-select sugar_beet_detection

# Source
source install/setup.bash
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

## 🚀 Kullanım

### Jetson Nano'da

```bash
# ROS Domain ID ayarla (her iki tarafta da aynı)
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0

# Publisher'ı başlat
ros2 launch sugar_beet_detection jetson_publisher.launch.py

# Parametrelerle:
ros2 launch sugar_beet_detection jetson_publisher.launch.py \
    camera_type:=csi \
    conf_threshold:=0.5 \
    verbose:=true
```

### Host Bilgisayarda

```bash
# ROS Domain ID ayarla
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0

# UI Node'unuzu başlatın (mevcut node'unuz)
ros2 run your_package ui_node.py
```

## 📡 Topic'ler

| Topic | Type | Açıklama |
|-------|------|----------|
| `/ai_output_image` | sensor_msgs/Image | Görselleştirilmiş görüntü |
| `/ai_detections` | DetectionData | Tespit sonuçları |
| `/system_metrics` | SystemMetrics | Performans metrikleri |
| `/fps` | std_msgs/Int32 | FPS |
| `/ai_inference_time` | std_msgs/Int32 | Inference süresi |
| `/camera1_check` | std_msgs/Bool | Kamera durumu |

## 🔍 Debug

```bash
# Topic'leri listele
ros2 topic list

# Topic'i izle
ros2 topic echo /ai_detections

# Görüntüyü görüntüle
ros2 run rqt_image_view rqt_image_view
```

## 📊 Performans

| Ağ | FPS | Gecikme |
|-----|-----|---------|
| Gigabit Ethernet | 25-30 | 50-80ms |
| 100 Mbps | 20-25 | 80-120ms |
| WiFi 5GHz | 15-20 | 100-150ms |

## 📝 Lisans

MIT License

## 🤝 Katkı

Pull request'ler memnuniyetle karşılanır!

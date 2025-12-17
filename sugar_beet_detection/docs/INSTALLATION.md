# Kurulum Rehberi

## 1. Ön Gereksinimler

### Jetson Nano
```bash
# ROS2 kurulu mu kontrol et
ros2 --version

# Gerekli paketler
sudo apt update
sudo apt install -y \
    python3-pip \
    python3-opencv \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-sensor-msgs

# Python paketleri
pip3 install numpy pycuda
```

### Host Bilgisayar
```bash
# ROS2 ve cv_bridge
sudo apt install -y \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-rqt-image-view
```

## 2. Workspace Kurulumu

```bash
# Workspace oluştur
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Paketi kopyala
# ZIP dosyasını buraya çıkarın veya:
git clone <repo-url> sugar_beet_detection

# Model dosyasını kopyala
cp /path/to/model2.engine sugar_beet_detection/models/

# Mevcut dosyalarınızı kopyala
cp /path/to/detector.py sugar_beet_detection/scripts/
cp /path/to/camera.py sugar_beet_detection/scripts/
cp /path/to/metrics.py sugar_beet_detection/scripts/
cp /path/to/visualizer.py sugar_beet_detection/scripts/

# Executable yap
chmod +x sugar_beet_detection/scripts/*.py
chmod +x sugar_beet_detection/launch/*.py
```

## 3. Build

```bash
cd ~/ros2_ws

# Build (sadece bu paket)
colcon build --packages-select sugar_beet_detection

# Veya tüm workspace
colcon build

# Source
source install/setup.bash

# Otomatik source için
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## 4. Test

```bash
# Message tiplerini kontrol et
ros2 interface list | grep sugar_beet
ros2 interface show sugar_beet_detection/msg/DetectionData
ros2 interface show sugar_beet_detection/msg/SystemMetrics

# Node'un çalıştığını kontrol et
ros2 run sugar_beet_detection ai_publisher_node.py --help
```

## 5. Ağ Ayarları (İki Bilgisayar İçin)

### Her İki Makinede

```bash
# ROS Domain ID (aynı olmalı)
export ROS_DOMAIN_ID=42

# Localhost dışında iletişim
export ROS_LOCALHOST_ONLY=0

# .bashrc'ye ekle
echo "export ROS_DOMAIN_ID=42" >> ~/.bashrc
echo "export ROS_LOCALHOST_ONLY=0" >> ~/.bashrc
```

### Multicast Test

```bash
# Jetson'da
ros2 multicast receive

# Host'ta (başka terminal)
ros2 multicast send

# Mesaj görünmeli: "Received from ..."
```

## Sorun Giderme

### Build hataları
```bash
# Bağımlılıkları kontrol et
rosdep install --from-paths src --ignore-src -r -y

# Temiz build
rm -rf build install log
colcon build --packages-select sugar_beet_detection
```

### Runtime hataları
```bash
# Python path kontrol
which python3
python3 --version

# PYTHONPATH kontrol
echo $PYTHONPATH

# cv_bridge test
python3 -c "from cv_bridge import CvBridge; print('OK')"
```

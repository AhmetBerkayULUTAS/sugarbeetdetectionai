# Proje Yapısı

```
sugar_beet_detection/
│
├── CMakeLists.txt                     # CMake build dosyası
├── package.xml                        # ROS2 paket bilgileri
├── README.md                          # Proje açıklaması
│
├── msg/                               # Custom message tipleri
│   ├── DetectionData.msg             # Tespit sonuçları mesajı
│   └── SystemMetrics.msg             # Performans metrikleri mesajı
│
├── scripts/                           # Python script'leri
│   ├── ai_publisher_node.py          # Ana AI publisher node (Jetson)
│   ├── detector.py                   # TensorRT detector (mevcut)
│   ├── camera.py                     # Kamera sınıfı (mevcut)
│   ├── metrics.py                    # Metrik hesaplama (mevcut)
│   └── visualizer.py                 # Görselleştirme (mevcut)
│
├── launch/                            # Launch dosyaları
│   └── jetson_publisher.launch.py    # Jetson publisher launcher
│
├── models/                            # AI modelleri
│   └── model2.engine                 # TensorRT model (buraya koyun)
│
├── config/                            # Konfigürasyon dosyaları
│   └── (opsiyonel yaml dosyaları)
│
└── docs/                              # Dökümanlar
    ├── STRUCTURE.md                  # Bu dosya
    ├── INSTALLATION.md               # Kurulum rehberi
    └── USAGE.md                      # Kullanım rehberi
```

## Dosya Açıklamaları

### Core Files
- `ai_publisher_node.py`: Jetson Nano'da çalışan ana node. Kameradan görüntü alır, AI ile işler ve ROS2'ye yayınlar.

### Message Files
- `DetectionData.msg`: Bounding box'lar, confidence skorları ve sınıf ID'leri içerir
- `SystemMetrics.msg`: FPS, latency, inference time gibi performans metrikleri

### Existing Files (Projenizden)
- `detector.py`: TensorRT ile inference
- `camera.py`: USB/CSI kamera yönetimi  
- `metrics.py`: Performans metrikleri
- `visualizer.py`: Bounding box çizimi

### Launch Files
- `jetson_publisher.launch.py`: Tüm parametreleri ayarlayarak node'u başlatır

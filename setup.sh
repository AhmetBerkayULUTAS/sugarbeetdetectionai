#!/bin/bash
set -e

echo "=========================================="
echo "🚜 PANCAR TESPİT SİSTEMİ KURULUMU (Jetson Nano)"
echo "=========================================="

PROJECT_DIR=$(pwd)
VENV_DIR="$PROJECT_DIR/venv"

echo "📁 Proje dizini: $PROJECT_DIR"
echo "🐍 Virtual Environment: $VENV_DIR"

# --- Sistem bağımlılıkları ---
echo "📦 Sistem paketleri kuruluyor..."
sudo apt update
sudo apt install -y \
    python3-venv python3-dev \
    build-essential libboost-all-dev \
    cmake git wget curl unzip \
    zlib1g-dev pkg-config \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libgtk-3-dev libcanberra-gtk* \
    libxvidcore-dev libx264-dev \
    libtbb-dev libv4l-dev v4l-utils qv4l2 \
    libopenblas-dev libatlas-base-dev libblas-dev \
    liblapack-dev liblapacke-dev libeigen3-dev gfortran \
    libhdf5-dev libprotobuf-dev protobuf-compiler \
    libgoogle-glog-dev libgflags-dev
echo "✅ Sistem bağımlılıkları kuruldu."

# --- CUDA kontrolü ---
if [ -d "/usr/local/cuda" ]; then
    CUDA_VERSION=$(cat /usr/local/cuda/version.txt 2>/dev/null || echo "Bilinmeyen")
    echo "✅ CUDA bulundu: $CUDA_VERSION"
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
else
    echo "❌ CUDA bulunamadı! Lütfen JetPack kurulumunu kontrol et."
    exit 1
fi

# --- VENV ---
if [ ! -d "$VENV_DIR" ]; then
    echo "🐍 Virtual Environment oluşturuluyor..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

echo "🎯 Python versiyonu: $(python --version)"
pip install --upgrade pip setuptools wheel

# ======================================================================
# === OpenCV 4.9.0 Derleme Fonksiyonu ===
# ======================================================================
install_opencv () {
    echo "🔧 OpenCV kurulumu başlıyor..."
    cd ~
    sudo rm -rf opencv* 
    wget -O opencv.zip https://github.com/opencv/opencv/archive/4.9.0.zip
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.9.0.zip
    unzip -q opencv.zip && unzip -q opencv_contrib.zip
    mv opencv-4.9.0 opencv && mv opencv_contrib-4.9.0 opencv_contrib
    mkdir -p ~/opencv/build && cd ~/opencv/build

    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr \
          -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
          -D WITH_CUDA=ON -D WITH_CUDNN=ON -D WITH_CUBLAS=ON \
          -D ENABLE_FAST_MATH=ON -D CUDA_FAST_MATH=ON \
          -D OPENCV_DNN_CUDA=ON -D WITH_V4L=ON -D WITH_GSTREAMER=ON \
          -D WITH_TBB=ON -D BUILD_EXAMPLES=OFF \
          -D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
          -D OPENCV_ENABLE_NONFREE=ON ..

    make -j2
    sudo make install
    sudo ldconfig
    echo "🎉 OpenCV 4.9.0 başarıyla kuruldu!"
}

# --- OpenCV kontrol & çağırma ---
if [ -d ~/opencv/build ]; then
    read -p "📁 ~/opencv/build mevcut. Yeniden kurulsun mu? (Y/n): " answer
    if [[ "$answer" =~ ^[Nn]$ ]]; then
        echo "➡️ OpenCV kurulumu atlandı."
    else
        install_opencv
    fi
else
    install_opencv
fi

# ======================================================================
# === stddef.h Kontrol ve Düzeltme ===
# ======================================================================
echo "🔍 stddef.h başlık dosyası kontrol ediliyor..."
if [ ! -f /usr/include/stddef.h ]; then
    echo "⚠️  /usr/include/stddef.h bulunamadı!"
    FOUND_STDDEF=$(sudo find /usr/include -name "stddef.h" 2>/dev/null | grep linux | head -n 1)
    if [ -n "$FOUND_STDDEF" ]; then
        echo "✅ Alternatif bulundu: $FOUND_STDDEF"
        echo "📎 Sembolik link oluşturuluyor..."
        sudo ln -sf "$FOUND_STDDEF" /usr/include/stddef.h
        echo "🔗 Link oluşturuldu: /usr/include/stddef.h -> $FOUND_STDDEF"
    else
        echo "❌ Uygun stddef.h bulunamadı, PyCUDA kurulumu başarısız olabilir."
    fi
else
    echo "✅ /usr/include/stddef.h mevcut."
fi

# ======================================================================
# === Python paketleri ve PyCUDA kurulumu ===
# ======================================================================
cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"

pip install numpy==1.19.5 PyYAML tqdm appdirs typing-extensions Cython

echo "🚀 PyCUDA (2020.1) kurulumu başlıyor..."
pip install Cython pytools
pip install pycuda==2020.1 || echo "⚠️ PyCUDA kurulamadı, lütfen CUDA'yı kontrol edin."

# ======================================================================
# === TensorRT kontrolü ===
# ======================================================================
echo "🧠 TensorRT kütüphaneleri kontrol ediliyor..."
sudo apt install -y --no-install-recommends \
    libnvinfer-dev libnvinfer-bin libnvinfer-plugin-dev \
    libnvparsers-dev libnvonnxparsers-dev || echo "ℹ️ TensorRT zaten kurulu."

# ======================================================================
# === Doğrulama ===
# ======================================================================
echo "=========================================="
echo "🔍 Kurulum doğrulanıyor..."
python - <<'EOF'
try:
    import pycuda.driver as cuda
    print("✅ PyCUDA OK")
except Exception as e:
    print("❌ PyCUDA hata:", e)
try:
    import cv2
    print(f"✅ OpenCV {cv2.__version__}")
except Exception as e:
    print("❌ OpenCV hata:", e)
try:
    import tensorrt
    print(f"✅ TensorRT {tensorrt.__version__}")
except Exception as e:
    print("❌ TensorRT hata:", e)
EOF
echo "------------------------------------------"
echo "🎉 TÜM KURULUM TAMAMLANDI"
echo "VENV: $VENV_DIR"
echo "CUDA: $(nvcc --version | grep release)"
echo "=========================================="

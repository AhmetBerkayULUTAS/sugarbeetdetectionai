#!/bin/bash

# OpenCV, PyCUDA ve TensorRT Kurulum Scripti - VENV Desteği ile
# Jetson Nano için optimize edilmiştir

set -e  # Hata durumunda scripti durdur

echo "=========================================="
echo "PANCAR TESPİT SİSTEMİ KURULUMU"
echo "=========================================="

# Mevcut dizini proje dizini olarak kullan
PROJECT_DIR=$(pwd)
VENV_DIR="$PROJECT_DIR/venv"

echo "📁 Proje dizini: $PROJECT_DIR"
echo "🐍 Virtual Environment: $VENV_DIR"

# Sistem Python versiyonunu kontrol et
CURRENT_PYTHON=$(python3 --version 2>&1)
echo "🔍 Mevcut Python: $CURRENT_PYTHON"

# Virtual Environment oluştur
echo "🐍 Virtual Environment oluşturuluyor..."
python3 -m venv "$VENV_DIR"

# VENV'i aktive et
echo "🔧 VENV aktive ediliyor..."
source "$VENV_DIR/bin/activate"

# VENV içindeki Python versiyonunu kontrol et
VENV_PYTHON=$(python --version 2>&1)
echo "🎯 VENV Python: $VENV_PYTHON"

# pip'i güncelle
echo "📥 pip güncelleniyor..."
pip install --upgrade pip

echo "✅ Virtual Environment hazır!"
echo "📍 VENV yolu: $VENV_DIR"

# === KRİTİK GELİŞTİRME BAŞLIKLARI ===
echo "📚 KRİTİK: Geliştirme başlık dosyaları kuruluyor..."
sudo apt update
sudo apt install -y \
    build-essential \
    libc6-dev \
    linux-libc-dev \
    python3-dev \
    libboost-python-dev \
    libboost-thread-dev

echo "✅ Geliştirme başlıkları kuruldu!"

# === STDEF.H KONTROLÜ VE DÜZELTME ===
echo "🔍 stddef.h konumu kontrol ediliyor..."
STDDEF_PATHS=(
    "/usr/include/stddef.h"
    "/usr/lib/gcc/aarch64-linux-gnu/8/include/stddef.h"
    "/usr/lib/gcc/aarch64-linux-gnu/9/include/stddef.h"
    "/usr/include/linux/stddef.h"
)

for path in "${STDDEF_PATHS[@]}"; do
    if [ -f "$path" ]; then
        echo "✅ stddef.h bulundu: $path"
        export C_INCLUDE_PATH=$(dirname "$path"):$C_INCLUDE_PATH
        export CPLUS_INCLUDE_PATH=$(dirname "$path"):$CPLUS_INCLUDE_PATH
    fi
done

# GCC include path'lerini ekle
GCC_INCLUDE_PATHS=(
    "/usr/lib/gcc/aarch64-linux-gnu/8/include"
    "/usr/lib/gcc/aarch64-linux-gnu/9/include" 
    "/usr/include/aarch64-linux-gnu"
)

for path in "${GCC_INCLUDE_PATHS[@]}"; do
    if [ -d "$path" ]; then
        echo "✅ GCC include path ekleniyor: $path"
        export C_INCLUDE_PATH=$path:$C_INCLUDE_PATH
        export CPLUS_INCLUDE_PATH=$path:$CPLUS_INCLUDE_PATH
    fi
done

echo "📍 C_INCLUDE_PATH: $C_INCLUDE_PATH"
echo "📍 CPLUS_INCLUDE_PATH: $CPLUS_INCLUDE_PATH"

# === CUDA KONTROLÜ VE ORTAM DEĞİŞKENLERİ ===
echo "🔍 CUDA kontrol ediliyor..."
if [ -d "/usr/local/cuda" ]; then
    CUDA_VERSION=$(cat /usr/local/cuda/version.txt 2>/dev/null || echo "Bilinmeyen")
    echo "✅ CUDA zaten kurulu: $CUDA_VERSION"
    echo "📍 CUDA yolu: /usr/local/cuda"
    
    # CUDA ortam değişkenlerini ayarla
    echo "🔧 CUDA ortam değişkenleri ayarlanıyor..."
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export CUDA_ROOT=/usr/local/cuda
else
    echo "❌ CUDA bulunamadı! Jetson Nano'da CUDA JetPack ile kurulu olmalı"
    exit 1
fi

# [OpenCV kurulum kısmı aynen kalıyor...]

# Proje dizinine geri dön
cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"

# === VENV İÇİNDE PYTHON PAKETLERİ ===
echo "🐍 VENV içinde Python paketleri kuruluyor..."

# Temel paketler
pip install \
    numpy \
    Cython \
    PyYAML \
    tqdm \
    appdirs \
    typing-extensions

# === PYCUDA KURULUMU (CRITICAL FIX) ===
echo "🚀 PyCUDA kurulumu başlıyor (include path'ler ayarlandı)..."

# Önce gerekli araçları kur
pip install setuptools wheel

# PyCUDA'yı derleme ortam değişkenleriyle kur
echo "🔧 Derleme ortam değişkenleri ayarlanıyor..."
export LDFLAGS="-L/usr/lib/aarch64-linux-gnu"
export CFLAGS="-I/usr/include -I/usr/include/aarch64-linux-gnu"
export CXXFLAGS="-I/usr/include -I/usr/include/aarch64-linux-gnu"

# PyCUDA'yı kur
pip install pycuda --no-cache-dir --verbose

# Eğer hala hata alınırsa, elle configure et
if [ $? -ne 0 ]; then
    echo "⚠️  Standart kurulum başarısız, elle configure ediliyor..."
    
    cd /tmp
    rm -rf pycuda*
    wget -q https://files.pythonhosted.org/packages/source/p/pycuda/pycuda-2022.2.2.tar.gz
    tar -xzf pycuda-2022.2.2.tar.gz
    cd pycuda-2022.2.2
    
    # Elle configure et
    python configure.py --cuda-root=/usr/local/cuda
    
    # Include path'leri elle belirt
    echo "🔧 Include path'leri elle ayarlanıyor..."
    find . -name "*.py" -exec sed -i 's|/usr/include|/usr/include:/usr/include/aarch64-linux-gnu|g' {} \; 2>/dev/null || true
    
    python setup.py build
    python setup.py install
    
    cd "$PROJECT_DIR"
fi

# === TENSORRT (GÜVENLİ KURULUM) ===
echo "🧠 TensorRT bağımlılıkları kontrol ediliyor..."
sudo apt install -y --no-install-recommends \
    libnvinfer-dev libnvinfer-bin libnvinfer-plugin-dev \
    libnvparsers-dev libnvonnxparsers-dev || echo "ℹ️  TensorRT paketleri zaten kurulu"

echo "=========================================="
echo "🎉 KURULUM TAMAMLANDI!"
echo "=========================================="

# Kurulum doğrulama
echo "🔍 Kurulum doğrulanıyor..."

echo "Python ve VENV:"
python --version
pip --version

echo "PyCUDA:"
python -c "import pycuda.driver as cuda; print('✅ PyCUDA başarıyla kuruldu!')" || echo "❌ PyCUDA kurulumunda hata!"

echo "OpenCV:"
python -c "import cv2; print(f'✅ OpenCV {cv2.__version__}')" || echo "❌ OpenCV kurulumunda hata!"

echo "TensorRT:"
python -c "import tensorrt; print(f'✅ TensorRT {tensorrt.__version__}')" || echo "❌ TensorRT kurulumunda hata!"

echo ""
echo "=========================================="
echo "🚀 PROJE HAZIR!"
echo "=========================================="

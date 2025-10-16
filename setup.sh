#!/bin/bash

# OpenCV ve TensorRT Kurulum Scripti - VENV Desteği ile
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

# === SİSTEM PAKETLERİ ===
echo "📦 Sistem paketleri kuruluyor..."
sudo apt update
sudo apt install -y \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    libc6-dev \
    linux-libc-dev \
    cmake \
    git \
    wget \
    curl \
    unzip \
    pkg-config \
    zlib1g-dev

echo "✅ Sistem paketleri kuruldu!"

# === CUDA KONTROLÜ ===
echo "🔍 CUDA kontrol ediliyor..."
if [ -d "/usr/local/cuda" ]; then
    CUDA_VERSION=$(cat /usr/local/cuda/version.txt 2>/dev/null || echo "Bilinmeyen")
    echo "✅ CUDA zaten kurulu: $CUDA_VERSION"
    echo "📍 CUDA yolu: /usr/local/cuda"
    
    # CUDA ortam değişkenlerini ayarla
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
else
    echo "❌ CUDA bulunamadı! Jetson Nano'da CUDA JetPack ile kurulu olmalı"
    exit 1
fi

# === PYCUDA WHEEL İNDİRME ===
echo "🚀 PyCUDA wheel indiriliyor..."
cd /tmp

# PyCUDA wheel dosyasını indir
PYCUDA_WHEEL="pycuda-2020.1-cp36-cp36m-linux_aarch64.whl"
echo "📥 PyCUDA wheel indiriliyor: $PYCUDA_WHEEL"

# Farklı kaynaklardan deneyelim
wget -O "$PYCUDA_WHEEL" "https://github.com/jetson-nano-wheels/python3.6-pycuda/releases/download/v1.0/pycuda-2020.1-cp36-cp36m-linux_aarch64.whl" || \
wget -O "$PYCUDA_WHEEL" "https://files.pythonhosted.org/packages/cp36/p/pycuda/pycuda-2020.1-cp36-cp36m-linux_aarch64.whl" || \
echo "❌ Wheel indirme başarısız, alternatif kaynak deneniyor..."

# Eğer wheel indirilemezse, alternatif yöntem
if [ ! -f "$PYCUDA_WHEEL" ]; then
    echo "🔧 Alternatif PyCUDA kaynağı deneniyor..."
    # NVIDIA'nın resmi PyCUDA wheel'ini indirmeyi dene
    wget -O "$PYCUDA_WHEEL" "https://developer.download.nvidia.com/assets/cuda/files/pycuda-2020.1-cp36-cp36m-linux_aarch64.whl" || \
    echo "⚠️  Wheel indirilemedi, pip ile kurulum deneneyecek"
fi

# Proje dizinine geri dön
cd "$PROJECT_DIR"

# === OpenCV Kurulum Fonksiyonu ===
install_opencv() {
    echo "=========================================="
    echo "OpenCV 4.9.0 Kurulumu Başlıyor"
    echo "=========================================="
    
    # Sistem kaynaklarını kontrol et
    CPU_CORES=$(nproc)
    SWAP_MEM=$(free -m | awk '/^Swap:/{print $2}')
    RAM_MEM=$(free -m | awk '/^Mem:/{print $2}')
    
    echo "💻 Sistem Kaynakları:"
    echo "- CPU Çekirdekleri: $CPU_CORES"
    echo "- RAM: ${RAM_MEM}MB"
    echo "- Swap: ${SWAP_MEM}MB"
    
    # Jetson model kontrolü
    if [ -e "/proc/device-tree/model" ]; then
        model=$(tr -d '\0' < /proc/device-tree/model)
        echo ""
        if [[ $model == *"Orin"* ]]; then
            echo "Detecting a Jetson Nano Orin."
            NO_JOB=4
            ARCH=8.7
            PTX="sm_87"
            echo "Jetson Orin tespit edildi. Derleme 4 çekirdek ile yapılacak."
        elif [[ $model == *"Jetson Nano"* ]]; then
            echo "Detecting a regular Jetson Nano."
            # Check GCC version
            GCC_MAJOR_VERSION=$(gcc -dumpversion | cut -d. -f1)
            if [[ "$GCC_MAJOR_VERSION" -ge 9 ]]; then
                echo ""
                echo "Detected GCC version $GCC_MAJOR_VERSION, which is too new for Jetson Nano CUDA compatibility."
                echo "OpenCV will fail to compile with this version."
                echo ""
                
                if [ -x /usr/bin/gcc-8 ] && [ -x /usr/bin/g++-8 ]; then
                    echo "GCC 8 is available on your system."
                    
                    printf "Do you want to temporarily switch to GCC 8 for this installation (Y/n)? "
                    read confirm_switch
                    
                    if [[ "$confirm_switch" != "${confirm_switch#[Nn]}" ]]; then
                        echo "Aborting installation as requested."
                        exit 1
                    fi
                    
                    echo "Switching to GCC 8..."
                    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 80
                    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 80
                    sudo update-alternatives --set gcc /usr/bin/gcc-8
                    sudo update-alternatives --set g++ /usr/bin/g++-8
                else
                    echo "GCC 8 is not installed. Please install it using:"
                    echo "  sudo apt-get install gcc-8 g++-8"
                    exit 1
                fi
            fi
            ARCH=5.3
            PTX="sm_53"
            
            # Swap kontrolü - 5.5GB altındaysa 2 çekirdek
            if [[ $SWAP_MEM -gt 5500 ]]; then
                NO_JOB=4
                echo "Yüksek swap alanı tespit edildi (>5.5GB). Derleme 4 çekirdek ile yapılacak."
            else
                NO_JOB=2
                echo "Swap alanı 5.5GB altında. Derleme 2 çekirdek ile yapılacak."
            fi
            
        else
            echo "Unable to determine the Jetson Nano model."
            exit 1
        fi
        echo ""
    else
        echo "Error: /proc/device-tree/model not found. Are you sure this is a Jetson Nano?"
        exit 1
    fi
    
    echo "🔄 OpenCV 4.9.0 kuruluyor..."
    echo "⏰ Bu işlem 2-3 saat sürebilir!"
    
    # CUDA location
    cd ~
    sudo sh -c "echo '/usr/local/cuda/lib64' >> /etc/ld.so.conf.d/nvidia-tegra.conf"
    sudo ldconfig
    
    # Jetson Nano dependencies
    if [[ $model == *"Jetson Nano"* ]]; then
        sudo apt-get install -y build-essential git unzip pkg-config zlib1g-dev
        sudo apt-get install -y python3-dev python3-numpy
        sudo apt-get install -y gstreamer1.0-tools libgstreamer-plugins-base1.0-dev
        sudo apt-get install -y libgstreamer-plugins-good1.0-dev
        sudo apt-get install -y libtbb2 libgtk-3-dev libxine2-dev
    fi
    
    # OS version specific dependencies
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        VERSION_MAJOR=$(echo "$VERSION_ID" | cut -d'.' -f1)
        if [ "$VERSION_MAJOR" = "22" ]; then
            sudo apt-get install -y libswresample-dev libdc1394-dev
        else
            sudo apt-get install -y libavresample-dev libdc1394-22-dev
        fi
    else
        sudo apt-get install -y libavresample-dev libdc1394-22-dev
    fi

    # Common dependencies
    sudo apt-get install -y cmake
    sudo apt-get install -y libjpeg-dev libjpeg8-dev libjpeg-turbo8-dev
    sudo apt-get install -y libpng-dev libtiff-dev libglew-dev
    sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev
    sudo apt-get install -y libgtk2.0-dev libgtk-3-dev libcanberra-gtk*
    sudo apt-get install -y python3-pip
    sudo apt-get install -y libxvidcore-dev libx264-dev
    sudo apt-get install -y libtbb-dev libxine2-dev
    sudo apt-get install -y libv4l-dev v4l-utils qv4l2
    sudo apt-get install -y libtesseract-dev libpostproc-dev
    sudo apt-get install -y libvorbis-dev
    sudo apt-get install -y libfaac-dev libmp3lame-dev libtheora-dev
    sudo apt-get install -y libopencore-amrnb-dev libopencore-amrwb-dev
    sudo apt-get install -y libopenblas-dev libatlas-base-dev libblas-dev
    sudo apt-get install -y liblapack-dev liblapacke-dev libeigen3-dev gfortran
    sudo apt-get install -y libhdf5-dev libprotobuf-dev protobuf-compiler
    sudo apt-get install -y libgoogle-glog-dev libgflags-dev
 
    # remove old versions or previous builds
    cd ~ 
    sudo rm -rf opencv*
    # download the 4.9.0 version
    wget -O opencv.zip https://github.com/opencv/opencv/archive/4.9.0.zip 
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.9.0.zip 
    # unpack
    unzip opencv.zip 
    unzip opencv_contrib.zip 
    # Some administration to make life easier later on
    mv opencv-4.9.0 opencv
    mv opencv_contrib-4.9.0 opencv_contrib
    # clean up the zip files
    rm opencv.zip
    rm opencv_contrib.zip
    
    # set install dir
    cd ~/opencv
    mkdir build
    cd build
    
    # run cmake
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
    -D WITH_OPENCL=OFF \
    -D CUDA_ARCH_BIN=${ARCH} \
    -D CUDA_ARCH_PTX=${PTX} \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D WITH_CUBLAS=ON \
    -D ENABLE_FAST_MATH=ON \
    -D CUDA_FAST_MATH=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D ENABLE_NEON=ON \
    -D WITH_QT=OFF \
    -D WITH_OPENMP=ON \
    -D BUILD_TIFF=ON \
    -D WITH_FFMPEG=ON \
    -D WITH_GSTREAMER=ON \
    -D WITH_TBB=ON \
    -D BUILD_TBB=ON \
    -D BUILD_TESTS=OFF \
    -D WITH_EIGEN=ON \
    -D WITH_V4L=ON \
    -D WITH_LIBV4L=ON \
    -D WITH_PROTOBUF=ON \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D BUILD_EXAMPLES=OFF \
    -D CMAKE_CXX_FLAGS="-march=native -mtune=native" \
    -D CMAKE_C_FLAGS="-march=native -mtune=native" ..
    
    echo "OpenCV derleniyor (bu işlem 2-3 saat sürebilir)..."
    echo "Kullanılan çekirdek sayısı: $NO_JOB"
    make -j ${NO_JOB} 
    
    directory="/usr/include/opencv4/opencv2"
    if [ -d "$directory" ]; then
        # Directory exists, so delete it
        sudo rm -rf "$directory"
    fi
    
    sudo make install
    sudo ldconfig
    
    # cleaning (frees 320 MB)
    make clean
    sudo apt-get update
    
    echo "OpenCV 4.9.0 başarıyla kuruldu!"
}

# OpenCV kurulumunu kontrol et ve çalıştır
cd ~

if [ -d ~/opencv/build ]; then
    echo " "
    echo "You have a directory ~/opencv/build on your disk."
    echo "Continuing the installation will replace this folder."
    echo " "
    
    printf "Do you wish to continue (Y/n)? "
    read answer

    if [ "$answer" != "${confirm_switch#[Nn]}" ]; then 
        echo "OpenCV kurulumu atlanıyor"
    else
        install_opencv
    fi
else
    install_opencv
fi

# Proje dizinine geri dön
cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"

# === VENV İÇİNDE PYTHON PAKETLERİ ===
echo "🐍 VENV içinde Python paketleri kuruluyor..."

# Temel paketler
pip install \
    numpy \
    PyYAML \
    tqdm \
    appdirs \
    typing-extensions

# === PYCUDA WHEEL KURULUMU ===
echo "🚀 PyCUDA wheel kurulumu..."

# Wheel dosyasını kontrol et ve kur
if [ -f "/tmp/pycuda-2020.1-cp36-cp36m-linux_aarch64.whl" ]; then
    echo "📦 PyCUDA wheel dosyası bulundu, kuruluyor..."
    pip install /tmp/pycuda-2020.1-cp36-cp36m-linux_aarch64.whl
else
    echo "⚠️  Wheel dosyası bulunamadı, pip ile kurulum deneniyor..."
    pip install pycuda
fi

# === TENSORRT ===
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
python -c "import cv2; print(f'✅ OpenCV {cv2.__version__}')"

echo "TensorRT:"
python -c "import tensorrt; print(f'✅ TensorRT {tensorrt.__version__}')"

echo ""
echo "=========================================="
echo "🚀 PROJE HAZIR!"
echo "=========================================="

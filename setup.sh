#!/bin/bash
set -e

echo "=========================================="
echo "🚜 PANCAR TESPİT SİSTEMİ KURULUMU (Jetson Nano)"
echo "=========================================="

PROJECT_DIR=$(pwd)
VENV_DIR="$PROJECT_DIR/venv"

echo "📁 Proje dizini: $PROJECT_DIR"
echo "🐍 Virtual Environment: $VENV_DIR"

# --- VENV ---
if [ ! -d "$VENV_DIR" ]; then
    echo "🐍 Virtual Environment oluşturuluyor..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "🎯 Python versiyonu: $(python --version)"
pip install --upgrade pip setuptools wheel

# --- Sistem bağımlılıkları ---
echo "📦 Sistem paketleri kuruluyor..."
sudo apt update
sudo apt install -y \
    python3-dev \
    build-essential \
    libboost-all-dev \
    cmake \
    git \
    wget \
    curl \
    unzip \
    zlib1g-dev \
    pkg-config
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

# ======================================================================
# === OpenCV 4.9.0 Derleme Fonksiyonu (Senin fonksiyonun) ===
# ======================================================================
install_opencv () {
  # Check if the file /proc/device-tree/model exists
  if [ -e "/proc/device-tree/model" ]; then
      model=$(tr -d '\0' < /proc/device-tree/model)
      echo ""
      if [[ $model == *"Orin"* ]]; then
          echo "Detecting a Jetson Nano Orin."
          NO_JOB=4
          ARCH=8.7
          PTX="sm_87"
      elif [[ $model == *"Jetson Nano"* ]]; then
          echo "Detecting a regular Jetson Nano."
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
          FREE_MEM="$(free -m | awk '/^Swap/ {print $2}')"
          if [[ "$FREE_MEM" -gt "5500" ]]; then
            NO_JOB=4
          else
            echo "Due to limited swap, make only uses 1 core"
            NO_JOB=1
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
  
  echo "Installing OpenCV 4.9.0 on your Nano"
  echo "It will take 3.5 hours !"
  
  cd ~
  sudo sh -c "echo '/usr/local/cuda/lib64' >> /etc/ld.so.conf.d/nvidia-tegra.conf"
  sudo ldconfig
  
  if [[ $model == *"Jetson Nano"* ]]; then
    sudo apt-get install -y build-essential git unzip pkg-config zlib1g-dev
    sudo apt-get install -y python3-dev python3-numpy
    sudo apt-get install -y python-dev python-numpy
    sudo apt-get install -y gstreamer1.0-tools libgstreamer-plugins-base1.0-dev
    sudo apt-get install -y libgstreamer-plugins-good1.0-dev
    sudo apt-get install -y libtbb2 libgtk-3-dev libxine2-dev
  fi
  
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

  sudo apt-get install -y cmake
  sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev libglew-dev
  sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev
  sudo apt-get install -y libgtk2.0-dev libgtk-3-dev libcanberra-gtk*
  sudo apt-get install -y python3-pip
  sudo apt-get install -y libxvidcore-dev libx264-dev
  sudo apt-get install -y libtbb-dev libv4l-dev v4l-utils qv4l2
  sudo apt-get install -y libtesseract-dev libpostproc-dev libvorbis-dev
  sudo apt-get install -y libfaac-dev libmp3lame-dev libtheora-dev
  sudo apt-get install -y libopenblas-dev libatlas-base-dev libblas-dev
  sudo apt-get install -y liblapack-dev liblapacke-dev libeigen3-dev gfortran
  sudo apt-get install -y libhdf5-dev libprotobuf-dev protobuf-compiler
  sudo apt-get install -y libgoogle-glog-dev libgflags-dev
  
  cd ~
  sudo rm -rf opencv*
  wget -O opencv.zip https://github.com/opencv/opencv/archive/4.9.0.zip 
  wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.9.0.zip 
  unzip opencv.zip 
  unzip opencv_contrib.zip 
  mv opencv-4.9.0 opencv
  mv opencv_contrib-4.9.0 opencv_contrib
  rm opencv.zip
  rm opencv_contrib.zip
  
  cd ~/opencv
  mkdir build
  cd build
  
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
 
  make -j ${NO_JOB} 
  
  directory="/usr/include/opencv4/opencv2"
  if [ -d "$directory" ]; then
    sudo rm -rf "$directory"
  fi
  
  sudo make install
  sudo ldconfig
  make clean
  sudo apt-get update
  
  echo "🎉 OpenCV 4.9.0 başarıyla kuruldu!"
}

# --- OpenCV kontrol & çağırma ---
cd ~
if [ -d ~/opencv/build ]; then
  echo " "
  echo "You have a directory ~/opencv/build on your disk."
  echo "Continuing the installation will replace this folder."
  echo " "
  printf "Do you wish to continue (Y/n)? "
  read answer
  if [ "$answer" != "${answer#[Nn]}" ]; then 
      echo "Leaving without installing OpenCV"
  else
      install_opencv
  fi
else
  install_opencv
fi

# ======================================================================
# === Python paketleri ve PyCUDA kurulumu ===
# ======================================================================
cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"

echo "🐍 Python paketleri kuruluyor..."
pip install numpy==1.19.5 PyYAML tqdm appdirs typing-extensions Cython

echo "🚀 PyCUDA kurulumu başlıyor..."

# Önce PyCUDA'nın zaten kurulu olup olmadığını kontrol et
if python -c "import pycuda.driver" &>/dev/null; then
    echo "✅ PyCUDA zaten kurulu."
else
    echo "🔧 PyCUDA 2020.1 kurulumu deneniyor (Python 3.6 uyumlu)..."
    
    # 1. YÖNTEM: Direkt pip ile 2020.1 sürümü
    echo "📦 Yöntem 1: pip ile pycuda==2020.1"
    pip install pycuda==2020.1
    
    # 2. YÖNTEM: Eğer wheel bulunamazsa, source'tan derle
    if [ $? -ne 0 ]; then
        echo "⚠️  Wheel bulunamadı, source'tan derleme deneniyor..."
        echo "📦 Yöntem 2: Cython ve pytools kurulumu"
        pip install Cython pytools
        
        echo "🔨 PyCUDA source'tan derleniyor..."
        pip install pycuda==2020.1 --no-binary=:all:
    fi
    
    # 3. YÖNTEM: Eğer hala başarısız olursa, git'ten clone et ve configure et
    if [ $? -ne 0 ]; then
        echo "⚠️  Pip kurulumu başarısız, git'ten clone ediliyor..."
        echo "📦 Yöntem 3: Git clone ve manuel kurulum"
        
        cd "$PROJECT_DIR"
        if [ ! -d "pycuda" ]; then
            git clone https://github.com/inducer/pycuda.git
        fi
        cd pycuda
        git checkout v2020.1  # 2020.1 sürümüne geç
        
        echo "⚙️  Configure çalıştırılıyor..."
        python configure.py --cuda-root=/usr/local/cuda
        
        echo "🔨 Derleme başlatılıyor..."
        python setup.py install
        
        cd "$PROJECT_DIR"
    fi
fi

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
echo "------------------------------------------"

echo "PyCUDA:"
python -c "import pycuda.driver as cuda; print('✅ PyCUDA başarıyla kuruldu!')" || echo "❌ PyCUDA kurulumunda hata!"

echo "OpenCV:"
python -c "import cv2; print(f'✅ OpenCV {cv2.__version__}')" || echo "❌ OpenCV kurulumunda hata!"

echo "TensorRT:"
python -c "import tensorrt; print(f'✅ TensorRT {tensorrt.__version__}')" || echo "❌ TensorRT kurulumunda hata!"

echo "NumPy:"
python -c "import numpy; print(f'✅ NumPy {numpy.__version__}')" || echo "❌ NumPy kurulumunda hata!"

echo "------------------------------------------"
echo "🎉 TÜM KURULUM TAMAMLANDI"
echo "📍 VENV: $VENV_DIR"
echo "🐍 Python: $(python --version)"
echo "🔧 CUDA: $(nvcc --version | grep release || echo 'NVCC bulunamadı')"
echo "=========================================="

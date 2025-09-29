import tensorrt as trt
import os

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)  # ✅ Daha detaylı log

def build_engine(onnx_file_path, engine_file_path):
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"🔍 ONNX modeli yükleniyor: {onnx_file_path}")
    
    with open(onnx_file_path, "rb") as f:
        if not parser.parse(f.read()):
            print("❌ ONNX parsing failed!")
            for i in range(parser.num_errors):
                print(f"Error {i}: {parser.get_error(i)}")
            return False

    print("✅ ONNX modeli başarıyla parse edildi")
    
    # Network bilgilerini yazdır
    print(f"📊 Network bilgisi:")
    print(f"  - Input sayısı: {network.num_inputs}")
    print(f"  - Output sayısı: {network.num_outputs}")
    
    for i in range(network.num_inputs):
        input = network.get_input(i)
        print(f"  - Input {i}: {input.name}, shape: {input.shape}")
    
    for i in range(network.num_outputs):
        output = network.get_output(i)
        print(f"  - Output {i}: {output.name}, shape: {output.shape}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    # YOLOv8 için optimizasyon profili
    profile = builder.create_optimization_profile()
    
    # Input shape'ini ayarla (YOLOv8 genellikle dynamic batch destekler)
    input_name = network.get_input(0).name
    input_shape = network.get_input(0).shape
    
    # Dynamic shape için min/opt/max değerleri
    profile.set_shape(input_name, 
                     (1, 3, 640, 640),   # min
                     (1, 3, 640, 640),   # opt  
                     (1, 3, 640, 640))   # max
    config.add_optimization_profile(profile)

    print("🔨 TensorRT engine oluşturuluyor...")
    
    # TRT10 için
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("❌ Engine oluşturulamadı!")
        return False

    # Serialize edilmiş engine'i dosyaya yaz
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    
    print(f"✅ Engine başarıyla oluşturuldu: {engine_file_path}")
    
    # Engine bilgilerini kontrol et
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_file_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
        
    print("🔍 Engine bilgisi:")
    print(f"  - Tensor sayısı: {engine.num_io_tensors}")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        print(f"  - Tensor {i}: {name}, mode: {mode}, shape: {shape}, dtype: {dtype}")
    
    return True

if __name__ == "__main__":
    onnx_path = "model1.onnx"
    engine_path = "model1.engine"
    
    if not os.path.exists(onnx_path):
        print(f"❌ ONNX dosyası bulunamadı: {onnx_path}")
    else:
        print(f"📁 ONNX dosyası bulundu: {os.path.getsize(onnx_path)} bytes")
        
        if build_engine(onnx_path, engine_path):
            print("🎉 Model dönüşümü başarılı!")
        else:
            print("💥 Model dönüşümü başarısız!")
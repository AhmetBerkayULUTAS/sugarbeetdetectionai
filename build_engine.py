import tensorrt as trt
import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path):
    print("🔧 TensorRT ENGINE BUILDER (Jetson Nano uyumlu)")

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"🔍 ONNX modeli yükleniyor: {onnx_file_path}")

    with open(onnx_file_path, "rb") as f:
        if not parser.parse(f.read()):
            print("❌ ONNX parsing başarısız!")
            for i in range(parser.num_errors):
                print(f"Error {i}: {parser.get_error(i)}")
            return False

    print("✅ ONNX modeli başarıyla parse edildi")

    # Network bilgileri
    print(f"📊 Network bilgisi:")
    print(f"  - Input sayısı: {network.num_inputs}")
    print(f"  - Output sayısı: {network.num_outputs}")

    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        print(f"  - Input {i}: {input_tensor.name}, shape: {input_tensor.shape}")

    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        print(f"  - Output {i}: {output_tensor.name}, shape: {output_tensor.shape}")

    # Builder Config
    config = builder.create_builder_config()
    workspace_bytes = 1 << 30  # 1GB

    # ✅ TensorRT sürümüne göre doğru fonksiyonu kullan
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        print("✅ Workspace ayarlandı (set_memory_pool_limit, TRT ≥ 8.5)")
    else:
        config.max_workspace_size = workspace_bytes
        print("✅ Workspace ayarlandı (max_workspace_size, TRT < 8.5)")

    # FP16 precision kontrolü
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✅ FP16 precision etkinleştirildi")
    else:
        print("⚠️  FP16 desteklenmiyor, FP32 kullanılacak")

    # Optimization Profile (sabit 640x640)
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    fixed_shape = (1, 3, 640, 640)
    profile.set_shape(input_name, fixed_shape, fixed_shape, fixed_shape)
    config.add_optimization_profile(profile)
    print(f"✅ Optimization profile eklendi: {fixed_shape}")

    print("🔨 TensorRT engine oluşturuluyor... (Jetson Nano'da birkaç dakika sürebilir)")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("❌ Engine oluşturulamadı! Muhtemel sebepler:")
        print("   - GPU belleği yetersiz olabilir")
        print("   - ONNX modeli uyumsuz olabilir (simplify deneyin)")
        print("   - Workspace alanını artırmayı deneyin")
        return False

    # Engine'i dosyaya kaydet
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)

    print(f"✅ Engine başarıyla oluşturuldu: {engine_file_path}")

    # Engine doğrulaması
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_file_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        print("⚠️  Engine deserialize edilemedi!")
        return False

    print("🔍 Engine bilgisi:")
    print(f"  - Tensor sayısı: {engine.num_io_tensors}")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        print(f"  - Tensor {i}: {name}, mode={mode}, shape={shape}, dtype={dtype}")

    print("✅ ENGINE BUILD TAMAMLANDI")
    return True


if __name__ == "__main__":
    onnx_path = "model1.onnx"
    engine_path = "model1.engine"

    if not os.path.exists(onnx_path):
        print(f"❌ ONNX dosyası bulunamadı: {onnx_path}")
    else:
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"📁 ONNX dosyası bulundu ({size_mb:.2f} MB)")

        if build_engine(onnx_path, engine_path):
            print("🎉 Model dönüşümü başarılı!")
        else:
            print("💥 Model dönüşümü başarısız!")

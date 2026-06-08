#include <iostream>
#include <onnxruntime_cxx_api.h>

int main() {

    // this initializes the ONNX Runtime Environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx-inference");

    Ort::SessionOptions session_options;

    session_options.SetInterOpNumThreads(1);

    // wchar_t since we are on Windows
    const wchar_t* model_path = L"onnx/resnet50-v2-7.onnx";

    Ort::Session session(env, model_path, session_options);

    std::vector<int64_t> input_shape = {1, 3, 224, 224};
    size_t input_tensor_size = 1 * 3 * 224 * 224;
    std::vector<float> input_tensor_values(input_tensor_size);

    std::fill(input_tensor_values.begin(), input_tensor_values.end(), 0.5f);

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_size,
        input_shape.data(),
        input_shape.size()
    );

    Ort::AllocatorWithDefaultOptions allocator;

    auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
    auto output_name_alloc = session.GetOutputNameAllocated(0, allocator);

    const char* input_names[] = {input_name_alloc.get()};
    const char* output_names[] = {output_name_alloc.get()};

    std::vector<Ort::Value> output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );

    float* float_array = output_tensors[0].GetTensorMutableData<float>();

    auto output_type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    size_t total_elements = output_type_info.GetElementCount();

    for (size_t i{0}; i < std::min(total_elements, size_t(5)); ++i) {
        std::cout << "Element [" << i << "]: " << float_array[i] << "\n";
    }

    return 0;
}
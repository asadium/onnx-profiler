#include <iostream>
#include <onnxruntime_cxx_api.h>

int main() {
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx-test");
        std::cout << "ONNX Runtime linked and initialized successfully.\n";
        return 0;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << '\n';
        return 1;
    }
}
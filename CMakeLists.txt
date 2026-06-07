# 1. Set the minimum version of CMake required to build this project
cmake_minimum_required(VERSION 3.10)

# 2. Name your project and specify the programming language (CXX means C++)
project(profiler LANGUAGES CXX)

# 3. Force the compiler to use a modern C++ standard (e.g., C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 4. Create an executable named 'hello_executable' from the source file 'main.cpp'
add_executable(profiler src/main.cpp)

find_package(onnxruntime CONFIG REQUIRED)
target_link_libraries(profiler PRIVATE onnxruntime::onnxruntime)
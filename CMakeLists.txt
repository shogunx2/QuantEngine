cmake_minimum_required(VERSION 3.16)
project(qe_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -mcpu=apple-m1")

find_package(pybind11 REQUIRED HINTS /opt/homebrew/opt/pybind11/share/cmake/pybind11)

pybind11_add_module(qe_core
    core/bindings.cpp
    core/src/indicators.cpp
    core/src/screener.cpp
)

target_include_directories(qe_core PRIVATE core core/include)

cmake_minimum_required(VERSION 3.16)

project(VolumeFog LANGUAGES CXX C)

add_subdirectory(ext/CGUtils)




file(
        GLOB_RECURSE
        SRCS
        "src/*.cpp"
)

add_executable(VolumeFog ${SRCS})

target_link_libraries(VolumeFog PRIVATE CGUtils)

target_compile_features(VolumeFog PRIVATE cxx_std_20)
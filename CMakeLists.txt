cmake_minimum_required(VERSION 3.10.0)

project(AMK-NN LANGUAGES CXX)

# Add subdirectories
add_subdirectory(amknn)
add_subdirectory(demo/image_upscale)
add_subdirectory(demo/conv_network)
add_subdirectory(demo/console_tests)

add_compile_options(-Wno-specific-warning-flag)
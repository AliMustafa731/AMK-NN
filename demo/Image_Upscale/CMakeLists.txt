
add_executable(image_upscale
               src/main.cpp 
		       src/loaders.cpp)

target_link_libraries(image_upscale PRIVATE amknn)

target_include_directories(image_upscale PRIVATE ${CMAKE_SOURCE_DIR}/amknn/src)

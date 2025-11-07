
add_executable(conv_network
               src/main.cpp 
		       src/loaders.cpp)

target_link_libraries(conv_network PRIVATE amknn)

target_include_directories(conv_network PRIVATE ${CMAKE_SOURCE_DIR}/amknn/src)

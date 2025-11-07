
add_executable(console_tests
               src/main.cpp)

target_link_libraries(console_tests PRIVATE amknn)

target_include_directories(console_tests PRIVATE ${CMAKE_SOURCE_DIR}/amknn/src)

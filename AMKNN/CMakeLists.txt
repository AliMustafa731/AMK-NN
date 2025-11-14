
file(GLOB_RECURSE AMKNN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(amknn STATIC ${AMKNN_SOURCES})

target_include_directories(amknn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

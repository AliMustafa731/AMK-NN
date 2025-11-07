
file(GLOB_RECURSE AMKNN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(amknn STATIC ${AMKNN_SOURCES})

target_link_libraries(amknn PUBLIC
        comctl32
        user32
        gdi32
        ole32
        uuid)

target_include_directories(amknn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

message(STATUS "Building amknn with sources: ${AMKNN_SOURCES}")

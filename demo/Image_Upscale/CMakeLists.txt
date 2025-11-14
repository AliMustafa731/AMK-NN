# --- wxWidgets setup ---
set(wxWidgets_ROOT_DIR "D:/Files/lib/wxWidgets/gcc/x86")
set(wxWidgets_LIB_DIR "${wxWidgets_ROOT_DIR}/lib/gcc1510_dll")

# You can choose one:
#   mswu  = Release Unicode
#   mswud = Debug Unicode
set(wxWidgets_CONFIGURATION mswu)

find_package(wxWidgets REQUIRED COMPONENTS core base gl)
include(${wxWidgets_USE_FILE})

add_executable(image_upscale WIN32
    src/main.cpp
    src/loaders.cpp
    src/components/MainWindow.cpp
    src/components/Image.cpp
    src/windows/resource.rc
)

target_include_directories(image_upscale PRIVATE
    ${CMAKE_SOURCE_DIR}/amknn/src
    ${wxWidgets_INCLUDE_DIRS}
)

target_link_libraries(image_upscale PRIVATE
    amknn
    ${wxWidgets_LIBRARIES}
)

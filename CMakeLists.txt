
cmake_minimum_required(VERSION 2.8)

project( normal_map CXX)

find_package( OpenCV REQUIRED )
message("OpenCV_INCLUDE_DIRS:${OpenCV_INCLUDE_DIRS}")
message("OpenCV_LIBS:${OpenCV_LIBS}")

file(GLOB SRC_FILES	"NormalMap/*.cpp")

include_directories(${CMAKE_CURRENT_BINARY_DIR} ${OpenCV_INCLUDE_DIRS})

add_executable( ${PROJECT_NAME}  ${SRC_FILES})

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS})

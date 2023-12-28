cmake_minimum_required(VERSION 3.0)

project(lon_mpc)

set(CMAKE_CXX_COMPILER "g++")
set (CMKAE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

find_package(Protobuf REQUIRED)
find_package(glog REQUIRED)
find_package(osqp REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
    ${PROTOBUF_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
    ${OSQP_INCLUDE_DIRS}
)

set(PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto)
file(GLOB proto_files proto/*.proto)
find_program(PROTOC protoc)

foreach(FIL ${proto_files})
    get_filename_component(FILE_NAME ${FIL} NAME)
    message("file_name: " ${FILE_NAME})
    set(PROTO_FILE ${FILE_NAME})
    execute_process(
        COMMAND ${PROTOC} -I=${PROTO_DIR} --cpp_out=${PROTO_DIR} ${PROTO_FILE}
    )
endforeach()

file(GLOB_RECURSE src_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
file(GLOB_RECURSE header_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

add_library(${PROJECT_NAME} SHARED ${src_files})
target_link_libraries(
    ${PROJECT_NAME} 
    ${GLOG_LIBRARIES} 
    ${PROTOBUF_LIBRARIES} 
    glog::glog 
    osqp::osqp
)

install(
    TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    FILES ${header_files} DESTINATION include
)
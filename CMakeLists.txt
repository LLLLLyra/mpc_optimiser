cmake_minimum_required(VERSION 3.5)

project(lon_mpc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

find_package(Protobuf REQUIRED)
find_package(glog REQUIRED)
find_package(osqp REQUIRED)
find_package(Eigen3 REQUIRED)

set(PROTOBUF_LINK_TARGET ${PROTOBUF_LIBRARIES})
if (TARGET protobuf::libprotobuf)
    set(PROTOBUF_LINK_TARGET protobuf::libprotobuf)
elseif (TARGET protobuf::protobuf)
    set(PROTOBUF_LINK_TARGET protobuf::protobuf)
endif()

set(PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto)
file(GLOB proto_files proto/*.proto)
if (TARGET protobuf::protoc)
    set(PROTOC $<TARGET_FILE:protobuf::protoc>)
else()
    find_program(PROTOC protoc)
endif()

foreach(FIL ${proto_files})
    get_filename_component(FILE_NAME ${FIL} NAME)
    message("file_name: " ${FILE_NAME})
    set(PROTO_FILE ${FILE_NAME})
    execute_process(
        COMMAND ${PROTOC} -I=${PROTO_DIR} --cpp_out=${PROTO_DIR} ${PROTO_FILE}
    )
endforeach()

file(GLOB_RECURSE src_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
file(GLOB_RECURSE header_files ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

add_library(${PROJECT_NAME} SHARED ${src_files})
target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROTOBUF_INCLUDE_DIRS}
        ${GLOG_INCLUDE_DIRS}
        ${OSQP_INCLUDE_DIRS}
)
target_link_libraries(
    ${PROJECT_NAME} 
    ${GLOG_LIBRARIES} 
    ${PROTOBUF_LINK_TARGET}
    Eigen3::Eigen
    glog::glog 
    osqp::osqp
)
if (BUILD_TESTING)
add_executable(
        ${PROJECT_NAME}_test
        test/test.cpp
        ${src_files}
)
target_include_directories(
    ${PROJECT_NAME}_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROTOBUF_INCLUDE_DIRS}
        ${GLOG_INCLUDE_DIRS}
        ${OSQP_INCLUDE_DIRS}
)
target_link_libraries(
    ${PROJECT_NAME}_test 
    ${GLOG_LIBRARIES} 
    ${PROTOBUF_LINK_TARGET}
    Eigen3::Eigen
    glog::glog 
    osqp::osqp
)
endif()

install(
    TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
)

install (FILES ${header_files} DESTINATION include)
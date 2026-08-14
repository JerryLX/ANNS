cmake_minimum_required(VERSION 3.20)
project(distributed_hnsw_threshold LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(hnsw_threshold src/main.cpp)
target_compile_options(hnsw_threshold PRIVATE -Wall -Wextra -Wpedantic -O3)

include(CTest)
if(BUILD_TESTING)
  add_test(NAME smoke COMMAND hnsw_threshold --n 500 --queries 5 --dim 8
           --clusters 8 --shards 2 --k 5 --ef-search 20 --latency-ns 1000)
  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME make_binary_fixture COMMAND ${Python3_EXECUTABLE}
             ${CMAKE_SOURCE_DIR}/tests/make_fixture.py ${CMAKE_BINARY_DIR}/fixture)
    set_tests_properties(make_binary_fixture PROPERTIES FIXTURES_SETUP binary_data)
    add_test(NAME load_fbin COMMAND hnsw_threshold
             --base ${CMAKE_BINARY_DIR}/fixture_base.fbin
             --query ${CMAKE_BINARY_DIR}/fixture_query.fbin
             --groundtruth ${CMAKE_BINARY_DIR}/fixture_gt.bin
             --shards 2 --k 5 --ef-search 10 --M 4 --ef-construction 20)
    set_tests_properties(load_fbin PROPERTIES FIXTURES_REQUIRED binary_data)
    add_test(NAME load_u8bin COMMAND hnsw_threshold
             --base ${CMAKE_BINARY_DIR}/fixture_base.u8bin
             --query ${CMAKE_BINARY_DIR}/fixture_query.u8bin
             --data-type auto --max-queries 2
             --shards 2 --k 5 --ef-search 10 --M 4 --ef-construction 20)
    set_tests_properties(load_u8bin PROPERTIES FIXTURES_REQUIRED binary_data)
  endif()
endif()

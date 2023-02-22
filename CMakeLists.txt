cmake_minimum_required(VERSION 3.4.1 FATAL_ERROR)
project(bvtest)

# set C++ build flags
set(CXX_STANDARD c++20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=gnu++20 ${GCC_WARNINGS} -ldl -fopenmp")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -ggdb")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

## If ninja is used, this enables colored output for the build output
option (FORCE_COLORED_OUTPUT "Always produce ANSI-colored output (GNU/Clang only)." TRUE)
if (${FORCE_COLORED_OUTPUT})
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        add_compile_options (-fdiagnostics-color=always)
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        add_compile_options (-fcolor-diagnostics)
    endif ()
endif ()
#add_subdirectory(subprojects/la_vector)
#add_subdirectory(subprojects/BitMagic)

add_executable(bvtest src/bvtest.cpp)

target_include_directories(bvtest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/subprojects/DYNAMIC/include)
target_include_directories(bvtest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/subprojects/DYNAMIC/deps/hopscotch_map/include)
target_include_directories(bvtest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/subprojects/BitMagic/src)
target_include_directories(bvtest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/subprojects/la_vector/include)
target_include_directories(bvtest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/subprojects/la_vector/lib/sdsl-lite/include)
target_include_directories(bvtest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/subprojects/progressbar/include)

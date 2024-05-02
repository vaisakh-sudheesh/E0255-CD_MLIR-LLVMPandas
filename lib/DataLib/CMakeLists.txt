cmake_minimum_required(VERSION 3.12)

cmake_policy(SET CMP0135 NEW) 

project(DataLib)

# SQLite3 library target
set(SQLLITE_SOURCES
    sqlite-amalgamation-3450200/sqlite3.c
    sqlite-amalgamation-3450200/shell.c
)

set(SQLLITE_COMPILER_WARNINGS
        -Wno-implicit-fallthrough
        -Wno-unused-variable
        -Wno-unused-parameter
        -Wno-unused-function
        -Wno-unused-but-set-variable
        -Wno-unused-value
        -Wno-unused-result
        -Wno-discarded-qualifiers
        -Wno-sign-compare
        -Wno-cast-qual
        -Wno-format-overflow
        -Wno-incompatible-pointer-types-discards-qualifiers
        -Wno-null-pointer-subtraction
)


add_library(SQLite STATIC ${SQLLITE_SOURCES})
target_include_directories(SQLite PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/sqlite-amalgamation-3450200)
target_compile_options(SQLite PRIVATE ${SQLLITE_COMPILER_WARNINGS} -fPIC -g)

# Add the library target for CXX Version of DataLib
add_library(DataLibCXX SHARED src/DataFrame.cpp)
target_include_directories(DataLibCXX PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/CXX)
target_link_libraries(DataLibCXX SQLite)
target_compile_options(DataLibCXX PRIVATE -Wno-sign-compare)

# Add the tests for CXX Version of DataLib
add_executable(TestsCXX tests/testsCXX.cpp)
target_include_directories(TestsCXX PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(TestsCXX gtest_main DataLibCXX)

# Add the library target for C Version of DataLib
add_library(DataLibC SHARED src/DataFrame.c )
target_include_directories(DataLibC PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/C)
target_link_libraries(DataLibC SQLite)
target_compile_options(DataLibC PRIVATE -Wno-format-overflow -Wno-cast-qual)

# Add the tests for C Version of DataLib
add_executable(TestsC tests/testsC.cpp)
target_include_directories(TestsC PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(TestsC gtest_main DataLibC)

# Fetch GoogleTest target for the tests
include(FetchContent) 
FetchContent_Declare( googletest URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip) 
FetchContent_MakeAvailable(googletest)


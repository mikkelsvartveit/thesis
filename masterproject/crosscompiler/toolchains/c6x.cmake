# C6x CMake Toolchain File
# This file configures CMake to use the C6x cross-compiler correctly

# Specify the target system
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR c6x)

# Specify the cross compiler
set(CMAKE_C_COMPILER /cross-c6x/bin/c6x-unknown-uclinux-gcc)
set(CMAKE_CXX_COMPILER /cross-c6x/bin/c6x-unknown-uclinux-g++)
set(CMAKE_ASM_COMPILER /cross-c6x/bin/c6x-unknown-uclinux-gcc)

# Where to look for the target environment
set(CMAKE_SYSROOT /cross-c6x)
set(CMAKE_FIND_ROOT_PATH /cross-c6x)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Based on the error message and file inspection, set appropriate flags
# Force big-endian mode to match the library format
add_compile_options(-mbig-endian)
add_link_options(-mbig-endian)

# Library path settings
# Make sure CMake looks for libraries in the correct location
set(CMAKE_C_IMPLICIT_LINK_DIRECTORIES 
    /cross-c6x/lib
    /cross-c6x/c6x-unknown-uclinux/lib
    /cross-c6x/c6x-unknown-uclinux/lib/be
    /cross-c6x/lib/gcc/c6x-unknown-uclinux/14.2.0
)

# Tell the linker where to find the libraries explicitly
add_link_options(
    "-Wl,-rpath-link,/cross-c6x/lib"
    "-Wl,-rpath-link,/cross-c6x/c6x-unknown-uclinux/lib"
)

# Prevent the linker from using host system libraries
set(CMAKE_C_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_CXX_IMPLICIT_LINK_LIBRARIES "")

# Set the compiler runtime search directories
set(CMAKE_INSTALL_RPATH "/cross-c6x/lib")

# Disable features that might not be available in the C6X target environment
# Disable pcre2grep since it requires fork() which is unavailable
set(PCRE2_BUILD_PCRE2GREP OFF CACHE BOOL "Build pcre2grep" FORCE)

# Disable example/test programs in libjpeg-turbo
set(WITH_TURBOJPEG OFF CACHE BOOL "Include the TurboJPEG API library and associated test programs" FORCE)
set(ENABLE_STATIC ON CACHE BOOL "Build static libraries" FORCE)
set(ENABLE_SHARED OFF CACHE BOOL "Build shared libraries" FORCE)

# Force disable all libjpeg-turbo executables
# This is a more direct approach since WITH_TURBOJPEG=OFF doesn't disable all programs
set(PCRE2_BUILD_TESTS OFF CACHE BOOL "Build the tests" FORCE)
set(ENABLE_SHARED OFF CACHE BOOL "Enable shared library" FORCE)


# Prevent CMake from adding implicit link flags that might interfere
set(CMAKE_C_LINK_EXECUTABLE "<CMAKE_C_COMPILER> <FLAGS> <CMAKE_C_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")

# Avoid linking issues with specific libraries
# If you're having issues with specific libraries, you might need to explicitly exclude them
# or specify static linking for certain components
# Uncomment the following line if needed:
# add_link_options(-static-libgcc)

# Set appropriate flags for C6x architecture
# You might need to specify the exact C6x processor variant if needed
# For example, for C674x:
# add_compile_options(-march=c674x)

# Disable the compiler's attempts to use system libraries
set(CMAKE_C_STANDARD_LIBRARIES "" CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD_LIBRARIES "" CACHE STRING "" FORCE)

# Configure how CMake determines if a program can run
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# Display cross-compiling status for debugging purposes
message(STATUS "Cross-compiling for C6x with toolchain: ${CMAKE_C_COMPILER}")
message(STATUS "Sysroot: ${CMAKE_SYSROOT}")
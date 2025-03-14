# M32C toolchain file for CMake

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR m32r)

# Specify cross compilers and tools
set(CMAKE_C_COMPILER m32r-unknown-elf-gcc)
set(CMAKE_CXX_COMPILER false)
set(CMAKE_ASM_COMPILER m32r-unknown-elf-gcc)
set(CMAKE_AR m32r-unknown-elf-ar)
set(CMAKE_RANLIB m32r-unknown-elf-ranlib)
set(CMAKE_STRIP m32r-unknown-elf-strip)

# Where is the target environment located
set(CMAKE_FIND_ROOT_PATH /cross-m32r)

# Search for programs only in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# Search for libraries and headers only in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Cache flags - adjust these based on specific M32C requirements
set(CMAKE_C_FLAGS_INIT "")
set(CMAKE_CXX_FLAGS_INIT "")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined -Wl,--unresolved-symbols=ignore-all")

set(XZ_THREADS no)
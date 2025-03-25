# M32C toolchain file for CMake

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR epiphany)

# Specify cross compilers and tools
set(CMAKE_C_COMPILER epiphany-unknown-elf-gcc)
set(CMAKE_CXX_COMPILER false)
set(CMAKE_ASM_COMPILER epiphany-unknown-elf-gcc)
set(CMAKE_AR epiphany-unknown-elf-ar)
set(CMAKE_RANLIB epiphany-unknown-elf-ranlib)
set(CMAKE_STRIP epiphany-unknown-elf-strip)

# Where is the target environment located
set(CMAKE_FIND_ROOT_PATH /cross-epiphany)

# Search for programs only in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# Search for libraries and headers only in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)


set(CMAKE_C_FLAGS_INIT "-O1")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ")
set(CMAKE_CXX_FLAGS_INIT "")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined -Wl,--unresolved-symbols=ignore-all")

set(XZ_THREADS no)
set(WEBP_USE_THREAD OFF)
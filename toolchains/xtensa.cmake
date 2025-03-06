# Xtensa (ESP32) toolchain file for CMake

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR xtensa)

# Specify cross compilers and tools
set(CMAKE_C_COMPILER xtensa-esp32-elf-gcc)
set(CMAKE_CXX_COMPILER xtensa-esp32-elf-g++)
set(CMAKE_ASM_COMPILER xtensa-esp32-elf-gcc)
set(CMAKE_AR xtensa-esp32-elf-ar)
set(CMAKE_RANLIB xtensa-esp32-elf-ranlib)
set(CMAKE_STRIP xtensa-esp32-elf-strip)

# Where is the target environment located
set(CMAKE_FIND_ROOT_PATH /opt/toolchains/xtensa-esp32-elf)

# Search for programs only in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# Search for libraries and headers only in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Cache flags
set(CMAKE_C_FLAGS_INIT "-mlongcalls")
set(CMAKE_CXX_FLAGS_INIT "-mlongcalls")

# Disable some features that might cause problems with Xtensa
set(CMAKE_C_COMPILER_WORKS 1)
set(CMAKE_CXX_COMPILER_WORKS 1)

set(ENABLE_SHARED FALSE)
set(ENABLE_TESTING FALSE)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lm")
set(ENABLE_SHARED FALSE)
set(ENABLE_TESTING FALSE) 
set(WITH_TURBOJPEG FALSE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lm -mlongcalls") 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lm -mlongcalls") 

set(LIBXML2_WITH_PYTHON OFF)
set(LIBXML2_WITH_THREADS OFF)
set(LIBXML2_WITH_LZMA OFF)
set(LIBXML2_WITH_ZLIB OFF)
set(LIBXML2_WITH_MODULES OFF)


# stubs:
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/dup_stub.c
"#include <unistd.h>
#include <errno.h>

int dup(int oldfd) {
    errno = ENOSYS;  // Not implemented
    return -1;
}
")
set(LIBXML2_SRCS ${LIBXML2_SRCS} dup_stub.c)
# Xtensa (ESP32) toolchain file for CMake

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR m88k)

# Specify cross compilers and tools
set(CMAKE_C_COMPILER m88k-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER m88k-linux-gnu-g++)
set(CMAKE_ASM_COMPILER m88k-linux-gnu-gcc)
set(CMAKE_AR m88k-linux-gnu-ar)
set(CMAKE_RANLIB m88k-linux-gnu-ranlib)
set(CMAKE_STRIP m88k-linux-gnu-strip)

# Where is the target environment located
set(CMAKE_FIND_ROOT_PATH /usr/m88k-linux-gnu)

# Search for programs only in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# Search for libraries and headers only in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Cache flags
set(CMAKE_C_FLAGS_INIT "-mbroken-saverestore -midentify-revision")
set(CMAKE_CXX_FLAGS_INIT "-mbroken-saverestore -midentify-revision")

# Disable some features that might cause problems with m88k
# set(M_LIBRARY m)
# set(CMAKE_C_COMPILER_WORKS 1)
# set(CMAKE_CXX_COMPILER_WORKS 1)

# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined -Wl,--unresolved-symbols=ignore-all")

# set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -lm")

# set(ENABLE_SHARED FALSE)
# set(ENABLE_TESTING FALSE) 
# set(WITH_TURBOJPEG FALSE)
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mlongcalls -I/workspace/patches/mock_headers/") 
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mlongcalls -I/workspace/patches/mock_headers/") 

# set(LIBXML2_WITH_PYTHON OFF)
# set(LIBXML2_WITH_THREADS OFF)
# set(LIBXML2_WITH_LZMA OFF)
# set(LIBXML2_WITH_ZLIB OFF)
# set(LIBXML2_WITH_MODULES OFF)

# #HARFBUZZ
# set(HAVE_SYS_MMAN_H OFF)
# set(HAVE_MMAP OFF)
# set(HAVE_MPROTECT OFF)

# # PCRE2
# set(PCRE2_BUILD_TESTS OFF)

# # libwebp
# set(WEBP_USE_THREAD OFF)

# set(XZ_THREADS no)
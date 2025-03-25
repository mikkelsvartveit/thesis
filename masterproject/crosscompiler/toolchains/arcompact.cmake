# ARCompact toolchain file for CMake

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arc)

set(CMAKE_PLATFORM_NO_SHARED TRUE)

# Specify cross compilers and tools
set(CMAKE_C_COMPILER arc-elf32-gcc)
set(CMAKE_CXX_COMPILER arc-elf32-g++)
set(CMAKE_ASM_COMPILER arc-elf32-gcc)
set(CMAKE_AR arc-elf32-ar)
set(CMAKE_RANLIB arc-elf32-ranlib)
set(CMAKE_STRIP arc-elf32-strip)

# Where is the target environment located
set(CMAKE_FIND_ROOT_PATH /opt/arc/arc_gnu_2021.03_prebuilt_elf32_le_linux_install/arc-elf32)

# Search for programs only in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# Search for libraries and headers only in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Cache flags - using arc700 which is a common ARCompact implementation
set(CMAKE_C_FLAGS_INIT "-mcpu=arc700")
set(CMAKE_CXX_FLAGS_INIT "-mcpu=arc700")

# Disable specific features in CMake that might cause issues
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)


set(CMAKE_C_COMPILER_WORKS 1)
set(CMAKE_CXX_COMPILER_WORKS 1)

set(ENABLE_TESTING FALSE) 
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined -Wl,--unresolved-symbols=ignore-all")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -lm")

# LIBXML2
set(LIBXML2_WITH_PYTHON OFF)
set(LIBXML2_WITH_THREADS OFF)
set(LIBXML2_WITH_LZMA OFF)
set(LIBXML2_WITH_ZLIB OFF)
set(LIBXML2_WITH_MODULES OFF)

# LIBTURBOJPEG
set(WITH_TURBOJPEG FALSE)

#HARFBUZZ
set(HAVE_SYS_MMAN_H OFF)
set(HAVE_MMAP OFF)
set(HAVE_MPROTECT OFF)

# PCRE2
set(PCRE2_BUILD_TESTS OFF)

# libwebp
set(WEBP_USE_THREAD OFF)

set(XZ_THREADS no)

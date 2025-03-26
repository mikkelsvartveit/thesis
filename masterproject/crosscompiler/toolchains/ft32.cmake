set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR ft32)

# Specify the cross-compiler
set(CMAKE_C_COMPILER /cross-ft32/bin/ft32-unknown-elf-gcc)
set(CMAKE_CXX_COMPILER /cross-ft32/bin/ft32-unknown-elf-g++)
set(CMAKE_LINKER /cross-ft32/bin/ft32-unknown-elf-ld)

# Where to look for the target environment
set(CMAKE_FIND_ROOT_PATH /cross-ft32)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Disable features that might cause problems
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)

# Add compiler flags to fix section overlap and type inconsistencies
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-exceptions -fno-asynchronous-unwind-tables -fdata-sections -ffunction-sections -Wno-incompatible-pointer-types")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -fno-asynchronous-unwind-tables -fdata-sections -ffunction-sections -Wno-incompatible-pointer-types")

# Add additional defines for architecture-specific type handling
add_definitions(-DHAVE_CONFIG_H -DPCRE2_STATIC)

# Add linker flags to handle section placement
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")

# Disable threading for libraries that might use it
set(XZ_THREADS no CACHE BOOL "Disable threads in XZ" FORCE)
set(WEBP_USE_THREAD OFF CACHE BOOL "Disable threads in WebP" FORCE)

# Library specific settings
set(ZLIB_DISABLE_TESTS ON CACHE BOOL "Disable zlib tests" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)

# PCRE2 specific settings
set(PCRE2_BUILD_PCRE2_8 ON CACHE BOOL "Build 8 bit PCRE2 library" FORCE)
set(PCRE2_BUILD_PCRE2_16 OFF CACHE BOOL "Disable 16 bit PCRE2 library" FORCE)
set(PCRE2_BUILD_PCRE2_32 OFF CACHE BOOL "Disable 32 bit PCRE2 library" FORCE)
set(PCRE2_SUPPORT_JIT OFF CACHE BOOL "Disable JIT support" FORCE)
set(PCRE2_SUPPORT_UNICODE OFF CACHE BOOL "Disable Unicode support for embedded targets" FORCE)
set(PCRE2_HEAP_MATCH_RECURSE ON CACHE BOOL "Use heap for match recursion" FORCE)

# You might need these for 32-bit targets with limited memory
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Os")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Os")
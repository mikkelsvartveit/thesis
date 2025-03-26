# V850 Toolchain File - Comprehensive configuration

# Basic system information
SET(CMAKE_SYSTEM_NAME Generic)
SET(CMAKE_SYSTEM_PROCESSOR v850)

# Define the cross compiler locations
SET(CMAKE_C_COMPILER /cross-v850/bin/v850e-uclinux-uclibc-gcc)
SET(CMAKE_CXX_COMPILER /cross-v850/bin/v850e-uclinux-uclibc-g++)
SET(CMAKE_ASM_COMPILER /cross-v850/bin/v850e-uclinux-uclibc-gcc)

# Critical: Skip compiler tests that require building executables
SET(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# Define library paths
SET(CMAKE_FIND_ROOT_PATH /cross-v850)

# Search programs, libraries and includes in the target environment
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Disable position independent code for embedded target
SET(CMAKE_POSITION_INDEPENDENT_CODE OFF)

# Disable threading for WebP
SET(WEBP_USE_THREAD OFF)
SET(XZ_USE_PTHREAD FALSE)

# Disable building executables/tools
SET(WEBP_BUILD_CWEBP OFF)
SET(WEBP_BUILD_DWEBP OFF)
SET(WEBP_BUILD_GIF2WEBP OFF)
SET(WEBP_BUILD_IMG2WEBP OFF)
SET(WEBP_BUILD_WEBPINFO OFF)
SET(WEBP_BUILD_WEBPMUX OFF)
SET(WEBP_BUILD_EXTRAS OFF)
SET(WEBP_BUILD_ANIM_UTILS OFF)

# Compiler and linker flags suitable for embedded targets
# Changed -mcpu=v850e to -mv850e as per compiler suggestion
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mv850e -mno-app-regs")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mv850e -mno-app-regs")
SET(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")

# Force static library building only
SET(BUILD_SHARED_LIBS OFF)
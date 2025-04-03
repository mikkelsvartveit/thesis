set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR rx)

set(CMAKE_C_COMPILER rx-unknown-elf-gcc)
set(CMAKE_CXX_COMPILER rx-unknown-elf-g++)
set(CMAKE_AR rx-unknown-elf-ar)
set(CMAKE_RANLIB rx-unknown-elf-ranlib)
set(CMAKE_STRIP rx-unknown-elf-strip)

# add_compile_options(-mbig-endian-data)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined -Wl,--unresolved-symbols=ignore-all")

set(CMAKE_FIND_ROOT_PATH "/cross-rx")
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
set(XZ_THREADS no)
set(WEBP_USE_THREAD OFF)
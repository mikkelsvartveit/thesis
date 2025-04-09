set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR mcore)

set(CMAKE_C_COMPILER mcore-unknown-elf-gcc)
set(CMAKE_CXX_COMPILER mcore-unknown-elf-g++)
set(CMAKE_AR mcore-unknown-elf-ar)
set(CMAKE_RANLIB mcore-unknown-elf-ranlib)
set(CMAKE_STRIP mcore-unknown-elf-strip)

# add_compile_options(-mbig-endian)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined -Wl,--unresolved-symbols=ignore-all")

set(CMAKE_FIND_ROOT_PATH "/cross-mcore")
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
set(XZ_THREADS no)
set(WEBP_USE_THREAD OFF)
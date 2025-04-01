set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR moxie)

set(CMAKE_C_COMPILER moxie-unknown-elf-gcc)
set(CMAKE_CXX_COMPILER moxie-unknown-elf-g++)
set(CMAKE_AR moxie-unknown-elf-ar)
set(CMAKE_RANLIB moxie-unknown-elf-ranlib)
set(CMAKE_STRIP moxie-unknown-elf-strip)

#add_compile_options(-mel -Wa,-EL)
#add_link_options(-mel -Wa,-EL)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined -Wl,--unresolved-symbols=ignore-all")

set(CMAKE_FIND_ROOT_PATH "/cross-moxie")
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
set(XZ_THREADS no)
set(WEBP_USE_THREAD OFF)
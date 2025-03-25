set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR h8300)

set(CMAKE_C_COMPILER /cross-h8300/bin/h8300-unknown-linux-uclibc-gcc)
set(CMAKE_CXX_COMPILER /cross-h8300/bin/h8300-unknown-linux-uclibc-g++)

set(CMAKE_SYSROOT /cross-h8300)
set(CMAKE_FIND_ROOT_PATH /cross-h8300)

# Explicitly set library paths
set(CMAKE_C_LIBRARY_ARCHITECTURE h8300-unknown-linux-uclibc)
set(CMAKE_LIBRARY_PATH 
    /cross-h8300/lib/gcc/h8300-unknown-linux-uclibc/14.2.0
    /cross-h8300/h8300-unknown-linux-uclibc/lib
    /cross-h8300/usr/lib
)

# Don't use flags meant for the host
set(CMAKE_C_FLAGS "" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "" CACHE STRING "" FORCE)

# Essential search behavior
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Disable dynamic linking flags
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")

# Disable position independent code for h8300
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)

# Explicitly set the system name to avoid -rdynamic

# Specific h8300 flags - choose the appropriate variant
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mh" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mh" CACHE STRING "" FORCE)
# ATLAS/CMakeLists.txt — Extension root
cmake_minimum_required(VERSION 3.16.3)

project(ATLAS)

# ---- Extension meta-data (shown in Slicer Extension Manager) ----
set(EXTENSION_HOMEPAGE         "https://github.com/agporto/ATLAS")
set(EXTENSION_CATEGORY         "ATLAS")     # Pick something appropriate
set(EXTENSION_CONTRIBUTORS     "Arthur Porto")
set(EXTENSION_DESCRIPTION      "Tools for template alignment, SSM-guided CPD, etc.")
set(EXTENSION_ICONURL          "")              # Optional
set(EXTENSION_SCREENSHOTURLS   "")              # Optional
set(EXTENSION_DEPENDS          "")              # Space-separated Slicer extensions your extension depends on

find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

# Add your modules
add_subdirectory(PREDICT)
add_subdirectory(BUILDER)
add_subdirectory(DATABASE)



# Packaging, testing
include(${Slicer_EXTENSION_CPACK})

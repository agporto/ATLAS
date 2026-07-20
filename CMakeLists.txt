cmake_minimum_required(VERSION 3.5)

project(CoherentAtlas)

# Extension metadata displayed in the Slicer Extensions Manager.
set(EXTENSION_HOMEPAGE "https://github.com/agporto/SlicerATLAS")
set(EXTENSION_CATEGORY "Registration")
set(EXTENSION_CONTRIBUTORS "Arthur Porto (Florida Museum of Natural History, University of Florida)")
set(EXTENSION_DESCRIPTION "CoherentAtlas provides workflows for constructing population atlases, managing reusable statistical shape model libraries, transferring landmarks to new surface models, and generating corresponding surface fragments. It combines rigid registration, shape-model-guided deformable registration, and optional surface projection.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/agporto/SlicerATLAS/main/PREDICT/Resources/Icons/PREDICT.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/agporto/SlicerATLAS/main/tutorial/images/20.png")
set(EXTENSION_DEPENDS "")

find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

add_subdirectory(CoherentAtlasConstruction)
add_subdirectory(CoherentAtlasLibrary)
add_subdirectory(CoherentAtlasTransfer)
add_subdirectory(CoherentAtlasFragmentation)

include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})

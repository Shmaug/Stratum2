cmake_minimum_required(VERSION 3.20)

project(tinyvkpt)

# Find dependencies

find_package(Vulkan REQUIRED)
find_package(assimp CONFIG)
find_package(OpenXR CONFIG)
find_package(OpenVDB CONFIG)


# download slang release rom github if it doesnt already exist
set(SLANG_ROOT "${CMAKE_CURRENT_LIST_DIR}/extern/slang")
if (NOT EXISTS ${SLANG_ROOT})
	message(STATUS "Downloading slang")
	set(SLANG_URL https://github.com/shader-slang/slang/releases/download/v0.24.35/slang-0.24.35)
	if (WIN32)
		file(DOWNLOAD "${SLANG_URL}-win64.zip" "${SLANG_ROOT}.zip")
		message(STATUS "Extracting slang")
		file(ARCHIVE_EXTRACT INPUT "${SLANG_ROOT}.zip" DESTINATION "${SLANG_ROOT}")
		file(REMOVE "${SLANG_ROOT}.zip")
	else()
		file(DOWNLOAD  "${SLANG_URL}-linux-x86_64.zip" "${SLANG_ROOT}.zip")
		message(STATUS "Extracting slang")
		file(ARCHIVE_EXTRACT INPUT "${SLANG_ROOT}.zip" DESTINATION "${SLANG_ROOT}")
		file(REMOVE "${SLANG_ROOT}.zip")
	endif()
endif()

# store path to slang compiler
set(SLANGC "")
if (WIN32)
	set(SLANGC "${SLANG_ROOT}/bin/windows-x64/release/slangc.exe")
else()
	set(SLANGC "${SLANG_ROOT}/bin/linux-x64/release/slangc")
endif()

include_directories(extern)
include_directories(src)

set(TARGET_NAME tinyvkpt)

set(CMAKE_CXX_STANDARD 23)
set(CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Add sources
file(GLOB_RECURSE GLOBBED_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/**.cpp")
list(APPEND STRATUM_SOURCE ${GLOBBED_SRCS})
list(APPEND STRATUM_SOURCE
	"${CMAKE_CURRENT_SOURCE_DIR}/extern/miniz.c"
	"${CMAKE_CURRENT_SOURCE_DIR}/extern/pugixml.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/extern/imgui/imgui_draw.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/extern/imgui/imgui_widgets.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/extern/imgui/imgui_tables.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/extern/imgui/imgui_demo.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/extern/imgui/imgui_impl_glfw.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/extern/imgui/imgui_impl_vulkan.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/extern/imgui/imgui.cpp"
)

add_executable(${TARGET_NAME} ${STRATUM_SOURCE} src/main.cpp)
set_target_properties(${TARGET_NAME} PROPERTIES ENABLE_EXPORTS TRUE)
set_target_properties(${TARGET_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_definitions(${TARGET_NAME} PUBLIC _USE_MATH_DEFINES IMGUI_DEFINE_MATH_OPERATORS)

option(ENABLE_DEBUG_LAYERS "Enable debug layers" TRUE)
if (${ENABLE_DEBUG_LAYERS})
    target_compile_definitions(${TARGET_NAME} PUBLIC ENABLE_DEBUG_LAYERS)
endif()

configure_file("${CMAKE_CURRENT_LIST_DIR}/extern/imgui/misc/fonts/DroidSans.ttf" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DroidSans.ttf" COPYONLY)

# Link dependencies

include_directories(extern/glfw/deps)
add_subdirectory(extern/glfw)

target_include_directories(${TARGET_NAME} PUBLIC extern/glfw/include)
target_link_libraries(${TARGET_NAME} PUBLIC Vulkan::Vulkan glfw)

if(UNIX)
    target_link_libraries(${TARGET_NAME} PUBLIC pthread ${CMAKE_DL_LIBS} xcb xcb-keysyms)
    target_compile_definitions(${TARGET_NAME} PUBLIC VK_USE_PLATFORM_XCB_KHR _GLFW_X11)

	# Link slang
    target_link_libraries(${TARGET_NAME} PUBLIC "${SLANG_ROOT}/bin/linux-x64/release/libslang.so")
elseif(WIN32)
    target_compile_definitions(${TARGET_NAME} PUBLIC VK_USE_PLATFORM_WIN32_KHR _GLFW_WIN32 WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS NOMINMAX)
    target_compile_options(${TARGET_NAME} PUBLIC /bigobj /MP)

	# Link slang
    target_compile_definitions(${TARGET_NAME} PUBLIC SLANG_STATIC)
    target_link_libraries(${TARGET_NAME} PUBLIC "${SLANG_ROOT}/bin/windows-x64/release/slang.lib")

	# Copy slang DLLs to output directory
	configure_file("${SLANG_ROOT}/bin/windows-x64/release/slang.dll" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/slang.dll" COPYONLY)
	configure_file("${SLANG_ROOT}/bin/windows-x64/release/slang-glslang.dll" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/slang-glslang.dll" COPYONLY)
	configure_file("${SLANG_ROOT}/bin/windows-x64/release/gfx.dll" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gfx.dll" COPYONLY)
endif()

if (assimp_FOUND)
	if (WIN32)
		target_link_libraries(${TARGET_NAME} PRIVATE assimp::assimp)
	else()
		target_link_libraries(${TARGET_NAME} PRIVATE assimp)
	endif()
    target_compile_definitions(${TARGET_NAME} PUBLIC ENABLE_ASSIMP)
    message(STATUS "Assimp enabled")
endif()

if (OpenXR_FOUND)
    target_link_libraries(${TARGET_NAME} PUBLIC OpenXR::openxr_loader)
    target_compile_definitions(${TARGET_NAME} PUBLIC ENABLE_OPENXR XR_USE_GRAPHICS_API_VULKAN)
    message(STATUS "OpenXR enabled")
endif()

if (OpenVDB_FOUND)
    target_link_libraries(${TARGET_NAME} PRIVATE OpenVDB::openvdb)
    target_compile_definitions(${TARGET_NAME} PUBLIC ENABLE_OPENVDB)
    message(STATUS "OpenVDB enabled")
endif()

# Shaders

include(CompileShader.cmake)

file(GLOB_RECURSE STRATUM_SHADERS "**.[gh]lsl")
stm_add_shaders(SOURCES ${STRATUM_SHADERS} DST_FOLDER "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Shaders" INCLUDES "${CMAKE_CURRENT_LIST_DIR}/Shaders" "${CMAKE_CURRENT_LIST_DIR}/extern" DEPENDS tinyvkpt)
file(GLOB_RECURSE STRATUM_SHADERS "**.frag")
stm_add_shaders(SOURCES ${STRATUM_SHADERS} DST_FOLDER "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Shaders" INCLUDES "${CMAKE_CURRENT_LIST_DIR}/Shaders" "${CMAKE_CURRENT_LIST_DIR}/extern" DEPENDS tinyvkpt)
file(GLOB_RECURSE STRATUM_SHADERS "**.vert")
stm_add_shaders(SOURCES ${STRATUM_SHADERS} DST_FOLDER "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Shaders" INCLUDES "${CMAKE_CURRENT_LIST_DIR}/Shaders" "${CMAKE_CURRENT_LIST_DIR}/extern" DEPENDS tinyvkpt)
file(GLOB_RECURSE STRATUM_SHADERS "**.slang")
stm_add_shaders(SOURCES ${STRATUM_SHADERS} DST_FOLDER "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Shaders" INCLUDES "${CMAKE_CURRENT_LIST_DIR}/Shaders" "${CMAKE_CURRENT_LIST_DIR}/extern" DEPENDS tinyvkpt)

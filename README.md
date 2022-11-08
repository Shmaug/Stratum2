Lightweight vulkan wrapper and scene graph, includes required dependencies. Can load environment maps (\*.hdr, \*.exr), GLTF scenes (\*.glb, \*.gltf), as well a NVDB or Mitsuba volumes (\*.nvdb, \*.vol)

Required dependencies are in 'extern'. Optional packages (searched via find_package in CMake) are:
- 'assimp' for loading \*.fbx scenes
- 'OpenVDB' for loading \*.vdb volumes
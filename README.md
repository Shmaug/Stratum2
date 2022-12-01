Lightweight vulkan wrapper and scene graph, includes required dependencies. Can load environment maps (\*.hdr, \*.exr), GLTF scenes (\*.glb, \*.gltf), as well a NVDB or Mitsuba volumes (\*.nvdb, \*.vol)

# Dependencies
Required dependencies are in 'extern'. slang is downloaded automatically. Optional dependences (searched via find_package in CMake) are:
- 'assimp' for loading \*.fbx scenes
- 'OpenVDB' for loading \*.vdb volumes

# Command-line arguments
Stratum stores commandline arguments in the `Instance` class and can be queried with the `findArgument` and `findArguments` functions. Many core components use this

# Core arguments
* --instanceExtension=`string`
* --deviceExtension=`string`
* --validationLayer=`string`
* --debugMessenger
* --noPipelineCache
* --font=`path,float`
* --shaderKernelPath=`path`
* --shaderInclude=`path`
## Window arguments
* --width=`int`
* --height=`int`
* --presentMode=`string`
## Scene arguments
* --scene=`path`

## Path tracer arguments
* --minBounces=`int`
* --maxBounces=`int`
* --maxDiffuseBounces=`int`
* --exposure=`float`

Currently, Stratum should be run with the following arguments:
* --shaderKernelPath=${Stratum2_Dir}/src/Shaders/kernels
* --shaderInclude=${Stratum2_Dir}/extern
* --shaderInclude=${Stratum2_Dir}/src/Shaders
* --deviceExtension=VK_KHR_ray_query
* --debugMessenger
* --font=DroidSans.ttf,16
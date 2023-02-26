Lightweight vulkan wrapper and scene graph. By default, supports loading the following files:
* Environment maps (\*.hdr, \*.exr)
* GLTF scenes (\*.glb, \*.gltf)
* NVDB or Mitsuba volumes (\*.nvdb, \*.vol)

# Dependencies
Required dependencies are in 'extern'. slang is downloaded automatically. Optional dependences (searched via find_package in CMake) are:
- 'assimp' for loading \*.fbx scenes
- 'OpenVDB' for loading \*.vdb volumes

# Command-line arguments
Stratum stores commandline arguments in the `Instance` class and can be queried with the `findArgument` and `findArguments` functions.

## Core arguments
* --instanceExtension=`string`
* --deviceExtension=`string`
* --validationLayer=`string`
* --debugMessenger
* --noPipelineCache
* --shaderKernelPath=`path`
* --shaderInclude=`path`
* --font=`path,float`
## Window arguments
* --width=`int`
* --height=`int`
* --presentMode=`string`
## Scene arguments
* --scene=`path`
* --cameraPosition=`x,y,z`
* --cameraOrientation=`x,y`

## Path tracer arguments
* --minPathLength=`int`
* --maxPathLength=`int`
* --exposure=`float`
* --noReprojection
* --noNormalCheck
* --noDepthCheck

# Required arguments
Currently, Stratum requires the following arguments in order to function:
* --shaderKernelPath=${Stratum2_Dir}/src/Shaders/kernels
* --shaderInclude=${Stratum2_Dir}/extern
* --shaderInclude=${Stratum2_Dir}/src/Shaders
* --debugMessenger

Optionally, the following arguments are recommended:
* --font=DroidSans.ttf,16
* --width=1920
* --height=1080
* --presentMode=Immediate
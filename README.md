nau
===

nau 3D engine: OpenGL + Optix 7 (WIP) + Lua + ImGui + Anttweakbar + Bullet + PhysX

Documentation at http://nau3d.di.uminho.pt/

# credits

Besides myself, many people have contributed to this project:

* Bruno Oliveira - software architecture, programming
* Pedro Ângelo - programming
* Marta Pereira - initial implementation of the event system manager
* Leander Beernaert - initial Linux version, initial CMake building system
* André Lui - initial implementation of the debug features available in nau
* Jaime Campos and João Meira - initial Bullet integration
* David Leal - Bullet and PhysX plugins
* Bruno Martins - developing a plugin for Intel's embree (not ready yet)

ImGuiFileDialog from https://github.com/aiekick/ImGuiFileDialog

# building

To build nau, and GLUT and GLFW (with ImGui) demo, 
the project is self-contained for Windows. For Linux install opengl, and devil
* Devil 
	* sudo apt-get install libdevil-dev
* Freeglut (as a short cut to installing opengl)
	* sudo apt-get install freeglut3-dev
	
Note: IF fail to compile freeglut try
* cd /usr/include/X11/extensions
* sudo ln –s XI.h XInput.h

To build nau with nvidia's optix 7 support (optional) both cuda and optix are required.
* In the cmake project check the option "NAU_BUILD_WITH_OPTIX"
* Set the variable OPTIX_DIR to optix's installation directory
* CUDA is usually found by cmake and doesn't need any extra steps.
* If either CUDA or Optix are not found the process goes on without Optix support.		

		
To built bullet plugin, download the latest release (Windows only).

* Four libs are required, in both debug and release mode: BulletDynamics, BulletCollision, BulletSoftBody and LinearMath. 
*  Prior to building the libs do as follows for each lib:
	* Go to properties (right button click on libs name)
	* Select C/C++ -> Code Generation
	* For the Runtime Library option elect the respective DLL version (either Multi-threaded Debug DLL or  Multi-threaded DLL)
	* Build the named libraries.	
* In CMake set variables in the GUI: BULLET_INCLUDE_DIR (the source files dir of the bullet install) and BULLET_ROOT (bullet build dir)

To build PhysX plugin, in CMake gui:

* set PHYSX_INCLUDE_DIR to the path to PhysX include dir
* set PHYSX_LIBRARIES_DIR to the path to PhysX lib dir


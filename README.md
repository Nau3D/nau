Nau
===

Nau 3D engine: OpenGL + Optix 7 (WIP) + Lua + ImGui 

Nau is an API to speed up the creation of 3D shader pipelines. It integrates both rasterization and ray tracing in a single package, providing all the tools to create complex multipass graphic pipelines with shaders written in GLSL, and kernels written in CUDA for Optix7. 

It supports Lua scrippting to provide an easy way to expand functionality. When scripting is not enough or not efficient, plugins can be written for Nau to complement Nau's pass functionality. An example is provided in project nauPassPlugin.

The accompanying interface application (Composer) has debugging features that help the construction of such pipelines. With Composer the user can explore all the settings relating to the inputs and outputs of the graphics pipeline.

It can be used as a teaching tool for shader and ray tracing courses providing a seamless integration between the two rendering approaches.

Documentation at http://nau3d.di.uminho.pt/

# credits

Besides myself, many people have contributed to this project:

* Bruno Oliveira - software architecture, programming
* Pedro Ângelo - programming
* Marta Pereira - initial implementation of the event system manager
* Leander Beernaert - initial Linux version, initial CMake building system
* André Lui - initial implementation of the debug features 
* Jaime Campos and João Meira - initial Bullet integration
* David Leal - Bullet and PhysX plugins


ImGuiFileDialog from https://github.com/aiekick/ImGuiFileDialog

Nau3D uses the following 3rd party libraries:

* Optix (optional)
* imGui
* glBinding for OpenGL bindings and call tracing (https://github.com/cginternals/glbinding)
* Assimp for 3D asset loading (https://www.assimp.org/)
* Devil for image loading (http://openil.sourceforge.net/)

Interface applications:

* GLFW
* GLUT

All source code or libs is provided in the package (apart from Opix) to prevent disruption when any of those packages is updates.

# CMAKE settings and requirements

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
* Set the variable NAU_OPTIX_DIR to optix's installation directory
* CUDA is usually found by cmake and doesn't need any extra steps.
* If either CUDA or Optix are not found the process goes on without Optix support.		


# running

* composerImGUI is an almost complete interface to NAU3D with ImGui
* nauGLUT provides a simple example of a GLUT application working with NAU3D

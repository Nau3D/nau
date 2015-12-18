#include "../../src/nau.h"

#include "../../src/nau/debug/profile.h"

#ifdef _WIN32
#	ifdef _DEBUG
#		pragma comment(lib, "glbindingd.lib")
#	else
#		pragma comment(lib, "glbinding.lib")
#	endif
#endif

#define GLBINDING_STATIC
#include <glbinding/gl/gl.h>
#include <glbinding/Binding.h>
using namespace gl;

#define FREEGLUT_STATIC
#include <GL/freeglut.h>

#include <stdio.h>
// ------------------------------------------------------------
//
//			Reshape Callback Function
//
// ------------------------------------------------------------


void changeSize(int w, int h) {

	NAU->setWindowSize(w, h);
}


// --------------------------------------------------------
//
//			Render Stuff
//
// --------------------------------------------------------

void renderScene() {

	{
		PROFILE("Nau");
		NAU->step();
	}
	// swap buffers
	{
		PROFILE("Swap");
		glutSwapBuffers();
	}
#if NAU_PROFILE == NAU_PROFILE_CPU_AND_GPU
	Profile::CollectQueryResults();
#endif
	if (NAU->getProfileResetRequest())
		Profile::Reset();
}


// ------------------------------------------------------------
//
//			Main function
//
// ------------------------------------------------------------

#include "../../src/nau/system/file.h"

int main(int argc, char **argv) {

	int w = 640, h= 360;

	//  GLUT initialization
	glutInit(&argc, argv);
	// Standard display mode plus multisample (to provide some antialiasing)
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA | GLUT_MULTISAMPLE);
	// the OpenGL version (major, minor)
	glutInitContextVersion(4, 4);
	// Profile selection, the core profile ensures no deprecated functions are used
	glutInitContextProfile(GLUT_CORE_PROFILE);


	// standard glut settings
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(w, h);
	glutCreateWindow("");

	//  Callback Registration
	glutDisplayFunc(renderScene);
	glutReshapeFunc(changeSize);
	glutIdleFunc(renderScene);

	glbinding::Binding::initialize(false);

	// Display some general info
	printf("Vendor: %s\n", glGetString(GL_VENDOR));
	printf("Renderer: %s\n", glGetString(GL_RENDERER));
	printf("Version: %s\n", glGetString(GL_VERSION));
	printf("GLSL: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
	ContextProfileMask param;
	glGetIntegerv(GL_CONTEXT_PROFILE_MASK, (int *)&param);
	if (param == GL_CONTEXT_CORE_PROFILE_BIT)
		printf("Context Profile: Core\n");
	else
		printf("Context Profile: Compatibility\n");

	std::string s;

	try {
		NAU->init(true);
		if (argc == 1)
			s = "C:\\nau\\head\\projects\\simpe\\simple.xml";
		else
			s = std::string(argv[1]);
		std::string appPath = nau::system::File::GetAppFolder();
		std::string cleanAppPath = nau::system::File::CleanFullPath(appPath);
		std::string full = nau::system::File::GetFullPath(appPath, s);
		NAU->setWindowSize(w, h);
		NAU->readProjectFile(full, &w, &h);
		if (h != 0)
			glutReshapeWindow(w, h);
	}
	catch (std::string s) {
		printf("%s\n", s.c_str());
		exit(0);
	}
	//  GLUT main loop
	glutMainLoop();

	// because standard C++ requires a return value
	return(1);

}


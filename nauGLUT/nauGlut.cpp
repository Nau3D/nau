#define GLBINDING_STATIC
#include <glbinding/Binding.h>
#include <glbinding/gl/gl.h>

#include <nau.h>
#include <nau/debug/profile.h>
#include <nau/event/eventFactory.h>
#include <nau/event/cameraMotion.h>
#include <nau/event/cameraOrientation.h>

#define FREEGLUT_STATIC
#include <GL/freeglut.h>

using namespace gl;

#include <stdio.h>

Nau *nauInstance = NULL;



class Listener : public nau::event_::IListener {

public:
	std::string name;

	Listener(): name("Listener") { EVENTMANAGER->addListener("LOG", this); };
	~Listener() {};

	void eventReceived(const std::string &sender, const std::string &eventType,
			const std::shared_ptr<IEventData> &evt) {

		printf("Nau Log: %s\n", (*(std::string *)evt->getData()).c_str());
	};

	std::string &getName() {
		return name;
	}
};

Listener listLog;
// ------------------------------------------------------------
//
//			Reshape Callback Function
//
// ------------------------------------------------------------


void changeSize(int w, int h) {

	std::shared_ptr<nau::event_::IEventData> e3 = nau::event_::EventFactory::Create("Vec3");
	e3->setData(new vec3((float)w, (float)h, 0.0f));
	EVENTMANAGER->notifyEvent("WINDOW_SIZE_CHANGED", "Canvas", "", e3);
}


// --------------------------------------------------------
//
//			Render Stuff
//
// --------------------------------------------------------

void renderScene() {

	{
		PROFILE("Nau");
		nauInstance->step();
	}
	// swap buffers
	{
		PROFILE("Swap");
		glutSwapBuffers();
	}
#if NAU_PROFILE == NAU_PROFILE_CPU_AND_GPU
	Profile::CollectQueryResults();
#endif
	if (nauInstance->getProfileResetRequest())
		Profile::Reset();
}

// --------------------------------------------------------
//
//			Keyboard & Mouse
//
// --------------------------------------------------------


void processKeys(unsigned char c, int xx, int yy) {

	int mod = 0, glutMod;
	float velocity = 0.05f;

	glutMod = glutGetModifiers();

	if (glutMod & GLUT_ACTIVE_SHIFT) {
		mod |= Nau::KEY_MOD_SHIFT;
		velocity *= 10.0f;
	}
	else if (glutMod & GLUT_ACTIVE_ALT)
		mod |= Nau::KEY_MOD_ALT;
	else if (glutMod & GLUT_ACTIVE_CTRL) {
		mod |= Nau::KEY_MOD_CTRL;
		velocity *= 100.0f;
	}
	if (nauInstance->keyPressed(c, mod))
		return;

	// w = 0111 0111 W = 0101 0111  CTRL-W = 0001 0111
	// the least five significant digits remain the same :-)

	switch (c&31) {

	case 23: 
		{
			nau::event_::CameraMotion c("FORWARD", velocity);
			std::shared_ptr<IEventData> e = nau::event_::EventFactory::Create("Camera Motion");
			e->setData(&c);
			EVENTMANAGER->notifyEvent("CAMERA_MOTION", "MainCanvas", "", e);
		}
		break;
	case 19:
		{
			nau::event_::CameraMotion c("BACKWARD", velocity);
			std::shared_ptr<IEventData> e = nau::event_::EventFactory::Create("Camera Motion");
			e->setData(&c);
			EVENTMANAGER->notifyEvent("CAMERA_MOTION", "MainCanvas", "", e);
		}
		break;
	case 1:
		{
			nau::event_::CameraMotion c("LEFT", velocity);
			std::shared_ptr<IEventData> e = nau::event_::EventFactory::Create("Camera Motion");
			e->setData(&c);
			EVENTMANAGER->notifyEvent("CAMERA_MOTION", "MainCanvas", "", e);
		}
	break;
	case 4:
		{
			nau::event_::CameraMotion c("RIGHT", velocity);
			std::shared_ptr<IEventData> e = nau::event_::EventFactory::Create("Camera Motion");
			e->setData(&c);
			EVENTMANAGER->notifyEvent("CAMERA_MOTION", "MainCanvas", "", e);
		}
	break;

	}
}


bool tracking = false;
int oldX, oldY;
float oldAlpha, oldBeta;


void processMouseButtons(int button, int state, int xx, int yy) {

	Nau::MouseButton nauButton;
	switch(button) {
	case GLUT_LEFT_BUTTON: nauButton = Nau::MouseButton::LEFT; break;
	case GLUT_RIGHT_BUTTON: nauButton = Nau::MouseButton::RIGHT; break;
	case GLUT_MIDDLE_BUTTON: nauButton = Nau::MouseButton::MIDDLE; break;
	}
	Nau::MouseAction nauAction;
	switch (state) {
	case GLUT_DOWN: nauAction = Nau::MouseAction::PRESSED; break;
	case GLUT_UP: nauAction = Nau::MouseAction::RELEASED; break;
	}
	if (nauInstance->mouseButton(nauAction, nauButton, xx, yy))
		return;

	if (button == GLUT_LEFT_BUTTON) {

		if (state == GLUT_DOWN) {
			tracking = true;
			Camera *c = NAU->getActiveCamera();

			oldBeta = c->getPropf(Camera::ELEVATION_ANGLE);
			oldAlpha = c->getPropf(Camera::ZX_ANGLE);

			oldX = xx;
			oldY = yy;
		}
		else
			tracking = false;
	}
}


void processPassiveMouseMotion(int xx, int yy) {

	nauInstance->mouseMotion(xx, yy);
}



void processMouseMotion(int xx, int yy) {

	if (nauInstance->mouseMotion(xx, yy))
		return;

	float alpha, beta;

	if (!tracking)
		return;

	float m_ScaleFactor = 1.0f / 100.0f;

	alpha = oldAlpha - (float)(xx - oldX) * m_ScaleFactor;
	beta = oldBeta + (float)(oldY - yy) * m_ScaleFactor;

	nau::event_::CameraOrientation c(alpha, beta);
	std::shared_ptr<IEventData> e = nau::event_::EventFactory::Create("Camera Orientation");
	e->setData(&c);
	EVENTMANAGER->notifyEvent("CAMERA_ORIENTATION", "MainCanvas", "", e);
}

// ------------------------------------------------------------
//
//			Main function
//
// ------------------------------------------------------------

#include <nau/system/file.h>
#include <iostream>
//#include <limits.h>
//#include <unistd.h>

int main(int argc, char **argv) {
 //char path_save[PATH_MAX];
 // char abs_exe_path[PATH_MAX];
 // char *p;

 // if(!(p = strrchr(argv[0], '/')))
 //   getcwd(abs_exe_path, sizeof(abs_exe_path));
 // else
 // {
 //   *p = '\0';
 //   getcwd(path_save, sizeof(path_save));
 //   chdir(argv[0]);
 //   getcwd(abs_exe_path, sizeof(abs_exe_path));
 //   chdir(path_save);
 // }

 // printf("Absolute path to executable is: %s\n", abs_exe_path);


	if (argc == 1) {
		printf("The application requires the name of a project as a command line parameter\n");
		return 1;
	}

	int w = 640, h = 360;
	//  GLUT initialization
	glutInit(&argc, argv);
	// Standard display mode plus multisample (to provide some antialiasing)
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA | GLUT_MULTISAMPLE);
	// the OpenGL version (major, minor)
	glutInitContextVersion(3, 0);
	// Profile selection, the core profile ensures no deprecated functions are used
	//glutInitContextProfile(GLUT_CORE_PROFILE);


	// standard glut settings
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(w, h);
	glutCreateWindow("Nau GLUT Demo");

	//  Callback Registration
	glutDisplayFunc(renderScene);
	glutReshapeFunc(changeSize);
	glutIdleFunc(renderScene);

	glutKeyboardFunc(processKeys);
	glutMouseFunc(processMouseButtons);
	glutMotionFunc(processMouseMotion);
	glutPassiveMotionFunc(processPassiveMouseMotion);

	glbinding::Binding::initialize(false);

	// Display some general info
	printf("Vendor: %s\n", glGetString(GL_VENDOR));
	printf("Renderer: %s\n", glGetString(GL_RENDERER));
	printf("Version: %s\n", glGetString(GL_VERSION));
	printf("GLSL: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
	int param;
	glGetIntegerv(GL_CONTEXT_PROFILE_MASK, (int *)&param);
	if (param == (int)GL_CONTEXT_CORE_PROFILE_BIT)
		printf("Context Profile: Core\n");
	else
		printf("Context Profile: Compatibility\n");

	std::string s;

	nauInstance = (Nau *)Nau::GetInstance();

	try {
		s = std::string(argv[1]);
		std::string appPath = nau::system::File::GetAppFolder();
		std::string cleanAppPath = nau::system::File::CleanFullPath(appPath);
		std::string full = nau::system::File::GetFullPath(appPath, s);

	//printf("%s\n%s\n%s<n", appPath.c_str(),cleanAppPath.c_str(), full.c_str());
		nauInstance->init(false);
		nauInstance->setWindowSize(w, h);
		nauInstance->readProjectFile(full, &w, &h);
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
	return 1;

}


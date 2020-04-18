

#include <math.h>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define M_PIf float(M_PI)

#include <dirent.h>


#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include "ImGuiFileDialog.h"

#include <logIMGUI.h>

static AppConsole console;

#define GLBINDING_STATIC
#define GLFW_INCLUDE_NONE         // GLFW including OpenGL headers causes ambiguity or multiple definition errors.
#include <glbinding/Binding.h>  // Initialize with glbinding::initialize()
#include <glbinding/gl/gl.h>
using namespace gl;
using namespace glbinding;

// GLFW is the toolkit to interface with the OS
#include <GLFW/glfw3.h>




#include <nau.h>
#include <nau/debug/profile.h>
#include <nau/event/eventFactory.h>
#include <nau/event/cameraMotion.h>
#include <nau/event/cameraOrientation.h>
#include <nau/interface/interface.h>
#include <nau/loader/iTextureLoader.h>
#include <nau/math/matrix.h>
#include <nau/math/number.h>
#include <nau/math/vec2.h>
#include <nau/math/vec3.h>
#include <nau/math/vec4.h>

#include <nau/system/file.h>



#include <stdio.h>
#include <iostream>

Nau* nauInstance = NULL;

bool passFlowControl_nauPaused = false;
bool passFlowControl_stepPassRequest = false;
bool passFlowControl_framePassRequest = false;
int passFlowControl_framePassEnd = 0;
bool passFlowControl_framePassStageCompleteFrame = false;
bool passFlowControl_stepTillEndOfPipeline = false;

std::map<std::string, bool> projectWindows;

bool nauLoaded = false;
time_t nauProjectStartTime;


GLFWwindow* window;


bool tracking = false;
int oldX, oldY;
float oldAlpha, oldBeta;

nau::util::Tree* programInfo = NULL;

bool shaderDebugLogAvailable = false;

class Listener : public nau::event_::IListener {

public:
	std::string name;

	Listener() : name("Listener") { 
		registerListeners();
	};
	~Listener() {};

	void eventReceived(const std::string& sender, const std::string& eventType,
		const std::shared_ptr<IEventData>& evt) {

		if (eventType == "SHADER_DEBUG_INFO_AVAILABLE") {
			programInfo = RENDERER->getShaderDebugTree();
		}
		else if (eventType == "LOG") {
			console.AddLog("Nau Log: %s\n", (*(std::string*)evt->getData()).c_str());
			printf("%s\n", (*(std::string*)evt->getData()).c_str());
		}
	};

	void registerListeners() {
		EVENTMANAGER->addListener("LOG", this);
		EVENTMANAGER->addListener("SHADER_DEBUG_INFO_AVAILABLE", this);
		EVENTMANAGER->addListener("TRACE_FILE_READY", this);
	}
	std::string& getName() {
		return name;
	}
};

Listener listLog;

// ------------------------------------------------------------
//
// Reshape Callback Function
//

void changeSize(GLFWwindow* window, int w, int h) {

	// Prevent a divide by zero, when window is too short
	// (you cant make a window of zero width).
	if(h == 0)
		h = 1;

	// Set the viewport to be the entire window
	std::shared_ptr<nau::event_::IEventData> e3 = nau::event_::EventFactory::Create("Vec3");
	e3->setData(new vec3((float)w, (float)h, 0.0f));
	EVENTMANAGER->notifyEvent("WINDOW_SIZE_CHANGED", "Canvas", "", e3);
}


// ------------------------------------------------------------
//
// Render stuff
//

int frameCounter = 0;
double previousTime = 0;

void renderScene(void) {

	{
		PROFILE("Nau");
		nauInstance->step();
	}

	//  FPS Counter
	frameCounter++;

	double t = glfwGetTime();
	if (t - previousTime > 1) {
		char fps[256];

		sprintf(fps, "NAU - %s (FPS: %d - Triangles: %u)", NAU->getProjectName().c_str(), frameCounter, RENDERER->getCounter(nau::render::IRenderer::TRIANGLE_COUNTER));
		glfwSetWindowTitle(window, fps);
		frameCounter = 0;
		previousTime = t;
	}
}


// ------------------------------------------------------------
//
// Events from the Keyboard
//

//void processKeys(GLFWwindow* window, unsigned int key)
void processKeys(GLFWwindow* window, int key, int scancode, int action, int mods)
{

	int mod = 0;
	float velocity = 0.05f;	
	if (action == GLFW_RELEASE)
		if (nauInstance->keyPressed((unsigned char)key, mod))
			return;



	std::string s;

	if (mods & GLFW_MOD_SHIFT)
		velocity *= 10;
	else if (mods & GLFW_MOD_CONTROL)
		velocity *= 100;
	//if (action == GLFW_PRESS)
	//	return;

	switch (key) {

	case GLFW_KEY_W:
	{
		nau::event_::CameraMotion c("FORWARD", velocity);
		std::shared_ptr<IEventData> e = nau::event_::EventFactory::Create("Camera Motion");
		e->setData(&c);
		EVENTMANAGER->notifyEvent("CAMERA_MOTION", "MainCanvas", "", e);
	}
	break;
	case GLFW_KEY_S:
	{
		nau::event_::CameraMotion c("BACKWARD", velocity);
		std::shared_ptr<IEventData> e = nau::event_::EventFactory::Create("Camera Motion");
		e->setData(&c);
		EVENTMANAGER->notifyEvent("CAMERA_MOTION", "MainCanvas", "", e);
	}
	break;
	case GLFW_KEY_A:
	{
		nau::event_::CameraMotion c("LEFT", velocity);
		std::shared_ptr<IEventData> e = nau::event_::EventFactory::Create("Camera Motion");
		e->setData(&c);
		EVENTMANAGER->notifyEvent("CAMERA_MOTION", "MainCanvas", "", e);
	}
	break;
	case GLFW_KEY_D:
	{
		nau::event_::CameraMotion c("RIGHT", velocity);
		std::shared_ptr<IEventData> e = nau::event_::EventFactory::Create("Camera Motion");
		e->setData(&c);
		EVENTMANAGER->notifyEvent("CAMERA_MOTION", "MainCanvas", "", e);
	}
	break;
	case GLFW_KEY_Q:
	{
		nau::event_::CameraMotion c("UP", velocity);
		std::shared_ptr<IEventData> e = nau::event_::EventFactory::Create("Camera Motion");
		e->setData(&c);
		EVENTMANAGER->notifyEvent("CAMERA_MOTION", "MainCanvas", "", e);
	}
	break;
	case GLFW_KEY_Z:
	{
		nau::event_::CameraMotion c("DOWN", velocity);
		std::shared_ptr<IEventData> e = nau::event_::EventFactory::Create("Camera Motion");
		e->setData(&c);
		EVENTMANAGER->notifyEvent("CAMERA_MOTION", "MainCanvas", "", e);
	}
	break;
	case GLFW_KEY_R:
	{
		if (mods & GLFW_MOD_CONTROL) {
			NAU->setProfileResetRequest();
		}
	}
	break;
	}

	
}


// ------------------------------------------------------------
//
// Mouse Events
//

// Track mouse motion while buttons are pressed
void processMouseMotion(GLFWwindow* window, double xx, double yy) {

	if (nauInstance->mouseMotion((int)xx, (int)yy))
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


void processMouseButtons(GLFWwindow* window, int button, int action, int mods) {

	double xx, yy;
	glfwGetCursorPos(window, &xx, &yy);	
	Nau::MouseButton nauButton;
	switch (button) {
		case GLFW_MOUSE_BUTTON_LEFT: nauButton = Nau::MouseButton::LEFT; break;
		case GLFW_MOUSE_BUTTON_RIGHT: nauButton = Nau::MouseButton::RIGHT; break;
		case GLFW_MOUSE_BUTTON_MIDDLE: nauButton = Nau::MouseButton::MIDDLE; break;
	}
	Nau::MouseAction nauAction;
	switch (action) {
	case GLFW_PRESS: nauAction = Nau::MouseAction::PRESSED; break;
	case GLFW_RELEASE: nauAction = Nau::MouseAction::RELEASED; break;
	}
	if (nauInstance->mouseButton(nauAction, nauButton, (int)xx, (int)yy))
		return;

// do this to avoid processing mouse events when the mouse is initially clicked inside a ImGui window
	if (ImGui::GetIO().WantCaptureMouse)
		return;

	if (action == GLFW_PRESS) {

		if (button == GLFW_MOUSE_BUTTON_LEFT) {
			glfwSetCursorPosCallback(window, processMouseMotion);
			tracking = true;
			Camera* c = NAU->getActiveCamera();

			oldBeta = c->getPropf(Camera::ELEVATION_ANGLE);
			oldAlpha = c->getPropf(Camera::ZX_ANGLE);

			oldX = (int)xx;
			oldY = (int)yy;
		}
	}
	else {
		tracking = false;
		glfwSetCursorPosCallback(window, NULL);
	}
	
}


bool showMessageBox = false;
std::string messageBoxTitle = "";
std::string messageBoxMessage = "";



void loadProject(const char* s) {


	LOG_trace("Loading project %s", s);

	try {
		nauInstance->clear();
		listLog.registerListeners();
		projectWindows.clear();
		int width = 0, height = 0;
		std::string ProjectFile(s);
		nauInstance->readProjectFile(ProjectFile, &width, &height);
		if (width) {
			glfwSetWindowSize(window, width, height);
			changeSize(window, width, height);
		}

		LOG_trace("Loading done");

	}
	catch (nau::ProjectLoaderError & e) {
		showMessageBox = true;
		messageBoxTitle = "Project loader message";
		messageBoxMessage = e.getException();
		printf("%s\n", e.getException().c_str());
	}
	catch (std::string s) {
		showMessageBox = true;
		messageBoxTitle = "Project loader message";
		messageBoxMessage = s;
		printf("%s\n", s.c_str());
	}
	LOG_trace("Project loaded");
}


void loadModel(const char* s) {


	LOG_trace("Loading project %s", s);

	try {
		nauInstance->clear();
		projectWindows.clear();
		std::string ProjectFile(s);
		nauInstance->readModel(ProjectFile);
		listLog.registerListeners();
		LOG_trace("Loading done");

	}
	catch (std::string s) {
		printf("%s\n", s.c_str());
	}
	LOG_trace("Model loaded");
}


// ------------------------------------------------------------
//
// ImGUI stuff
//

int findIndex(std::vector<std::string>& a, std::string to_find) {

	int i = 0;
	while (i < a.size() && a[i] != to_find)
		++i;
	if (i == a.size())
		return -1;
	else
		return i;
}

bool combo(std::string title, std::vector<std::string> items, std::string activeItem, int* resIndex) {

	bool ok;
	int index = findIndex(items, activeItem);

	size_t k = 0;
	for (auto& str : items) {
		k += str.length() + 1;
	}
	k++;

	std::vector<char> options;
	options.resize(k);
	int l = 0;
	for (auto& str : items) {
		for (int i = 0; i < str.length(); ++i) {
			options[l++] = str[i];
		}
		options[l++] = '\0';
	}
	options[l] = '\0';

	int prevIndex = index;
	if (ImGui::Combo(title.c_str(), resIndex, (char *)&options[0], (int)items.size()))
		ok = true;
	else
		ok = false;
	return ok;
}





void createInt(std::unique_ptr<Attribute>& attr, AttributeValues* attribVal, bool readOnly) {

	bool editable = !attr->getReadOnlyFlag();
	int step = 1;
	int i = attribVal->getPropi((AttributeValues::IntProperty)attr->getId());
	if (editable && !readOnly) {
		if (ImGui::InputScalar(attr->getName().c_str(), ImGuiDataType_S32, &i, &step, NULL, "%d")) {			
			attribVal->setPropi((AttributeValues::IntProperty)attr->getId(), i);
		}
	}
	else {
		ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_SliderGrab, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.9f));

		ImGui::InputScalar(attr->getName().c_str(), ImGuiDataType_S32, &i, NULL, NULL, "%d");
		ImGui::PopStyleColor(4);
	}
}

void createIVec2(std::unique_ptr<Attribute>& attr, AttributeValues* attribVal, bool readOnly) {

	bool editable = !attr->getReadOnlyFlag();
	nau::math::ivec2 i2 = attribVal->getPropi2((AttributeValues::Int2Property)attr->getId());
	if (editable && !readOnly) {
		if (ImGui::InputInt2(attr->getName().c_str(), (int *)i2.getPtr())) {
			attribVal->setPropi2((AttributeValues::Int2Property)attr->getId(), i2);
		}
	}
	else {
		ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_SliderGrab, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.9f));

		ImGui::InputInt2(attr->getName().c_str(), (int*)i2.getPtr());
		ImGui::PopStyleColor(4);
	}
}

void createIVec3(std::unique_ptr<Attribute>& attr, AttributeValues* attribVal, bool readOnly) {

	bool editable = !attr->getReadOnlyFlag();
	nau::math::ivec3 i3 = attribVal->getPropi3((AttributeValues::Int3Property)attr->getId());
	if (editable && !readOnly) {
		if (ImGui::InputInt3(attr->getName().c_str(), (int*)i3.getPtr())) {
			attribVal->setPropi3((AttributeValues::Int3Property)attr->getId(), i3);
		}
	}
	else {
		ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_SliderGrab, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.9f));

		ImGui::InputInt2(attr->getName().c_str(), (int*)i3.getPtr());
		ImGui::PopStyleColor(4);
	}
}

void createIVec4(std::unique_ptr<Attribute>& attr, AttributeValues* attribVal, bool readOnly) {

	bool editable = !attr->getReadOnlyFlag();
	nau::math::ivec4 i4 = attribVal->getPropi4((AttributeValues::Int4Property)attr->getId());
	if (editable && !readOnly) {
		if (ImGui::InputInt4(attr->getName().c_str(), (int*)i4.getPtr())) {
			attribVal->setPropi4((AttributeValues::Int4Property)attr->getId(), i4);
		}
	}
	else {
		ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_SliderGrab, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.9f));

		ImGui::InputInt2(attr->getName().c_str(), (int*)i4.getPtr());
		ImGui::PopStyleColor(4);
	}
}

void createFloat(std::unique_ptr<Attribute>& attr, AttributeValues* attribVal, bool readOnly) {

	bool editable = !attr->getReadOnlyFlag();
	float f = attribVal->getPropf((AttributeValues::FloatProperty)attr->getId());
	if (editable && !readOnly) {
		if (ImGui::InputScalar(attr->getName().c_str(), ImGuiDataType_Float, &f, NULL, NULL)) {
			attribVal->setPropf((AttributeValues::FloatProperty)attr->getId(), f);
		}
	}
	else {
		ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_SliderGrab, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.9f));

		ImGui::InputScalar(attr->getName().c_str(), ImGuiDataType_Float, &f, NULL, NULL);
		ImGui::PopStyleColor(4);
	}
}


void createVec2(std::unique_ptr<Attribute>& attr, AttributeValues* attribVal, bool readOnly) {

	bool editable = !attr->getReadOnlyFlag();
	int step = 1;
	nau::math::vec2 v2 = attribVal->getPropf2((AttributeValues::Float2Property)attr->getId());
	if (editable && !readOnly) {
		if (ImGui::InputFloat2(attr->getName().c_str(), (float*)v2.getPtr())) {
			attribVal->setPropf2((AttributeValues::Float2Property)attr->getId(), v2);
		}
	}
	else {
		ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_SliderGrab, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.9f));

		ImGui::InputFloat2(attr->getName().c_str(), (float*)v2.getPtr());
		ImGui::PopStyleColor(4);
	}
}

void createVec3(std::unique_ptr<Attribute>& attr, AttributeValues* attribVal, bool readOnly) {

	bool editable = !attr->getReadOnlyFlag();
	int step = 1;
	nau::math::vec3 v3 = attribVal->getPropf3((AttributeValues::Float3Property)attr->getId());
	if (editable && !readOnly) {
		if (ImGui::InputFloat3(attr->getName().c_str(), (float*)v3.getPtr())) {
			attribVal->setPropf3((AttributeValues::Float3Property)attr->getId(), v3);
		}
	}
	else {
		ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_SliderGrab, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.9f));

		ImGui::InputFloat3(attr->getName().c_str(), (float*)v3.getPtr());
		ImGui::PopStyleColor(4);
	}
}

void createVec4(std::unique_ptr<Attribute>& attr, AttributeValues* attribVal, bool readOnly) {

	bool editable = !attr->getReadOnlyFlag();
	int step = 1;
	nau::math::vec4 v4 = attribVal->getPropf4((AttributeValues::Float4Property)attr->getId());
	if (editable && !readOnly) {
		if (ImGui::InputFloat4(attr->getName().c_str(), (float *)v4.getPtr())) {
			attribVal->setPropf4((AttributeValues::Float4Property)attr->getId(), v4);
		}
	}
	else {
		ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_SliderGrab, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.9f));

		ImGui::InputFloat4(attr->getName().c_str(), (float*)v4.getPtr());
		ImGui::PopStyleColor(4);
	}
}


void createColor(std::unique_ptr<Attribute>& attr, AttributeValues* attribVal, bool readOnly) {

	bool editable = !attr->getReadOnlyFlag();
	int step = 1;
	nau::math::vec4 v4 = attribVal->getPropf4((AttributeValues::Float4Property)attr->getId());

	if (editable && !readOnly) {
		if (ImGui::ColorEdit4(attr->getName().c_str(), (float*)v4.getPtr()), ImGuiColorEditFlags_Float) {
			attribVal->setPropf4((AttributeValues::Float4Property)attr->getId(), v4);
		}
	}
	else {
		ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_SliderGrab, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.9f));

		ImGui::ColorEdit4(attr->getName().c_str(), (float*)v4.getPtr());
		ImGui::PopStyleColor(4);
	}
}


void createUInt(std::unique_ptr<Attribute>& attr, AttributeValues* attribVal, bool readOnly) {

	bool editable = !attr->getReadOnlyFlag();
	int step = 1;
	unsigned int ui = attribVal->getPropui((AttributeValues::UIntProperty)attr->getId());
	if (editable && !readOnly) {
		if (ImGui::InputScalar(attr->getName().c_str(), ImGuiDataType_U32, &ui, &step, NULL, "%u")) {
			attribVal->setPropui((AttributeValues::UIntProperty)attr->getId(), ui);
		}
	}
	else {
		ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_SliderGrab, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.9f));

		ImGui::InputScalar(attr->getName().c_str(), ImGuiDataType_U32, &ui, NULL, NULL, "%u");
		ImGui::PopStyleColor(4);
	}
}


void createUIVec2(std::unique_ptr<Attribute>& attr, AttributeValues* attribVal, bool readOnly) {

	bool editable = !attr->getReadOnlyFlag();
	int step = 1;
	nau::math::uivec2 ui2 = attribVal->getPropui2((AttributeValues::UInt2Property)attr->getId());
	if (editable && !readOnly) {
		if (ImGui::InputScalarN(attr->getName().c_str(), ImGuiDataType_U32, (unsigned int *)ui2.getPtr(), 2)) {
			attribVal->setPropui2((AttributeValues::UInt2Property)attr->getId(), ui2);
		}
	}
	else {
		ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_SliderGrab, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.9f));

		ImGui::InputScalarN(attr->getName().c_str(), ImGuiDataType_U32, (unsigned int*)ui2.getPtr(), 2);
		ImGui::PopStyleColor(4);
	}
}

void createUIVec3(std::unique_ptr<Attribute>& attr, AttributeValues* attribVal, bool readOnly) {

	bool editable = !attr->getReadOnlyFlag();
	int step = 1;
	nau::math::uivec3 ui3 = attribVal->getPropui3((AttributeValues::UInt3Property)attr->getId());
	if (editable && !readOnly) {
		if (ImGui::InputScalarN(attr->getName().c_str(), ImGuiDataType_U32, (unsigned int*)ui3.getPtr(), 3)) {
			attribVal->setPropui3((AttributeValues::UInt3Property)attr->getId(), ui3);
		}
	}
	else {
		ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_SliderGrab, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.9f));

		ImGui::InputScalarN(attr->getName().c_str(), ImGuiDataType_U32, (unsigned int*)ui3.getPtr(), 3);
		ImGui::PopStyleColor(4);
	}
}

void createUIVec4(std::unique_ptr<Attribute>& attr, AttributeValues* attribVal, bool readOnly) {

	bool editable = !attr->getReadOnlyFlag();
	int step = 1;
	nau::math::uivec4 ui4 = attribVal->getPropui4((AttributeValues::UInt4Property)attr->getId());
	if (editable && !readOnly) {
		if (ImGui::InputScalarN(attr->getName().c_str(), ImGuiDataType_U32, (unsigned int*)ui4.getPtr(), 4)) {
			attribVal->setPropui4((AttributeValues::UInt4Property)attr->getId(), ui4);
		}
	}
	else {
		ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_SliderGrab, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.9f));

		ImGui::InputScalarN(attr->getName().c_str(), ImGuiDataType_U32, (unsigned int*)ui4.getPtr(), 4);
		ImGui::PopStyleColor(4);
	}
}


void createDouble(std::unique_ptr<Attribute>& attr, AttributeValues* attribVal, bool readOnly) {

	bool editable = !attr->getReadOnlyFlag();
	double d = attribVal->getPropd((AttributeValues::DoubleProperty)attr->getId());
	if (editable && !readOnly) {
		if (ImGui::InputScalar(attr->getName().c_str(), ImGuiDataType_Double, &d, NULL, NULL)) {
			attribVal->setPropd((AttributeValues::DoubleProperty)attr->getId(), d);
		}
	}
	else {
		ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_SliderGrab, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.9f));

		ImGui::InputScalar(attr->getName().c_str(), ImGuiDataType_Double, &d, NULL, NULL);
		ImGui::PopStyleColor(4);
	}
}

void createDVec2(std::unique_ptr<Attribute>& attr, AttributeValues* attribVal, bool readOnly) {

	bool editable = !attr->getReadOnlyFlag();
	int step = 1;
	nau::math::dvec2 d2 = attribVal->getPropd2((AttributeValues::Double2Property)attr->getId());
	if (editable && !readOnly) {
		if (ImGui::InputScalarN(attr->getName().c_str(), ImGuiDataType_Double, (double*)d2.getPtr(), 2)) {
			attribVal->setPropd2((AttributeValues::Double2Property)attr->getId(), d2);
		}
	}
	else {
		ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_SliderGrab, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.9f));

		ImGui::InputScalarN(attr->getName().c_str(), ImGuiDataType_Double, (double*)d2.getPtr(), 2);
		ImGui::PopStyleColor(4);
	}
}

void createDVec3(std::unique_ptr<Attribute>& attr, AttributeValues* attribVal, bool readOnly) {

	bool editable = !attr->getReadOnlyFlag();
	int step = 1;
	nau::math::dvec3 d3 = attribVal->getPropd3((AttributeValues::Double3Property)attr->getId());
	if (editable && !readOnly) {
		if (ImGui::InputScalarN(attr->getName().c_str(), ImGuiDataType_Double, (double*)d3.getPtr(), 2)) {
			attribVal->setPropd3((AttributeValues::Double3Property)attr->getId(), d3);
		}
	}
	else {
		ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_SliderGrab, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.9f));

		ImGui::InputScalarN(attr->getName().c_str(), ImGuiDataType_Double, (double*)d3.getPtr(), 3);
		ImGui::PopStyleColor(4);
	}
}

void createDVec4(std::unique_ptr<Attribute>& attr, AttributeValues* attribVal, bool readOnly) {

	bool editable = !attr->getReadOnlyFlag();
	int step = 1;
	nau::math::dvec4 d4 = attribVal->getPropd4((AttributeValues::Double4Property)attr->getId());
	if (editable && !readOnly) {
		if (ImGui::InputScalarN(attr->getName().c_str(), ImGuiDataType_Double, (double*)d4.getPtr(), 2)) {
			attribVal->setPropd4((AttributeValues::Double4Property)attr->getId(), d4);
		}
	}
	else {
		ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_SliderGrab, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.9f));

		ImGui::InputScalarN(attr->getName().c_str(), ImGuiDataType_Double, (double*)d4.getPtr(), 4);
		ImGui::PopStyleColor(4);
	}
}



void createEnum(std::unique_ptr<Attribute>& attr, AttributeValues* attribVal, bool readOnly) {

	bool editable = !attr->getReadOnlyFlag();
	int resIndex = attribVal->getPrope((AttributeValues::EnumProperty)attr->getId());
	std::vector<std::string> strs;

	attr->getOptionStringListSupported(&strs);
	std::vector<int> inds;
	attr->getOptionListSupported(&inds);
	int i = 0;
	while (resIndex != inds[i]) ++i;
	std::string activeItem = strs[i];

	if (!editable || readOnly) {
		ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_SliderGrab, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.9f));

		combo(attr->getName().c_str(), strs, activeItem, &i);
		ImGui::PopStyleColor(4);
	}
	else {
		if (combo(attr->getName().c_str(), strs, activeItem, &i)) {
			attribVal->setPrope((AttributeValues::EnumProperty)attr->getId(), inds[i]);
		}
	
	}
}


void createBool(std::unique_ptr<Attribute>& attr, AttributeValues* attribVal, bool readOnly) {

	bool editable = !attr->getReadOnlyFlag();
	bool current = attribVal->getPropb((AttributeValues::BoolProperty)attr->getId());

	if (editable && !readOnly) {
		if (ImGui::Checkbox(attr->getName().c_str(), &current)) {
			attribVal->setPropb((AttributeValues::BoolProperty)attr->getId(), current);
		}
	}
	else {
		ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.5f));
		ImGui::PushStyleColor(ImGuiCol_SliderGrab, (ImVec4)ImColor::HSV(1.0f, 0.0f, 0.9f));

		ImGui::Checkbox(attr->getName().c_str(), &current);
		ImGui::PopStyleColor(4);
	}

}


void addAttribute(std::unique_ptr<Attribute>& a, AttributeValues* attribVal, bool allReadOnly = false) {

	bool editable = a->getReadOnlyFlag();

	switch (a->getType()) {

		case Enums::ENUM: createEnum(a, attribVal, allReadOnly); break;
		case Enums::BOOL: createBool(a, attribVal, allReadOnly); break;
		//case Enums::BVEC4: createBVec4(pg, a); break;
		case Enums::INT: createInt(a, attribVal, allReadOnly); break;
		case Enums::IVEC2: createIVec2(a, attribVal, allReadOnly); break;
		case Enums::IVEC3: createIVec3(a, attribVal, allReadOnly); break;
		case Enums::IVEC4: createIVec4(a, attribVal, allReadOnly); break;
		case Enums::UINT: createUInt(a, attribVal, allReadOnly); break;
		case Enums::UIVEC2: createUIVec2(a, attribVal, allReadOnly); break;
		case Enums::UIVEC3: createUIVec3(a, attribVal, allReadOnly); break;
		case Enums::UIVEC4: createUIVec4(a, attribVal, allReadOnly); break;
		case Enums::FLOAT: createFloat(a, attribVal, allReadOnly); break;
		case Enums::VEC2: createVec2(a, attribVal, allReadOnly); break;
		case Enums::VEC3: createVec3(a, attribVal, allReadOnly); break;
		case Enums::VEC4: 
			if (a->getSemantics() == nau::Attribute::COLOR)
				createColor(a, attribVal, allReadOnly);
			else
				createVec4(a, attribVal, allReadOnly);
			break;
		case Enums::DOUBLE: createDouble(a, attribVal, allReadOnly); break;
		case Enums::DVEC2: createDVec2(a, attribVal, allReadOnly); break;
		case Enums::DVEC3: createDVec3(a, attribVal, allReadOnly); break;
		case Enums::DVEC4: createDVec4(a, attribVal, allReadOnly); break;
		//case Enums::MAT3: createMat3(pg, a); break;
		//case Enums::MAT4: createMat4(pg, a); break;
		//case Enums::STRING: createString(pg, a); break;
		//default: assert(false && "Missing datatype in property manager");

	}
}


bool inList(std::string attr, std::vector<std::string>& list) {

	for (auto s : list) {

		if (s == attr)
			return true;
	}
	return false;
}


void createOrderedGrid(nau::AttribSet& attribs, AttributeValues* attribVal, std::vector<std::string>& list, bool readOnly = false) {

	// First add the attributes in the list
	for (auto name : list) {

		std::unique_ptr<Attribute>& attr = attribs.get(name);
		addAttribute(attr, attribVal, readOnly);
	}
	// Add editable attributes
	std::map<std::string, std::unique_ptr<Attribute>>& attributes = attribs.getAttributes();

	for (auto& attrib : attributes) {

		if (!inList(attrib.second->getName(), list) && attrib.second->getReadOnlyFlag() == false)
			addAttribute(attrib.second, attribVal, readOnly);
	}

	// Add the non-editable attributes
	for (auto& attrib : attributes) {

		if (!inList(attrib.second->getName(), list) && attrib.second->getReadOnlyFlag() == true)
			addAttribute(attrib.second, attribVal);
	}
}


void separator(float space = 10) {
	ImGui::Dummy(ImVec2(0.0f, space));
	ImGui::Separator();
	ImGui::Dummy(ImVec2(0.0f, space));
}

void space(float space = 10) {
	ImGui::Dummy(ImVec2(0.0f, space));
}


void renderWindowPass() {

	static int activePipIndex = 0;
	static int activePassIndex = 0;

	std::vector<std::string> pips;
	RENDERMANAGER->getPipelineNames(&pips);
	if (!(activePipIndex < pips.size()))
		activePipIndex = 0;
	std::string pipName = pips[activePipIndex];

	std::vector<char> pipOptions;

	size_t k = 0;
	for (auto& str : pips) {
		k += str.length() + 1;
	}
	k++;

	pipOptions.resize(k);
	int l = 0;
	for (auto& str : pips) {
		for (int i = 0; i < str.length(); ++i) {
			pipOptions[l++] = str[i];
		}
		pipOptions[l++] = '\0';
	}
	pipOptions[l] = '\0';


	int prevIndex = activePipIndex;
	ImGui::Combo("Pipeline", &activePipIndex, (char *)&pipOptions[0], (int)pips.size());
	if (prevIndex != activePipIndex)
		activePassIndex = 0;
	pipName = pips[activePipIndex];



	std::shared_ptr<Pipeline>& pip = RENDERMANAGER->getPipeline(pipName);
	std::vector<std::string> passes;
	pip->getPassNames(&passes);
	if (!(activePassIndex < passes.size()))
		activePassIndex = 0;

	std::vector<char> passOptions;

	k = 0;
	for (auto& str : passes) {
		k += str.length() + 1;
	}
	k++;

	passOptions.resize(k);
	l = 0;
	for (auto& str : passes) {
		for (int i = 0; i < str.length(); ++i) {
			passOptions[l++] = str[i];
		}
		passOptions[l++] = '\0';
	}
	passOptions[l] = '\0';

	prevIndex = activePassIndex;
	ImGui::Combo("Pass", &activePassIndex, (char*)&passOptions[0], (int)passes.size());
	std::string passName = passes[activePassIndex];

	ImGui::Separator();

	Pass* p = RENDERMANAGER->getPipeline(pipName)->getPass(passName);
	std::string className = p->getClassName();
	ImGui::Text("Class:");
	ImGui::SameLine(150); ImGui::Text(className.c_str());

	// cameras
	std::vector<std::string> cameras;
	RENDERMANAGER->getCameraNames(&cameras);
	std::string activeCam = p->getCameraName();
	int index = findIndex(cameras, activeCam);
	ImGui::Text("Camera:");
	ImGui::SameLine(150);
	if (activeCam != "") {
		if (combo("##hidelabeCamera", cameras, activeCam, &index)) {
			p->setCamera(cameras[index]);
		}
	}
	else {
		ImGui::Text("None");
	}

	// viewports
	ImGui::Text("Viewport:");
	ImGui::SameLine(150);
	if (p->hasRenderTarget() && p->isRenderTargetEnabled()) {
		ImGui::Text("From render terget");
	}
	else {
		std::vector<std::string> viewports;
		RENDERMANAGER->getViewportNames(&viewports);
		std::shared_ptr<Viewport> vp = p->getViewport();
		if (vp) {
			std::string activeVP = vp->getName();
			index;
			if (combo("##hidelabelVP", viewports, activeVP, &index)) {
				p->setViewport(RENDERMANAGER->getViewport(viewports[index]));
			}
		}
		else {
			ImGui::Text("From camera");
		}
	}

	// render targets
	ImGui::Text("Use render target:");
	ImGui::SameLine(150);
	nau::render::IRenderTarget* rt = p->getRenderTarget();
	if (rt)
		ImGui::Text(rt->getName().c_str());
	else
		ImGui::Text("None");

	// lights
	std::vector<std::string> lights;
	RENDERMANAGER->getLightNames(&lights);
	if (lights.size() == 0) {
		ImGui::Text("Lights:");
		ImGui::SameLine(150);
		ImGui::Text("None");
	}
	else {
		if (ImGui::TreeNode("Lights:")) {
			ImVec2 dummySize(100, 10);
			for (int i = 0; i < lights.size(); i++) {
				ImGui::Dummy(dummySize); ImGui::SameLine();
				//ImGui::Text(lights[i].c_str());
				//ImGui::SameLine();
				bool checked;
				if (p->hasLight(lights[i]))
					checked = true;
				else
					checked = false;
				std::string s = "##hidel" + std::to_string(i);
				bool prev = checked;
				if (ImGui::Checkbox(lights[i].c_str(), &checked)) {
					if (checked)
						p->addLight(lights[i]);
					else
						p->removeLight(lights[i]);
				}
			}
			ImGui::TreePop();
		}
	}


	// scenes
	std::vector<std::string> scenes;
	RENDERMANAGER->getSceneNames(&scenes);
	if (scenes.size() == 0) {
		ImGui::Text("Lights:");
		ImGui::SameLine(150);
		ImGui::Text("None");
	}
	else {
		if (ImGui::TreeNode("Scenes:")) {
			ImVec2 dummySize(100, 10);
			for (int i = 0; i < scenes.size(); i++) {
				ImGui::Dummy(dummySize); ImGui::SameLine();
				//ImGui::Text(lights[i].c_str());
				//ImGui::SameLine();
				bool checked;
				if (p->hasScene(scenes[i]))
					checked = true;
				else
					checked = false;
				std::string s = "##hidel" + std::to_string(i);
				bool prev = checked;
				if (ImGui::Checkbox(scenes[i].c_str(), &checked)) {
					if (checked)
						p->addScene(scenes[i]);
					else
						p->removeScene(scenes[i]);
				}
			}
			ImGui::TreePop();
		}
	}

	ImGui::Separator();

	std::vector<std::string> order = {};
	createOrderedGrid(Pass::GetAttribs(), (AttributeValues*)p, order);

	std::vector<std::string> libList;
	std::vector<std::string> matList;
	std::vector<std::string> completeList;
	MATERIALLIBMANAGER->getNonEmptyLibNames(&libList);
	for (auto& lib : libList) {

		MATERIALLIBMANAGER->getMaterialNames(lib, &matList);
		for (auto& mat : matList)
			completeList.push_back(lib + "::" + mat);
	}

	std::vector<char> matOptions;

	k = 0;
	for (auto& str : matList) {
		k += str.length() + 1;
	}
	k++;

	matOptions.resize(k);
	l = 0;
	for (auto& str : matList) {
		for (int i = 0; i < str.length(); ++i) {
			matOptions[l++] = str[i];
		}
		matOptions[l++] = '\0';
	}
	matOptions[l] = '\0';

	std::map<std::string, nau::material::MaterialID> mm = p->getMaterialMap();

	if (ImGui::TreeNode("Material Maps:")) {
		ImVec2 dummySize(100, 10);
		for (auto mat : mm) {
			int index = findIndex(completeList, mat.second.getLibName() + "::" + mat.second.getMaterialName());
			if (ImGui::Combo(mat.first.c_str(), &index, (char *)&matOptions[0], (int)completeList.size())) {
			std::string result = completeList[index];
				int pos = (int)result.find(":");
				std::string matName = result.substr(pos + 2);
				std::string libName = result.substr(0, pos);;

				p->remapMaterial(mat.first.c_str(), libName, matName);
			}

		}
		ImGui::TreePop();
	}
}


void renderWindowCameras() {

	static int index = 0;
	std::vector<std::string> cameras;
	RENDERMANAGER->getCameraNames(&cameras);
	if (index >= cameras.size())
		index = 0;
	combo("Camera", cameras, cameras[index], &index);

	separator();

	std::shared_ptr<Camera> &c = RENDERMANAGER->getCamera(cameras[index]);
	std::string vpName = c->getViewport()->getName();

	std::vector<std::string> viewports;
	RENDERMANAGER->getViewportNames(&viewports);
	int indexvp;
	if (combo("Viewport", viewports, vpName, &indexvp))
		c->setProps((AttributeValues::StringProperty)Camera::VIEWPORT, viewports[indexvp]);
	ImGui::Separator();


	std::vector<std::string> order = { "POSITION", "VIEW", "UP", "LOOK_AT_POINT", "TYPE",
			"FOV", "NEAR", "FAR", "LEFT", "RIGHT", "BOTTOM", "TOP" };	
	createOrderedGrid(Camera::GetAttribs(), (AttributeValues*)c.get(), order);

	separator();

	if (ImGui::Button("Activate Camera", ImVec2(120, 0)))
		NAU->setActiveCameraName(cameras[index]);
	ImGui::SameLine();
	if (ImGui::Button("New Camera", ImVec2(120, 0))) {
		ImGui::OpenPopup("New Camera");

	}
	if (ImGui::BeginPopupModal("New Camera", NULL, ImGuiWindowFlags_AlwaysAutoResize))
	{
		static char str0[64];
		ImGui::InputText("input name \n\n", str0, 64);

		if (ImGui::Button("OK", ImVec2(120, 0))) { 
			if (findIndex(cameras, str0) == -1) {
				RENDERMANAGER->createCamera(str0);
				ImGui::CloseCurrentPopup();
			}
		}
		ImGui::SetItemDefaultFocus();
		ImGui::SameLine();
		if (ImGui::Button("Cancel", ImVec2(120, 0))) { ImGui::CloseCurrentPopup(); }
		ImGui::EndPopup();
	}

}


void renderWindowLights() {

	static int index = 0;
	std::vector<std::string> lights;
	RENDERMANAGER->getLightNames(&lights);
	if (index >= lights.size())
		index = 0;
	combo("Light", lights, lights[index], &index);
	ImGui::Separator();

	std::shared_ptr<Light>& c = RENDERMANAGER->getLight(lights[index]);

	std::vector<std::string> order = { "ENABLED", "POSITION", "DIRECTION", "AMBIENT", "COLOR" };
	createOrderedGrid(Light::GetAttribs(), (AttributeValues*)c.get(), order);

	separator();

	if (ImGui::Button("New Light", ImVec2(120, 0))) {
		ImGui::OpenPopup("New Light");

	}
	if (ImGui::BeginPopupModal("New Light", NULL, ImGuiWindowFlags_AlwaysAutoResize))
	{
		static char str0[64];
		ImGui::InputText("input name \n\n", str0, 64);

		if (ImGui::Button("OK", ImVec2(120, 0))) {
			if (findIndex(lights, str0) == -1) {
				RENDERMANAGER->createLight(str0);
				ImGui::CloseCurrentPopup();
			}
		}
		ImGui::SetItemDefaultFocus();
		ImGui::SameLine();
		if (ImGui::Button("Cancel", ImVec2(120, 0))) { ImGui::CloseCurrentPopup(); }
		ImGui::EndPopup();
	}

}


void renderWindowViewports() {

	static int index = 0;
	std::vector<std::string> viewports;
	RENDERMANAGER->getViewportNames(&viewports);
	if (index >= viewports.size())
		index = 0;
	combo("Viewport", viewports, viewports[index], &index);
	ImGui::Separator();

	std::shared_ptr<Viewport>& c = RENDERMANAGER->getViewport(viewports[index]);

	std::vector<std::string> order = {  };
	createOrderedGrid(Viewport::GetAttribs(), (AttributeValues*)c.get(), order);

	separator();

	if (ImGui::Button("New Viewport", ImVec2(120, 0))) {
		ImGui::OpenPopup("New Viewport");

	}
	if (ImGui::BeginPopupModal("New Viewport", NULL, ImGuiWindowFlags_AlwaysAutoResize))
	{
		static char str0[64];
		ImGui::InputText("input name \n\n", str0, 64);

		if (ImGui::Button("OK", ImVec2(120, 0))) {
			if (findIndex(viewports, str0) == -1) {
				RENDERMANAGER->createViewport(str0);
				ImGui::CloseCurrentPopup();
			}
		}
		ImGui::SetItemDefaultFocus();
		ImGui::SameLine();
		if (ImGui::Button("Cancel", ImVec2(120, 0))) { ImGui::CloseCurrentPopup(); }
		ImGui::EndPopup();
	}

}


void renderWindowScenes() {

	static int index = 0;
	std::vector<std::string> scenes;
	RENDERMANAGER->getSceneNames(&scenes);
	if (index >= scenes.size())
		index = 0;
	combo("Scenes", scenes, scenes[index], &index);

	if (ImGui::Button("Save as NBO")) {
		ImGuiFileDialog::Instance()->OpenDialog("SaveFileDlg", "Save NBO File", ".*", ".");
		strcpy(ImGuiFileDialog::FileNameBuffer,"");
	}
	separator();

	std::shared_ptr<IScene>& scene = RENDERMANAGER->getScene(scenes[index]);

	std::vector<std::string> order = { "TRANSFORM_ORDER", "TRANSLATE", "ROTATE", "SCALE" };
	createOrderedGrid(IScene::GetAttribs(), (AttributeValues*)scene.get(), order);

	separator();

	if (ImGui::TreeNode("Scene List")) {

		std::vector<std::shared_ptr<SceneObject>> objList;
		scene->getAllObjects(&objList);

		for (auto item:objList) {
			if (ImGui::TreeNode(item->getName().c_str())) {

				std::shared_ptr<IRenderable> renderable = item->getRenderable();
				int vert = renderable->getNumberOfVertices();
				std::string s = "Primitive: " + renderable->getDrawingPrimitiveString();
				ImGui::Text(s.c_str());
				s = "Vertices: " + std::to_string(renderable->getNumberOfVertices());
				ImGui::Text(s.c_str());
				if (ImGui::TreeNode("Materials: ")) {

					std::vector<std::shared_ptr<nau::material::MaterialGroup>>& mgs = renderable->getMaterialGroups();
					for (auto mg : mgs) {
						ImGui::Text((mg->getMaterialName() + std::string(" - indices: ") + std::to_string(mg->getIndexData()->getIndexSize())).c_str());
					}
					ImGui::TreePop();
				}
				ImGui::TreePop();
			}
		}
		ImGui::TreePop();

	}
	if (ImGuiFileDialog::Instance()->FileDialog("SaveFileDlg"))
	{
		// action if OK
		if (ImGuiFileDialog::Instance()->IsOk == true)
		{
			std::string filePathName = ImGuiFileDialog::Instance()->GetFinalFileName();
			std::string filePath = ImGuiFileDialog::Instance()->GetCurrentPath();
			printf(" Saving scene: %s\n", filePathName.c_str());
			nau::NAU->writeAssets("NBO", filePathName.c_str(), scenes[index]);

		}
		ImGuiFileDialog::Instance()->CloseDialog("SaveFileDlg");
	}


}


void renderWindowMaterialLibrary() {

	static int indexL = 0, indexM = 0;

	std::vector<std::string> libs, mats;
	MATERIALLIBMANAGER->getNonEmptyLibNames(&libs);
	if (indexL >= libs.size())
		indexL = 0;

	combo("Library", libs, libs[indexL], &indexL);

	MATERIALLIBMANAGER->getMaterialNames(libs[indexL], &mats);
	if (indexM >= mats.size())
	indexM = 0;
	combo("Material", mats, mats[indexM], &indexM);

	separator();

	std::shared_ptr<Material> mat = MATERIALLIBMANAGER->getMaterial(libs[indexL], mats[indexM]);

	if (ImGui::BeginTabBar("MaterialTabs")) {

		if (ImGui::BeginTabItem("Colors")) {
			space();
			std::vector<std::string> order = { "DIFFUSE", "AMBIENT", "EMISSION", "SPECULAR", "SHININES" };
			createOrderedGrid(ColorMaterial::GetAttribs(), (AttributeValues*)mat.get(), order);
			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("Shaders")) {
			space();
			std::vector<std::string>* names = RESOURCEMANAGER->getProgramNames();
			std::string progName = mat->getProgramName();
			int i = 0;
			while ((*names)[i] != progName) ++i;
			if (combo("Shader", *names, progName, &i)) {
				mat->attachProgram((*names)[i]);
			}
			space();
			if (ImGui::TreeNode("Uniforms")) {
				std::map<std::string, nau::material::ProgramValue> progValues = mat->getProgramValues();
				if (progValues.size() > 0) { // add default block
					if (ImGui::TreeNode("Default block")) {
						for (auto prog : progValues) {
							if (ImGui::TreeNode(prog.first.c_str())) {
								std::string s = "data type: " + Enums::GetDataTypeToString()[prog.second.getValueType()];
								ImGui::Text(s.c_str());
								s = "type: " + prog.second.getType();
								ImGui::Text(s.c_str());
								s = "context: " + prog.second.getContext();
								ImGui::Text(s.c_str());
								s = "component: " + prog.second.getValueOf();
								ImGui::Text(s.c_str());
								ImGui::TreePop();
							}
						}
						ImGui::TreePop();
					}
					std::map<std::pair<std::string, std::string>, nau::material::ProgramBlockValue> progBlockValues;
					progBlockValues = mat->getProgramBlockValues();
					std::set<std::string> blocks;
					for (auto pbv : progBlockValues) {
						blocks.insert(pbv.first.first);
					}

					if (blocks.size() > 0) { // display blocks
						for (auto b : blocks) {

							if (ImGui::TreeNode(b.c_str())) {
								for (auto pbv : progBlockValues) {
									if (pbv.first.first == b) {
										if (ImGui::TreeNode(pbv.first.second.c_str())) {
											std::string s = "data type: " + Enums::GetDataTypeToString()[pbv.second.getValueType()];
											ImGui::Text(s.c_str());
											s = "type: " + pbv.second.getType();
											ImGui::Text(s.c_str());
											s = "context: " + pbv.second.getContext();
											ImGui::Text(s.c_str());
											s = "component: " + pbv.second.getValueOf();
											ImGui::Text(s.c_str());
											ImGui::TreePop();
										}
									}
								}
								ImGui::TreePop();

							}
						}
					}

				}



				ImGui::TreePop();
			}

			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("Textures")) {
			space();

			static int textureUnit = 0;

			for (int i = 0; i < 8; ++i) {
				nau::material::ITexture* t = mat->getTexture(i);
				if (t == NULL) {
					std::string label = "##" + std::to_string(i);
					if (ImGui::Button(label.c_str(), ImVec2(96, 96)))
						textureUnit = i;

				}
				else {
					unsigned int id = (unsigned int)t->getPropi((AttributeValues::IntProperty)ITexture::ID);
					ImGui::PushID(i);
					if (ImGui::ImageButton((ImTextureID)id, ImVec2(96, 96), ImVec2(0, 0), ImVec2(1, 1), 0))
						textureUnit = i;
					ImGui::PopID();
				}
				if (i != 3 && i != 7) ImGui::SameLine();
			}

			ITextureSampler* ts = mat->getTextureSampler(textureUnit);
			nau::material::ITexture* t = mat->getTexture(textureUnit);
			ImGui::Combo("Texture Unit", &textureUnit, " 0\0 1\0 2\0 3\0 4\0 5\0 6\0 7\0\0");
			std::vector<std::string> tnames;
			tnames.push_back("No Texture");
			std::string tname;
			if (ts == NULL) {
				tname = "No Texture";
			}
			else {
				tname = t->getLabel();
			}
			int count = RESOURCEMANAGER->getNumTextures();
			for (unsigned int i = 0; i < (unsigned int)count; ++i) {
				ITexture* t1 = nauInstance->getResourceManager()->getTexture(i);
				std::string &s = t1->getLabel();
				tnames.push_back(s);
			}

			int nameIndex = findIndex(tnames, tname);
			if (combo("Name", tnames, tname, &nameIndex)) {
				if (nameIndex != 0) {
					ITexture* tt = RESOURCEMANAGER->getTexture(tnames[nameIndex]);
					mat->attachTexture(textureUnit, tt);
				}
				else
					mat->unsetTexture(textureUnit);

			}

			if (ts != NULL) {
				int w = t->getPropi(ITexture::WIDTH), h = t->getPropi(ITexture::HEIGHT), d = t->getPropi(ITexture::DEPTH);
	
				char sc[96];
				sprintf(sc, "W x H x D: %d x %d x %d", w,h,d);
				ImGui::Text(sc, 96, sc);
	
				std::vector<std::string> order = { "WRAP_S", "WRAP_T", "WRAP_R", "MIN_FILTER", "MAG_FILTER", "COMPARE_MODE",
					"COMPARE_FUNC", "BORDER_COLOR" };

				createOrderedGrid(ITextureSampler::GetAttribs(), (AttributeValues*)ts, order);
			}
			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("State")) {
			space();

			std::vector<std::string> order = { "DEPTH_FUNC", "DEPTH_TEST", "DEPTH_MASK", "CULL_TYPE", "CULL_FACE",
				"ORDER", "BLEND", "BLEND_SRC", "BLEND_DST", "BLEND_EQUATION", "BLEND_COLOR", "COLOR_MASK_B4" };
			nau::material::IState* m_glState = mat->getState();

			createOrderedGrid(nau::material::IState::GetAttribs(), (AttributeValues *)m_glState, order);
			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("Buffers")) {
			space();

			std::vector<unsigned int> units;
			std::vector<string> sunits;
			mat->getBufferBindings(&units);
			if (units.size() > 0) {
				for (auto unit : units)
					sunits.push_back(std::to_string(unit));
				static int indexB = 0;
				combo("Bindding point", sunits, sunits[indexB], &indexB);

				IBuffer* bu = mat->getBuffer(units[indexB]);
				std::vector<std::string> order;
				createOrderedGrid(IBuffer::GetAttribs(), (AttributeValues*)bu, order);
			}

			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("ImageTextures")) {
			space();

			std::vector<unsigned int> units;
			std::vector<string> sunits;
			mat->getImageTextureUnits(&units);
			if (units.size() > 0) {
				for (auto unit : units)
					sunits.push_back(std::to_string(unit));
				static int indexU = 0;
				combo("Unit", sunits, sunits[indexU], &indexU);

				IImageTexture* it = mat->getImageTexture(units[indexU]);
				std::vector<std::string> order;
				createOrderedGrid(IImageTexture::GetAttribs(), (AttributeValues*)it, order);
			}

			ImGui::EndTabItem();
		}
		ImGui::EndTabBar();
	}
}



void renderWindowTextureLibrary() {


	int count = RESOURCEMANAGER->getNumTextures();

	ImGui::BeginChild("scrollingTextures", ImVec2(0, 130), true, ImGuiWindowFlags_HorizontalScrollbar);
	static int textureIndex = 1;
	int framePadding;

	for (int i = 0; i < count; ++i) {
		ITexture* t = RESOURCEMANAGER->getTexture(i);
		unsigned int id = (unsigned int)t->getPropi(ITexture::ID);

		if (i > 0)
			ImGui::SameLine();
		ImGui::PushID(i);
		if (id == textureIndex)
			framePadding = 3;
		else
			framePadding = 0;
		if (ImGui::ImageButton((ImTextureID)id, ImVec2(96, 96), ImVec2(0, 1), ImVec2(1, 0), framePadding, ImColor(255, 0, 0, 255))) {
			textureIndex = id;
		}
		ImGui::PopID();
	}
	ImGui::EndChild();
	ImGui::NewLine();
	ImGui::NewLine();
	if (textureIndex != 0) {

		ITexture* t = RESOURCEMANAGER->getTextureByID(textureIndex);

		if (t == NULL)
			textureIndex = 0;
		else {
			if (ImGui::Button("Save PNG")) {
				std::string s = t->getLabel() + ".png";
				std::string sname = nau::system::File::Validate(s);
				nau::loader::ITextureLoader::Save(t, nau::loader::ITextureLoader::PNG);
			}
			if (ImGui::Button("Save RAW")) {
				char name[256];
				sprintf(name, "%s.raw", t->getLabel().c_str());
				std::string sname = nau::system::File::Validate(name);
				nau::loader::ITextureLoader::SaveRaw(t, sname);
			}
			if (ImGui::Button("Save HDR")) {
				std::string s = t->getLabel() + ".hdr";
				std::string sname = nau::system::File::Validate(s);
				nau::loader::ITextureLoader::Save(t, nau::loader::ITextureLoader::HDR);
			}
			ImGui::NewLine();
			ImGui::Text("Name"); ImGui::SameLine(150 - ImGui::GetCursorPos().x, 0);
			ImGui::Text(t->getLabel().c_str());
			ImGui::NewLine();

			std::vector<std::string> order = { "ID", "FORMAT", "TYPE", "INTERNAL_FORMAT", "DIMENSION",
				"WIDTH", "HEIGHT", "DEPTH", "LAYERS", "SAMPLES", "LEVELS", "MIPMAP", "COMPONENT_COUNT", "ELEMENT_SIZE" };
			createOrderedGrid(ITexture::GetAttribs(), (AttributeValues*)t, order, true);
		}
	}
		
}


#include <nau/render/opengl/glMaterialGroup.h>
#include <nau/geometry/mesh.h>
#include <nau/system/file.h>

void renderWindowBufferLibrary() {

	int dist = 250;
	int distSmall = 75;
	int lines = 16;
	static int activeBuffer = 0;

	std::vector <std::string> bnames;
	RESOURCEMANAGER->getBufferNames(&bnames);
	if (ImGui::BeginTabBar("BufferTabs")) {

		if (ImGui::BeginTabItem("Buffers")) {

			ImGui::BeginGroup();
			if (ImGui::TreeNode("Buffers                                   ")) {
				std::vector <std::string> bnames;
				RESOURCEMANAGER->getBufferNames(&bnames);
				for (unsigned int i = 0; i < bnames.size(); ++i) {

					IBuffer* b = RESOURCEMANAGER->getBuffer(bnames[i]);
					if (ImGui::TreeNode(nau::system::File::GetName(b->getLabel()).c_str())) {

						ImGui::Text("ID"); ImGui::SameLine(50 + distSmall - ImGui::GetCursorPos().x, 0);
						ImGui::Text("# bytes"); ImGui::SameLine(50 + distSmall * 2 - ImGui::GetCursorPos().x, 0);
						ImGui::Text("# pages"); ImGui::SameLine(50 + distSmall * 3 - ImGui::GetCursorPos().x, 0);
						//ImGui::Text("# lines"); ImGui::SameLine(50 + distSmall*4 - ImGui::GetCursorPos().x, 0);
						ImGui::Text("# components");

						int s = 0;
						for (auto t : b->getStructure()) {
							s += Enums::getSize(t);
						}
						int bsize = b->getPropui(IBuffer::SIZE);
						int totalLines = bsize / s;
						if (bsize % s != 0)
							totalLines++;

						int totalPages = totalLines / lines;
						if (totalLines % lines != 0)
							totalPages++;
						ImGui::Text(std::to_string(b->getPropi(IBuffer::ID)).c_str()); ImGui::SameLine(50 + distSmall - ImGui::GetCursorPos().x, 0);
						ImGui::Text(std::to_string(b->getPropui(IBuffer::SIZE)).c_str()); ImGui::SameLine(50 + distSmall * 2 - ImGui::GetCursorPos().x, 0);
						//ImGui::Text(std::to_string(totalLines).c_str()); ImGui::SameLine(50 + distSmall * 3 - ImGui::GetCursorPos().x, 0);
						ImGui::Text(std::to_string(totalPages).c_str()); ImGui::SameLine(50 + distSmall * 3 - ImGui::GetCursorPos().x, 0);

						ImGui::Text(std::to_string(s).c_str());
						ImGui::TreePop();

						activeBuffer = i;
					}
				}
				ImGui::TreePop();
			}

			ImGui::EndGroup();
			ImGui::SameLine();
			
			ImGui::BeginGroup();

			static int currentLine = 0;
			static int activeBuffer = 0;
			static int currentPage = 0;
			if (activeBuffer >= (int)RESOURCEMANAGER->getNumBuffers())
				activeBuffer = 0;

			IBuffer* b = RESOURCEMANAGER->getBuffer(bnames[activeBuffer]);
			unsigned int bsize = b->getPropui(IBuffer::SIZE);
			combo("Buffer", bnames, bnames[activeBuffer], &activeBuffer);

			int lineSize = 0;
			for (auto t : b->getStructure()) {
				lineSize += Enums::getSize(t);
			}
			
			int totalLines = bsize / lineSize;
			if (bsize % lineSize != 0)
				totalLines++;
			if (currentLine > totalLines)
				currentLine = totalLines < 16 ? 0 : totalLines - 16;
			int totalPages = totalLines / lines;
			//if (totalLines % lines != 0)
			//	totalPages++;
			int minPage = 0;
			ImGui::SliderScalar("page", ImGuiDataType_S32, &currentPage, &minPage, &totalPages, "%d");
			const std::vector<std::string> &enumsString = Enums::GetDataTypeToString();
			std::vector<Enums::DataType> stru = b->getStructure();
			if (b != NULL) {
				ImGui::Columns((int)(stru.size() + 1), "mycolumns"); // 4-ways, with border
				ImGui::Separator();
				ImGui::Text("Row\\Type"); ImGui::NextColumn();
				
				for (auto t : stru) {
					ImGui::Text(enumsString[t].c_str()); ImGui::NextColumn();
				}
				ImGui::Separator();

				void* bufferValues;
				int pageSize = lineSize * 16;
				int pageOffset = currentPage * pageSize;
				bufferValues = malloc(pageSize);

				int dataRead = (int)b->getData(pageOffset, pageSize, bufferValues);
				int pointerIndex = 0;
				std::string value;
				currentLine = 16 * currentPage;
				int maxRow = currentLine + 16 > totalLines ? totalLines : currentLine + 16;
				for (int row = currentLine; row < maxRow; ++row) {
					ImGui::Text(std::to_string(row).c_str()); ImGui::NextColumn();
					for (int col = 0; col < stru.size(); ++col) {
						if (pointerIndex < dataRead) {
							void* ptr = (char*)bufferValues + pointerIndex;
							value = Enums::pointerToString(stru[col], ptr);
						}
						ImGui::Text(value.c_str()); ImGui::NextColumn();
						pointerIndex += Enums::getSize(stru[col]);
					}
				}
				free(bufferValues);

				ImGui::Columns(1);
			}

			ImGui::EndGroup();
			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("VAOs")) {

			std::vector<std::string> snames;
			RENDERMANAGER->getSceneNames(&snames);
			for (auto name : snames) {
				std::shared_ptr<IScene>& scene = RENDERMANAGER->getScene(name);
				std::vector<std::shared_ptr<SceneObject>> so;
				scene->getAllObjects(&so);
				for (auto obj : so) {
					std::vector < std::shared_ptr<nau::material::MaterialGroup>>& mgs = obj->getRenderable()->getMaterialGroups();
					for (auto mg : mgs) {

						nau::render::opengl::GLMaterialGroup* g = (nau::render::opengl::GLMaterialGroup*)mg.get();
						int vao = g->getVAO();
						if (ImGui::TreeNode(std::to_string(vao).c_str())) {
							unsigned int indexID = g->getIndexData()->getBufferID();
							if (ImGui::TreeNode("Index Array")) {
								ImGui::Text("ID"); ImGui::SameLine(dist - ImGui::GetCursorPos().x, 0);
								ImGui::Text("Scene # Material"); 
								ImGui::Text(std::to_string(indexID).c_str()); ImGui::SameLine(dist - ImGui::GetCursorPos().x, 0);
								ImGui::Text(nau::system::File::GetName(g->getName()).c_str());

								ImGui::TreePop();
							}
							if (ImGui::TreeNode("Attribute Arrays")) {
								ImGui::Text("ID"); ImGui::SameLine(dist - ImGui::GetCursorPos().x, 0);
								ImGui::Text("Scene : Material");
								for (int i = 0; i < VertexData::MaxAttribs; ++i) {
									unsigned int vid = g->getParent().getVertexData()->getBufferID(i);
									
									if (vid != 0) {
										ImGui::Text(std::to_string(vid).c_str()); ImGui::SameLine(dist - ImGui::GetCursorPos().x, 0);
										ImGui::Text(nau::system::File::GetName(RESOURCEMANAGER->getBufferByID(vid)->getLabel()).c_str());

									}

								}
								ImGui::TreePop();
							}

							ImGui::TreePop();
						}
					}
				}
			}
			ImGui::EndTabItem();
		}
		ImGui::EndTabBar();
	}
}



void displayGroup(std::string type, std::vector<std::string> files) {

	ImGui::BeginGroup();
	ImGui::Text(type.c_str()); 
	ImGui::Dummy(ImVec2(200 - ImGui::GetCursorPos().x, 0 ));
	ImGui::EndGroup();
	ImGui::SameLine();
	ImGui::BeginGroup();
	for (int i = 0; i < files.size(); ++i) {
		ImGui::Text(files[i].c_str());
	}
	ImGui::EndGroup();
}

void displayBool(bool b) {
	if (b)
		ImGui::Text("True");
	else
		ImGui::Text("False");
}

#include <nau/render/opengl/glProgram.h>
#include <nau/material/iUniformBlock.h>
#include <nau/material/uniformBlockManager.h>

void renderWindowShaderLibrary() {

	std::vector<std::string> *pNames = RESOURCEMANAGER->getProgramNames();

	static int pIndex = 0;

	combo("Program", *pNames, (*pNames)[pIndex], &pIndex);

	separator();

	IProgram *p = RESOURCEMANAGER->getProgram((*pNames)[pIndex]);

	std::vector<std::string> pvf = p->getShaderFiles(IProgram::ShaderType::VERTEX_SHADER);
	std::vector<std::string> pgf = p->getShaderFiles(IProgram::ShaderType::GEOMETRY_SHADER);
	std::vector<std::string> ptcf = p->getShaderFiles(IProgram::ShaderType::TESS_CONTROL_SHADER);
	std::vector<std::string> ptef = p->getShaderFiles(IProgram::ShaderType::TESS_EVALUATION_SHADER);
	std::vector<std::string> pff = p->getShaderFiles(IProgram::ShaderType::FRAGMENT_SHADER);
	std::vector<std::string> pcf = p->getShaderFiles(IProgram::ShaderType::COMPUTE_SHADER);

	ImVec2 cpos = ImGui::GetCursorPos();
	float width = ImGui::GetWindowWidth();
	
	//ImDrawList * draw_list = ImGui::GetWindowDrawList();
	//draw_list->AddRectFilled(cpos, ImVec2(cpos.x + 300, cpos.y + 50), ImColor::HSV(0 / 7.0f, 0.6f, 0.6f));

	ImGui::TextColored((ImVec4)ImColor::HSV(0 / 7.0f, 0.6f, 0.6f),"Files");
	ImGui::Separator();
	ImGui::BeginGroup();

	displayGroup("Vertex:", pvf);
	displayGroup("Geometry:", pgf);
	displayGroup("Tess Control:", ptcf);
	displayGroup("Tess Eval:", ptef);
	displayGroup("Fragment:", pff);
	displayGroup("Compute:", pcf);

	ImGui::EndGroup();

	ImGui::Separator();
	int dist = 250;
	ImGui::Text("Link status"); ImGui::SameLine(dist - ImGui::GetCursorPos().x, 0);
	bool b = p->getPropertyb((int)GL_LINK_STATUS); displayBool(b);
	ImGui::Text("Validate status"); ImGui::SameLine(dist - ImGui::GetCursorPos().x, 0);
	b = p->getPropertyb((int)GL_VALIDATE_STATUS); displayBool(b);
	ImGui::Text("Active atomic counter buffers"); ImGui::SameLine(dist - ImGui::GetCursorPos().x, 0);
	int c  = p->getPropertyi((int)GL_ACTIVE_ATOMIC_COUNTER_BUFFERS); ImGui::Text(std::to_string(c).c_str());
	ImGui::Text("Active attributes"); ImGui::SameLine(dist - ImGui::GetCursorPos().x, 0);
	c = p->getPropertyi((int)GL_ACTIVE_ATTRIBUTES); ImGui::Text(std::to_string(c).c_str());
	ImGui::Text("Active uniforms"); ImGui::SameLine(dist - ImGui::GetCursorPos().x, 0);
	c = p->getPropertyi((int)GL_ACTIVE_UNIFORMS); ImGui::Text(std::to_string(c).c_str());
	ImGui::Separator();

	std::vector<std::string> blockNames, uniformNames;
	p->getUniformBlockNames(&blockNames);
	GLProgram *pgl = (nau::render::GLProgram*)p;
	GLUniform u;
	if (ImGui::TreeNode("Default Block")) {

		int uni = p->getNumberOfUniforms();
		for (int i = 0; i < uni; i++) {
			u = pgl->getUniform(i);
			ImGui::Text(u.getName().c_str()); ImGui::SameLine(dist - ImGui::GetCursorPos().x, 0);
			ImGui::Text(u.getStringSimpleType().c_str());
		}


		ImGui::TreePop();
	}

	IUniformBlock* ub;
	for (auto b : blockNames) {
		if (ImGui::TreeNode(b.c_str())) {
			ub = UNIFORMBLOCKMANAGER->getBlock(b);
			ub->getUniformNames(&uniformNames);
			for (auto n : uniformNames) {
				ImGui::Text(n.c_str()); ImGui::SameLine(dist - ImGui::GetCursorPos().x, 0);
				ImGui::Text(Enums::GetDataTypeToString()[ub->getUniformType(n)].c_str());
			}
			ImGui::TreePop();
		}
	}

	separator();

	if (ImGui::Button("Compile and Link")) {
		// compile shaders
		for (int i = 0; i < IProgram::SHADER_COUNT; ++i) {

			if (p->getShaderFiles((IProgram::ShaderType)i).size() != 0 && p->reloadShaderFile((IProgram::ShaderType)i)) {
				p->compileShader((IProgram::ShaderType)i);
			}
		}

		// if compiled, link program
		if (p->areCompiled())
			p->linkProgram();

	}

	separator();

	for (int i = 0; i < IProgram::SHADER_COUNT; ++i) {
		std::string infoLog = p->getShaderInfoLog((IProgram::ShaderType)i);
		if (infoLog != "")
			ImGui::Text(infoLog.c_str());
	}
}



void renderWindowAtomics() {

	int dist = 250;
	std::map<std::pair<std::string, unsigned int>, std::string>::iterator iter;
	iter = RENDERER->m_AtomicLabels.begin();
	std::vector<unsigned int> atValues = RENDERER->getAtomicCounterValues();

	iter = RENDERER->m_AtomicLabels.begin();
	if (RENDERER->m_AtomicLabels.size() == 0) 
		ImGui::TextColored(ImVec4(0.4f,0.4f,0.4f,1.0f), "No atomics defined in project");
	else
		for (unsigned int i = 0; i < RENDERER->m_AtomicLabels.size(); ++i, ++iter) {
			ImGui::Text(iter->second.c_str()); ImGui::SameLine(dist - ImGui::GetCursorPos().x, 0);
			ImGui::Text(std::to_string(atValues[i]).c_str());
		}

}



void renderProfiler(int l, int p, pTime calls, Profile::level* level, std::string &indent) {

	size_t siz;
	Profile::section* sec;

	siz = level[l].sec.size();

	for (size_t cur = 0; cur < siz; ++cur) {
		sec = &(level[l].sec[cur]);

		if (l == 0)
			calls = sec->calls;

		if ((p == -1) || (sec->parent == p)) {

			ImGui::Text((indent + sec->name).c_str()); ImGui::NextColumn();
			ImGui::Text(std::to_string((int)(sec->calls / calls)).c_str()); ImGui::NextColumn();
			ImGui::Text(std::to_string((float)(sec->totalTime) / (calls)).c_str()); ImGui::NextColumn();
			if (sec->profileGL) {
				ImGui::Text(std::to_string(sec->totalQueryTime / (1000000.0 * calls)).c_str()); ImGui::NextColumn();
			}
			else
				ImGui::NextColumn();
			ImGui::Text(std::to_string((float)(sec->wastedTime) / (calls)).c_str()); ImGui::NextColumn();
			if (l + 1 < 50)
				renderProfiler(l + 1, (int)cur, calls, level, indent + "  ");
		}
	}
}


void renderWindowProfiler() {

	Profile::level* level;
	if (ImGui::Button("Reset Profile"))
		NAU->setProfileResetRequest();
	ImGui::Columns(5, "mycolumns");
	ImGui::Separator();
	ImGui::Text("ID"); ImGui::NextColumn();
	ImGui::Text("#Calls"); ImGui::NextColumn();
	ImGui::Text("CPU"); ImGui::NextColumn();
	ImGui::Text("GPU"); ImGui::NextColumn();
	ImGui::Text("Wasted"); ImGui::NextColumn();
	ImGui::Separator();

	level = Profile::GetProfilerData();
	std::string indent = "";
	renderProfiler(0, -1, level[0].sec[0].calls, level, indent);
}


void updateTree(nau::util::Tree* t) {

	size_t count = t->getElementCount();
	nau::util::Tree* t1;
	std::string s;

	for (unsigned int i = 0; i < count; ++i) {

		t1 = t->getBranch(i);
		if (t1) {
			if (t->getValue(i) == "")
				s = t->getKey(i);
			else
				s = t->getKey(i) + std::string(": ") + t->getValue(i);

			if (ImGui::TreeNode(s.c_str())) {
				updateTree(t1);

				ImGui::TreePop();
			}
		}
		else {
				s = t->getKey(i) + std::string(": ") + t->getValue(i);
				ImGui::Text(s.c_str());
		}
	}
}


void renderWindowProgramInfo() {

	if (ImGui::Button("Refresh")) {
		RENDERER->setPropb(IRenderer::DEBUG_DRAW_CALL, true);
		programInfo = NULL;
	}
	if (programInfo != NULL) {
		std::string s = RENDERMANAGER->getActivePipelineName();
		s = "Pipeline: " + s;
		if (ImGui::TreeNode(s.c_str())) {
			updateTree(programInfo);

			ImGui::TreePop();
		}
	}
}



void renderWindowPassFlow() {

	std::shared_ptr<Pipeline>& pip = RENDERMANAGER->getActivePipeline();
	std::string pipName = pip->getName();
	std::string s = "Pipeline: " + pipName;
	ImGui::Text(s.c_str());

	std::vector<std::string> passes;
	pip->getPassNames(&passes);

	static int passSelected = 0;
	passSelected= RENDERMANAGER->getActivePipeline()->getPassCounter();

	for (int i = 0; i < passes.size(); i++) {

		ImGui::Selectable(passes[i].c_str(), passSelected == i);
	}

	if (passFlowControl_nauPaused) {
		if (ImGui::Button("Resume"))
			passFlowControl_nauPaused = false;

		if (ImGui::Button("Next pass")) {
			passFlowControl_stepPassRequest = true;
			
		}

		if (ImGui::Button("Next frame")) {
			int n = RENDERMANAGER->getActivePipeline()->getPassCounter();
			passFlowControl_framePassRequest = true;
			passFlowControl_framePassStageCompleteFrame = true;
			passFlowControl_framePassEnd = n;
		}
		if (ImGui::Button("Execute pipeline")) {
			passFlowControl_stepTillEndOfPipeline = true;
		}
	}
	else {
		if (ImGui::Button("Pause"))
			passFlowControl_nauPaused = true;

	}

}

#include <fstream> 
struct traceResults {

	int totalOpenGLCalls;
	std::map<std::string, int> functionCountSortdByName;
	nau::util::Tree callLog;

	typedef std::function<bool(std::pair<std::string, int>, std::pair<std::string, int>)> Comparator;

	// Defining a lambda function to compare two pairs. It will compare two pairs using second field
	Comparator compFunctor =
		[](std::pair<std::string, int> elem1, std::pair<std::string, int> elem2)
	{
		if (elem1.second != elem2.second)
			return elem1.second > elem2.second;
		else
			return elem1.first < elem2.first;
	};

	std::set<std::pair<std::string, int>, Comparator> functionSortedByCount;

	void orderByCount() {
		// Declaring a set that will store the pairs using above comparision logic
		functionSortedByCount = std::set<std::pair<std::string, int>, Comparator>(
			functionCountSortdByName.begin(), functionCountSortdByName.end(), compFunctor);
	}

	void clear() {
		totalOpenGLCalls = 0;
		functionCountSortdByName.clear();
		functionSortedByCount.clear();
		callLog.clear();
	}

} traceResult;

void traceShow() {

	if (ImGui::TreeNode("Statistics")) {

		if (ImGui::TreeNode("Ordered by name")) {

			for (auto item : traceResult.functionCountSortdByName) {
				ImGui::Text(item.first.c_str()); ImGui::SameLine(400 - ImGui::GetCursorPos().x, 0);
				ImGui::Text(std::to_string(item.second).c_str());
			}

			ImGui::TreePop();
		}

		if (ImGui::TreeNode("Ordered by call count")) {

			for (auto item : traceResult.functionSortedByCount) {
				ImGui::Text(item.first.c_str()); ImGui::SameLine(400 - ImGui::GetCursorPos().x, 0);
				ImGui::Text(std::to_string(item.second).c_str());
			}
			ImGui::TreePop();
		}
		ImGui::Text("Total GL calls: %d", traceResult.totalOpenGLCalls);

		ImGui::TreePop();
	}
	if (ImGui::TreeNode("Call log")) {

		for (int i = 0; i < traceResult.callLog.getElementCount(); ++i) {

			nau::util::Tree *taux = traceResult.callLog.getBranch(i);
			if (taux) {

				if (ImGui::TreeNode(traceResult.callLog.getKey(i).c_str())) {

					for (int i = 0; i < taux->getElementCount(); ++i) {

						ImGui::Text(taux->getKey(i).c_str());
					}
					ImGui::TreePop();
				}
			}
			else {
				ImGui::Text(traceResult.callLog.getKey(i).c_str());
			}
		}
		ImGui::TreePop();
	}
}



void traceProcess(std::string logfile) {

	std::ifstream filestream;
	filestream.open(logfile);
	string line;

	nau::util::Tree *currNode = &(traceResult.callLog);
	nau::util::Tree* prevNode, *luaNode;

	traceResult.clear();

	if (filestream) {
		while (getline(filestream, line)) {

			if (strcmp(line.substr(0, 4).c_str(), "#NAU") == 0) {
				if (strcmp(line.substr(5, 5).c_str(), "FRAME") == 0) {
					if (strcmp(line.substr(11, 5).c_str(), "START") == 0) {


					}
				}
				else if (strcmp(line.substr(5, 4).c_str(), "PASS") == 0) {
					if (strcmp(line.substr(10, 5).c_str(), "START") == 0) {
						prevNode = currNode;
						currNode = currNode->appendBranch("NAUPASS(" + line.substr(16, line.length() - 17) + ")");
					}
					else if (strcmp(line.substr(10, 3).c_str(), "END") == 0) {
						currNode = prevNode;
					}
				}
			}
			else if (strcmp(line.substr(0, 4).c_str(), "#LUA") == 0) {
				luaNode = currNode->appendBranch("LUASCRIPT(" + line.substr(5, line.length() - 5) + ")");
			}
			else if (strcmp(line.substr(0, 4).c_str(), "LUA:") == 0) {
				luaNode->appendItem(line.substr(5, line.length() - 5), "");
			}
			else {
				currNode->appendItem(line, "");
				std::string aux = line.substr(0, line.find('('));
				traceResult.totalOpenGLCalls++;

				if (traceResult.functionCountSortdByName.count(aux))
					traceResult.functionCountSortdByName[aux]++;
				else
					traceResult.functionCountSortdByName[aux] = 1;

			}
		}
	}

	traceResult.orderByCount();

	filestream.close();
}


void renderWindowTraceLog() {

	static std::string currentFile = "";

	if (ImGui::Button("Trace Frame")) {
		nauInstance->setTrace(1);
	}

	string logfile, filename;
	struct stat fst;
	DIR* dir;
	struct dirent* ent;
	//	time_t tempLastTime;
	std::map<int, std::string> filesToProcess;

	if (ImGui::TreeNode("./__nau3Dtrace/Frame_*.txt")) {

		if ((dir = opendir("./__nau3Dtrace")) != NULL) {
			//m_FileTimes.clear();
			//tempLastTime = m_LastTime;
			//unsigned long long tempLastTime = m_LastTime;
			// Read all files and directories in the directory
			while ((ent = readdir(dir)) != NULL) {

				// Filters files starting with Frame_* only
				if (ent->d_type == S_IFREG && strstr(ent->d_name, "Frame_")) {
					// Corresponding logfile with path
					filename = std::string(ent->d_name);
					logfile = std::string("./__nau3Dtrace/") + filename;

					// if file is at least as recent as project loading
					if (stat(logfile.c_str(), &fst) == 0 && fst.st_mtime >= nauProjectStartTime) {

						if (ImGui::TreeNode(filename.c_str())) {
							if (currentFile != logfile) {
								traceProcess(logfile);
								currentFile = logfile;
							}
							traceShow();

							ImGui::TreePop();
						}
					}
				}
			}
			closedir(dir);
		}
		ImGui::TreePop();
	}
}

#include <nau/render/iGlobalState.h>
#include <nau/loader/stateLoader.h>

void renderOpenGLProperties() {

	if (ImGui::BeginTabBar("GLTabs")) {

		if (ImGui::BeginTabItem("System")) {

			ImGui::Text("GL_VENDOR"); ImGui::SameLine(300 - ImGui::GetCursorPos().x, 0);
			ImGui::Text((char*)glGetString(GL_VENDOR));

			ImGui::Text("GL_RENDERER"); ImGui::SameLine(300 - ImGui::GetCursorPos().x, 0);
			ImGui::Text((char*)glGetString(GL_RENDERER));

			ImGui::Text("GL_VERSION"); ImGui::SameLine(300 - ImGui::GetCursorPos().x, 0);
			ImGui::Text((char*)glGetString(GL_VERSION));

			ImGui::Text("GL_SHADING_LANGUAGE_VERSION"); ImGui::SameLine(300 - ImGui::GetCursorPos().x, 0);
			ImGui::Text((char*)glGetString(GL_SHADING_LANGUAGE_VERSION));

			int n;
			glGetIntegerv(GL_NUM_EXTENSIONS, &n);
			ImGui::Text("Extension count"); ImGui::SameLine(300 - ImGui::GetCursorPos().x, 0);
			ImGui::Text("%d", n);

			ImGui::Dummy(ImVec2(10, 20));

			bool flag = 0;
			if (ImGui::CollapsingHeader("OpenGL Extensions", flag))
			{
				for (int i = 0; i < n; ++i) {

					ImGui::Text((char*)(char*)glGetStringi(GL_EXTENSIONS, i));
				}
			}
			ImGui::EndTabItem();
		}

		if (ImGui::BeginTabItem("OpenGL Limitations")) {

			nau::render::IGlobalState* gs = IGlobalState::Create();
			nau::loader::StateLoader::LoadStateXMLFile("nauSettings\\state.xml", gs);

			std::vector<std::string> enumNames;
			gs->getStateEnumNames(&enumNames);
			std::string value;
			for (std::string enumName : enumNames) {
				value = gs->getState(enumName);
				ImGui::Text(enumName.c_str()); ImGui::SameLine(300 - ImGui::GetCursorPos().x, 0);
				ImGui::Text(value.c_str());
			}
			ImGui::EndTabItem();
		}

		ImGui::EndTabBar();
	}
}



void renderAbout() {

	ImGui::Text("Composer - Nau 3D's GUI (ImGui version)");
	ImGui::Separator();
	ImGui::Text("Nau3D - A 3D engine for OpenGL + Optix, plus Lua scripting");
	ImGui::Text("https://github.com/Nau3D");
	ImGui::Text("http://nau3d.di.uminoh.pt");
}


bool renderMenuRenderPropsWindowChecked = false;
bool renderMenuWireframeChecked = false;
bool renderMenuShowBB = false;
bool renderMenuRenderPassWindowChecked = false;
bool assetsMenuCameraWindowChecked = false;
bool assetsMenuLightWindowChecked = false;
bool assetsMenuViewportWindowChecked = false;
bool assetsMenuScenesWindowChecked = false;
bool materialMenuMatLibWindowChecked = false;
bool materialMenuTextureWindowChecked = false;
bool materialMenuShaderWindowChecked = false;
bool materialMenuAtomicsWindowChecked = false;
bool materialMenuBufferWindowChecked = false;
bool debugMenuProfilerWindowChecked = false;
bool debugMenuLogWindowChecked = false;
bool debugMenuProgramInfoWindowChecked = false;
bool debugMenuPassFlowWindowChecked = false;
bool debugMenuTraceLogindowChecked = false;
bool infoMenuOpenGLPropsWindowChecked = false;
bool infoMenuAboutWindowChecked = false;



void messageBox(const std::string& title, const std::string& message) {

	ImGui::OpenPopup(title.c_str());
	if (ImGui::BeginPopupModal(title.c_str(), NULL, 0)) { //ImGuiWindowFlags_AlwaysAutoResize)) {
		ImGui::TextWrapped(message.c_str());
		if (ImGui::Button("OK", ImVec2(120, 0))) { 
			showMessageBox = false;
			ImGui::CloseCurrentPopup(); 
		}
		ImGui::EndPopup();
	}
}





void renderProjectWindow(const std::string& winName, const nau::inter::ToolBar::Items& winData) {

	for (auto& item : winData) {
		if (item.aClass == nau::inter::ToolBar::PIPELINE_LIST) {

			int p = RENDERMANAGER->getActivePipelineIndex(); 
			int oldP = p;
			if (ImGui::Combo(item.label.c_str(), &p, (char*)&item.options[0])) {
				if (oldP != p) {
					RENDERMANAGER->setActivePipeline(p);
					if (item.luaScript != "")
						NAU->callLuaScript(item.luaScript);
				}
			}
		}
		else if (item.aClass == nau::inter::ToolBar::CUSTOM_ENUM) {
			void *d = (Data *)NAU->getAttributeValue(item.type, item.context, item.component, item.id);
			int i = *(int*)d; int oldi = i;
			if (ImGui::Combo(item.label.c_str(), &i, (char *)&item.options[0])) {
				if (oldi != i) {
					NAU->setAttributeValue(item.type, item.context, item.component, item.id, (Data*)(new NauInt(i)));
					if (item.luaScript != "")
						NAU->callLuaScript(item.luaScript);
				}
			}
		}
		else {
			void* d = (Data*)NAU->getAttributeValue(item.type, item.context, item.component, item.id);

			if (item.dt == Enums::BOOL) {
				bool b = *(bool*)d; bool old_b = b;
				ImGui::Checkbox(item.label.c_str(), &b);
				if (old_b != b) {
					NAU->setAttributeValue(item.type, item.context, item.component, item.id, (Data*)(new NauInt(b)));
					if (item.luaScript != "")
						NAU->callLuaScript(item.luaScript);
				}
			}
			else if (item.dt == Enums::INT) {
				int i = *(int*)d; int oldi = i;
				if (item.min < item.max)
					ImGui::SliderInt(item.label.c_str(), &i, (int)item.min, (int)item.max);
				else
					ImGui::InputScalar(item.label.c_str(), ImGuiDataType_S32, &i, NULL, NULL, "%d");
				if (oldi != i) {
					NAU->setAttributeValue(item.type, item.context, item.component, item.id, (Data *)(new NauInt(i)));
					if (item.luaScript != "")
						NAU->callLuaScript(item.luaScript);
				}
			}
			else if (item.dt == Enums::IVEC2) {
				ivec2 i = *(ivec2*)d; ivec2 oldi = i;
				if (item.min < item.max)
					ImGui::SliderInt2(item.label.c_str(), (int*)i.getPtr(), (int)item.min, (int)item.max);
				else
					ImGui::InputInt2(item.label.c_str(), (int*)i.getPtr());
				if (oldi != i) {
					NAU->setAttributeValue(item.type, item.context, item.component, item.id, (Data*)&i);
					if (item.luaScript != "")
						NAU->callLuaScript(item.luaScript);
				}
			}
			else if (item.dt == Enums::IVEC3) {
				ivec3 i = *(ivec3*)d; ivec3 oldi = i;
				if (item.min < item.max)
					ImGui::SliderInt3(item.label.c_str(), (int*)i.getPtr(), (int)item.min, (int)item.max);
				else
					ImGui::InputInt3(item.label.c_str(), (int*)i.getPtr());
				if (oldi != i) {
					NAU->setAttributeValue(item.type, item.context, item.component, item.id, (Data*)&i);
					if (item.luaScript != "")
						NAU->callLuaScript(item.luaScript);
				}
			}
			else if (item.dt == Enums::IVEC4) {
				ivec4 i = *(ivec4*)d; ivec4 oldi = i;
				if (item.min < item.max)
					ImGui::SliderInt3(item.label.c_str(), (int*)i.getPtr(), (int)item.min, (int)item.max);
				else
					ImGui::InputInt4(item.label.c_str(), (int*)i.getPtr());
				if (oldi != i) {
					NAU->setAttributeValue(item.type, item.context, item.component, item.id, (Data*)&i);
					if (item.luaScript != "")
						NAU->callLuaScript(item.luaScript);
				}
			}
			if (item.dt == Enums::UINT) {
				unsigned int i = *(unsigned int*)d; unsigned int oldi = i;
				if (item.min < item.max) 
					ImGui::SliderScalar(item.label.c_str(), ImGuiDataType_U32, &i, &item.min, &item.max, "%u");				
				else
					ImGui::InputScalar(item.label.c_str(), ImGuiDataType_U32, &i, NULL, NULL, "%u");
				if (oldi != i) {
					NAU->setAttributeValue(item.type, item.context, item.component, item.id, (Data*)(new NauInt(i)));
					if (item.luaScript != "")
						NAU->callLuaScript(item.luaScript);
				}
			}
			else if (item.dt == Enums::UIVEC2) {
				uivec2 i = *(uivec2*)d; uivec2 oldi = i;   
				if (item.min < item.max)
					ImGui::SliderScalarN(item.label.c_str(), ImGuiDataType_U32, (int*)i.getPtr(), 2, &item.min, &item.max);
				else
					ImGui::InputScalarN(item.label.c_str(), ImGuiDataType_U32, (int*)i.getPtr(), 2);
				if (oldi != i) {
					NAU->setAttributeValue(item.type, item.context, item.component, item.id, (Data*)&i);
					if (item.luaScript != "")
						NAU->callLuaScript(item.luaScript);
				}
			}
			else if (item.dt == Enums::UIVEC3) {
				uivec3 i = *(uivec3*)d; uivec3 oldi = i;
				if (item.min < item.max)
					ImGui::SliderScalarN(item.label.c_str(), ImGuiDataType_U32, (int*)i.getPtr(), 3, &item.min, &item.max);
				else
					ImGui::InputScalarN(item.label.c_str(), ImGuiDataType_U32, (int*)i.getPtr(), 3);
				if (oldi != i) {
					NAU->setAttributeValue(item.type, item.context, item.component, item.id, (Data*)&i);
					if (item.luaScript != "")
						NAU->callLuaScript(item.luaScript);
				}
			}
			else if (item.dt == Enums::UIVEC4) {
				uivec4 i = *(uivec4*)d; uivec4 oldi = i;
				if (item.min < item.max)
					ImGui::SliderScalarN(item.label.c_str(), ImGuiDataType_U32, (int*)i.getPtr(), 4, &item.min, &item.max);
				else
					ImGui::InputScalarN(item.label.c_str(), ImGuiDataType_U32, (int*)i.getPtr(), 4);
				if (oldi != i) {
					NAU->setAttributeValue(item.type, item.context, item.component, item.id, (Data*)&i);
					if (item.luaScript != "")
						NAU->callLuaScript(item.luaScript);
				}
			}


			else if (item.dt == Enums::FLOAT) {
				float f = *(float*)d; float oldf = f;
				if (item.min < item.max)
					ImGui::SliderFloat(item.label.c_str(), &f, item.min, item.max);
				else
					ImGui::InputScalar(item.label.c_str(), ImGuiDataType_Float, &f, NULL, NULL);
				if (oldf != f) {
					NAU->setAttributeValue(item.type, item.context, item.component, item.id, (Data*)(new NauFloat(f)));
					if (item.luaScript != "")
						NAU->callLuaScript(item.luaScript);
				}
			}
			else if (item.dt == Enums::VEC2) {
				vec2 i = *(vec2*)d; vec2 oldi = i;
				if (item.min < item.max)
					ImGui::SliderFloat2(item.label.c_str(), (float*)i.getPtr(), item.min, item.max);
				else
					ImGui::InputFloat2(item.label.c_str(), (float*)i.getPtr());
				if (oldi != i) {
					NAU->setAttributeValue(item.type, item.context, item.component, item.id, (Data*)&i);
					if (item.luaScript != "")
						NAU->callLuaScript(item.luaScript);
				}
			}
			else if (item.dt == Enums::VEC3) {
				vec3 i = *(vec3*)d; vec3 oldi = i;
				if (item.min < item.max)
					ImGui::SliderFloat3(item.label.c_str(), (float*)i.getPtr(), item.min, item.max);
				else
					ImGui::InputFloat3(item.label.c_str(), (float*)i.getPtr());
				if (oldi != i) {
					NAU->setAttributeValue(item.type, item.context, item.component, item.id, (Data*)&i);
					if (item.luaScript != "")
						NAU->callLuaScript(item.luaScript);
				}
			}
			else if (item.dt == Enums::VEC4) {
				vec4 i = *(vec4*)d; vec4 oldi = i;
				if (item.semantics == nau::Attribute::Semantics::COLOR) {
					if (ImGui::ColorEdit4(item.label.c_str(), (float*)i.getPtr(), ImGuiColorEditFlags_Float)) {
						if (oldi != i) {
							NAU->setAttributeValue(item.type, item.context, item.component, item.id, (Data*)&i);
							if (item.luaScript != "")
								NAU->callLuaScript(item.luaScript);
						}
					}

				}
				else {
					if (item.min < item.max)
						ImGui::SliderFloat4(item.label.c_str(), (float*)i.getPtr(), item.min, item.max);
					else
						ImGui::InputFloat4(item.label.c_str(), (float*)i.getPtr());
					if (oldi != i) {
						NAU->setAttributeValue(item.type, item.context, item.component, item.id, (Data*)&i);
						if (item.luaScript != "")
							NAU->callLuaScript(item.luaScript);
					}
				}
			}

			if (item.dt == Enums::DOUBLE) {
				double i = *(double*)d; double oldi = i;
				if (item.min < item.max)
					ImGui::SliderScalar(item.label.c_str(), ImGuiDataType_Double, &i, &item.min, &item.max, "%f");
				else
					ImGui::InputScalar(item.label.c_str(), ImGuiDataType_Double, &i, NULL, NULL, "%f");
				if (oldi != i) {
					NAU->setAttributeValue(item.type, item.context, item.component, item.id, (Data*)(new NauDouble(i)));
					if (item.luaScript != "")
						NAU->callLuaScript(item.luaScript);
				}
			}
			else if (item.dt == Enums::DVEC2) {
				dvec2 i = *(dvec2*)d; dvec2 oldi = i;
				if (item.min < item.max)
					ImGui::SliderScalarN(item.label.c_str(), ImGuiDataType_Double, (int*)i.getPtr(), 2, &item.min, &item.max);
				else
					ImGui::InputScalarN(item.label.c_str(), ImGuiDataType_Double, (int*)i.getPtr(), 2);
				if (oldi != i) {
					NAU->setAttributeValue(item.type, item.context, item.component, item.id, (Data*)&i);
					if (item.luaScript != "")
						NAU->callLuaScript(item.luaScript);
				}
			}
			else if (item.dt == Enums::DVEC3) {
				dvec3 i = *(dvec3*)d; dvec3 oldi = i;
				if (item.min < item.max)
					ImGui::SliderScalarN(item.label.c_str(), ImGuiDataType_Double, (int*)i.getPtr(), 3, &item.min, &item.max);
				else
					ImGui::InputScalarN(item.label.c_str(), ImGuiDataType_Double, (int*)i.getPtr(), 3);
				if (oldi != i) {
					NAU->setAttributeValue(item.type, item.context, item.component, item.id, (Data*)&i);
					if (item.luaScript != "")
						NAU->callLuaScript(item.luaScript);
				}
			}
			else if (item.dt == Enums::DVEC4) {
				uivec4 i = *(uivec4*)d; uivec4 oldi = i;
				if (item.min < item.max)
					ImGui::SliderScalarN(item.label.c_str(), ImGuiDataType_Double, (int*)i.getPtr(), 4, &item.min, &item.max);
				else
					ImGui::InputScalarN(item.label.c_str(), ImGuiDataType_Double, (int*)i.getPtr(), 4);
				if (oldi != i) {
					NAU->setAttributeValue(item.type, item.context, item.component, item.id, (Data*)&i);
					if (item.luaScript != "")
						NAU->callLuaScript(item.luaScript);
				}
			}



		}

	}

}


void renderGUI(ImGuiIO& io) {

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	
	//ImGui::ShowMetricsWindow();

	if (ImGui::BeginMainMenuBar()) {
		if (ImGui::BeginMenu("File")) {
			if (ImGui::MenuItem("Open Project", "CTRL+O")) {
				ImGuiFileDialog::Instance()->OpenDialog("ChooseProjFileDlgKey", "Choose Project File", ".xml\0.*", ".");
			}
			//if (ImGui::MenuItem("Open Folder", "CTRL+F")) {}
			if (ImGui::MenuItem("Open Model", "CTRL+M")) {
				ImGuiFileDialog::Instance()->OpenDialog("ChooseModelFileDlgKey", "Choose 3D Model File", ".obj\0.3ds\0.nbo\0.dae\0.xml\0.blend\0.fbx\0.ply\0.lwo\0.stl\0.cob\0.scn\0.*", ".");
			}
			//if (ImGui::MenuItem("Save Project (WIP)")) {}
			//if (ImGui::MenuItem("Process Folder", "CTRL+P")) {}
			if (ImGui::MenuItem("Quit", "ALT+F4")) {
				glfwSetWindowShouldClose(window, true);
			}
			ImGui::EndMenu();
		}
		if (nauLoaded) {
			if (ImGui::BeginMenu("Render")) {
				if (ImGui::MenuItem("Renderer Props", "CTRL+R", renderMenuRenderPropsWindowChecked)) {
					renderMenuRenderPropsWindowChecked = !renderMenuRenderPropsWindowChecked;
				}
				if (ImGui::MenuItem("Pass Library", "F2", renderMenuRenderPassWindowChecked)) {
					renderMenuRenderPassWindowChecked = !renderMenuRenderPassWindowChecked;
				}
				if (ImGui::MenuItem("Reset Frame Count", "B")) {
					RENDERER->setPropui(IRenderer::FRAME_COUNT, 0);
				}
				if (ImGui::MenuItem("Wireframe", "CTRL+W", renderMenuWireframeChecked)) {
					renderMenuWireframeChecked = !renderMenuWireframeChecked;
					if (renderMenuWireframeChecked)
						RENDERMANAGER->setRenderMode(nau::render::IRenderer::WIREFRAME_MODE);
					else
						RENDERMANAGER->setRenderMode(nau::render::IRenderer::MATERIAL_MODE);
				}
				if (ImGui::MenuItem("Show Bounding Boxes", "CTRL+B", renderMenuShowBB)) {
					renderMenuShowBB = !renderMenuShowBB;
					NAU->setRenderFlag(nau::Nau::BOUNDING_BOX_RENDER_FLAG, renderMenuShowBB);
				}
				if (ImGui::MenuItem("Recompile Lua Scripts", "CTRL+B")) {
					NAU->compileLuaScripts();
				}
				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu("Assets")) {
				if (ImGui::MenuItem("Camera Library", "F3", assetsMenuCameraWindowChecked)) {
					assetsMenuCameraWindowChecked = !assetsMenuCameraWindowChecked;
				}
				if (ImGui::MenuItem("Light Library", "F4", assetsMenuLightWindowChecked)) {
					assetsMenuLightWindowChecked = !assetsMenuLightWindowChecked;
				}
				if (ImGui::MenuItem("Viewport Library", "F5", assetsMenuViewportWindowChecked)) {
					assetsMenuViewportWindowChecked = !assetsMenuViewportWindowChecked;
				}
				if (ImGui::MenuItem("Scene Library", "F6", assetsMenuScenesWindowChecked)) {
					assetsMenuScenesWindowChecked = !assetsMenuScenesWindowChecked;
				}
				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu("Materials")) {
				if (ImGui::MenuItem("Material Library", "F7", materialMenuMatLibWindowChecked)) {
					materialMenuMatLibWindowChecked = !materialMenuMatLibWindowChecked;
				}
				if (ImGui::MenuItem("Texture Library", "F8", materialMenuTextureWindowChecked)) {
					materialMenuTextureWindowChecked = !materialMenuTextureWindowChecked;
				}
				if (ImGui::MenuItem("Shader Library", "F9", materialMenuShaderWindowChecked)) {
					materialMenuShaderWindowChecked = !materialMenuShaderWindowChecked;
				}
				if (ImGui::MenuItem("Atomics Library", "F10", materialMenuAtomicsWindowChecked)) {
					materialMenuAtomicsWindowChecked = !materialMenuAtomicsWindowChecked;
				}
				if (ImGui::MenuItem("Buffer Library", "F11", materialMenuBufferWindowChecked)) {
					materialMenuBufferWindowChecked = !materialMenuBufferWindowChecked;
				}
				ImGui::EndMenu();
			}
			if (ImGui::BeginMenu("Debug")) {
				if (ImGui::MenuItem("Profiler", "CTRL+P", debugMenuProfilerWindowChecked)) {
					debugMenuProfilerWindowChecked = !debugMenuProfilerWindowChecked;
				}
				if (ImGui::MenuItem("Log", "CTRL+P", debugMenuLogWindowChecked)) {
					debugMenuLogWindowChecked = !debugMenuLogWindowChecked;
				}
				if (ImGui::MenuItem("Program Info", "CTRL+F1", debugMenuProgramInfoWindowChecked)) {
					debugMenuProgramInfoWindowChecked = !debugMenuProgramInfoWindowChecked;
				}
				if (ImGui::MenuItem("Pass Flow Control", "", debugMenuPassFlowWindowChecked)) {
					debugMenuPassFlowWindowChecked = !debugMenuPassFlowWindowChecked;
				}
				if (ImGui::MenuItem("Trace Log", "", debugMenuTraceLogindowChecked)) {
					debugMenuTraceLogindowChecked = !debugMenuTraceLogindowChecked;
				}
				if (ImGui::MenuItem("Screen Shot", "")) {
					RENDERER->saveScreenShot();;
				}
				ImGui::EndMenu();
			}

			const std::map<std::string, nau::inter::ToolBar::Items>& projWindows = INTERFACE_MANAGER->getWindows();
			if (projWindows.size() > 0) {

				if (projectWindows.size() == 0) {
					for (auto &proj:projWindows) {
						projectWindows[proj.first] = true;
					}
				}

				if (ImGui::BeginMenu("Project")) {

					for (auto window : projWindows) {

						if (ImGui::MenuItem(window.first.c_str(), "", projectWindows[window.first])) {
							projectWindows[window.first] = !projectWindows[window.first];
						}
					}
					ImGui::EndMenu();
				}
			}


		}			
		if (ImGui::BeginMenu("Info")) {
			if (ImGui::MenuItem("OpenGL Properties", "", infoMenuOpenGLPropsWindowChecked)) {
				infoMenuOpenGLPropsWindowChecked = !infoMenuOpenGLPropsWindowChecked;
			}
			if (ImGui::MenuItem("About", "", infoMenuAboutWindowChecked)) {
				infoMenuAboutWindowChecked = !infoMenuAboutWindowChecked;
			}
			ImGui::EndMenu();

		}

		ImGui::EndMainMenuBar();

	}

	// -----------------------------------------------------------------------
	// file dialog boxes

	// Project
	const std::map<std::string, nau::inter::ToolBar::Items>& projWindows = INTERFACE_MANAGER->getWindows();
	for (auto& win : projWindows) {

		if (projectWindows[win.first]) {


			if (ImGui::Begin(win.first.c_str(), &projectWindows[win.first]))
				renderProjectWindow(win.first, win.second);
			ImGui::End();

			
		}
	}



	if (ImGuiFileDialog::Instance()->FileDialog("ChooseProjFileDlgKey"))
	{
		// action if OK
		if (ImGuiFileDialog::Instance()->IsOk == true)
		{
			std::string filePathName = ImGuiFileDialog::Instance()->GetFinalFileName();
			std::string filePath = ImGuiFileDialog::Instance()->GetCurrentPath();
			printf("%s %s\n", filePathName.c_str(), filePath.c_str());
			time(&nauProjectStartTime);
			traceResult.clear();
			loadProject(filePathName.c_str());
			nauLoaded = true;

		}
		ImGuiFileDialog::Instance()->CloseDialog("ChooseProjFileDlgKey");
	}

	// Model

	if (ImGuiFileDialog::Instance()->FileDialog("ChooseModelFileDlgKey"))
	{
		// action if OK
		if (ImGuiFileDialog::Instance()->IsOk == true)
		{
			std::string filePathName = ImGuiFileDialog::Instance()->GetFinalFileName();
			std::string filePath = ImGuiFileDialog::Instance()->GetCurrentPath();
			printf("%s %s\n", filePathName.c_str(), filePath.c_str());
			time(&nauProjectStartTime);
			traceResult.clear();
			loadModel(filePathName.c_str());
			nauLoaded = true;
		}
		ImGuiFileDialog::Instance()->CloseDialog("ChooseModelFileDlgKey");
	}


	if (renderMenuRenderPropsWindowChecked) {
		if (ImGui::Begin("NAU - Render Props", &renderMenuRenderPropsWindowChecked)) {
			std::vector<std::string> order = {};
			createOrderedGrid(IRenderer::GetAttribs(), (AttributeValues*)RENDERER, order);
		}
		ImGui::End();
	}

	if (renderMenuRenderPassWindowChecked) {
		if (ImGui::Begin("Nau - Pass", &renderMenuRenderPassWindowChecked)) {
			renderWindowPass();
		}
		ImGui::End();
	}

	if (assetsMenuCameraWindowChecked) {
		if (ImGui::Begin("Nau - Cameras", &assetsMenuCameraWindowChecked))
			renderWindowCameras();
		ImGui::End();
	}

	if (assetsMenuLightWindowChecked) {
		if (ImGui::Begin("Nau - Lights", &assetsMenuLightWindowChecked))
			renderWindowLights();
		ImGui::End();
	}

	if (assetsMenuViewportWindowChecked) {
		if (ImGui::Begin("Nau - Viewports", &assetsMenuViewportWindowChecked))
			renderWindowViewports();
		ImGui::End();
	}

	if (assetsMenuScenesWindowChecked) {
		if (ImGui::Begin("Nau - Scenes", &assetsMenuScenesWindowChecked))
			renderWindowScenes();
		ImGui::End();
	}

	if (materialMenuMatLibWindowChecked) {
		if (ImGui::Begin("Nau - Material Library", &materialMenuMatLibWindowChecked))
			renderWindowMaterialLibrary();
		ImGui::End();
	}

	if (materialMenuTextureWindowChecked) {
		if (ImGui::Begin("Nau - Texture Library", &materialMenuTextureWindowChecked))
			renderWindowTextureLibrary();
		ImGui::End();
	}

	if (materialMenuShaderWindowChecked) {
		if (ImGui::Begin("Nau - Shader Library", &materialMenuShaderWindowChecked))
			renderWindowShaderLibrary();
		ImGui::End();
	}

	if (materialMenuAtomicsWindowChecked) {
		if (ImGui::Begin("Nau - Atomics Library", &materialMenuAtomicsWindowChecked))
			renderWindowAtomics();
		ImGui::End();
	}

	if (materialMenuBufferWindowChecked) {
		if (ImGui::Begin("Nau - Buffer Library", &materialMenuBufferWindowChecked))
			renderWindowBufferLibrary();
		ImGui::End();
	}

	if (debugMenuProfilerWindowChecked) {
		if (ImGui::Begin("Nau - Profiler", &debugMenuProfilerWindowChecked))
			renderWindowProfiler();
		ImGui::End();
	}

	if (debugMenuLogWindowChecked) {
		console.Draw("Nau - Log", &debugMenuLogWindowChecked);
	}

	if (debugMenuProgramInfoWindowChecked) {
		if (ImGui::Begin("Nau - Program Info", &debugMenuProgramInfoWindowChecked))
			renderWindowProgramInfo();
		ImGui::End();
	}

	if (debugMenuPassFlowWindowChecked) {
		if (ImGui::Begin("Nau - Pass FLow Control", &debugMenuPassFlowWindowChecked))
			renderWindowPassFlow();
		ImGui::End();
	}

	if (debugMenuTraceLogindowChecked) {
		if (ImGui::Begin("Nau - Trace Log", &debugMenuTraceLogindowChecked))
			renderWindowTraceLog();
		ImGui::End();
	}

	if (infoMenuOpenGLPropsWindowChecked) {
		if (ImGui::Begin("Nau - OpenGL Properties", &infoMenuOpenGLPropsWindowChecked))
			renderOpenGLProperties();
		ImGui::End();
	}

	if (infoMenuAboutWindowChecked) {
		if (ImGui::Begin("Nau - About", &infoMenuAboutWindowChecked))
			renderAbout();
		ImGui::End();
	}

	if (showMessageBox)
		messageBox(messageBoxTitle, messageBoxMessage);

	//ImGui::End();
	ImGui::Render();
	int display_w, display_h;
	glfwGetFramebufferSize(window, &display_w, &display_h);
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
		GLFWwindow* backup_current_context = glfwGetCurrentContext();
		ImGui::UpdatePlatformWindows();
		ImGui::RenderPlatformWindowsDefault();
		glfwMakeContextCurrent(backup_current_context);
	}

}




// ------------------------------------------------------------
//
// Main function
//


static void error_callback(int error, const char* description) {

	fprintf(stderr, "Error: %s\n", description);
}



int main(int argc, char **argv) {


	//if (argc == 1) {
	//	printf("The application requires the name of a project as a command line parameter\n");
	//	return 1;
	//}

	int width = 640, height = 360;
	glfwSetErrorCallback(error_callback);

	if (!glfwInit())
		exit(EXIT_FAILURE);

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
	window = glfwCreateWindow(width, height, "Nau3D", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}


//	Mouse and Keyboard Callbacks
	glfwSetMouseButtonCallback(window, processMouseButtons);
	glfwSetCursorPosCallback(window, processMouseMotion);
	glfwSetFramebufferSizeCallback(window, changeSize);
	//glfwSetCharCallback(window, processKeys);
	glfwSetKeyCallback(window, processKeys);


	glfwMakeContextCurrent(window);
	glfwSwapInterval(0);

	glbinding::Binding::initialize(false);
	// Display some general info
	printf("Vendor: %s\n", glGetString(GL_VENDOR));
	printf("Renderer: %s\n", glGetString(GL_RENDERER));
	printf("Version: %s\n", glGetString(GL_VERSION));
	printf("GLSL: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
	int param;
	glGetIntegerv(GL_CONTEXT_PROFILE_MASK, (int*)&param);
	if (param == (int)GL_CONTEXT_CORE_PROFILE_BIT)
		printf("Context Profile: Core\n");
	else
		printf("Context Profile: Compatibility\n");

	//  GLUT main loop
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;           // Enable Docking
	io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;         // Enable Multi-Viewport / Platform Windows

	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls
	ImGuiStyle& style = ImGui::GetStyle();
	if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
	{
		style.WindowRounding = 0.0f;
		style.Colors[ImGuiCol_WindowBg].w = 1.0f;
	}


	// Setup Dear ImGui style
	//ImGui::StyleColorsDark();
	ImGui::StyleColorsClassic();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	const char* glsl_version = "#version 460";
	ImGui_ImplOpenGL3_Init(glsl_version);
	renderGUI(io);

	// Setup Platform/Renderer bindings

	std::string s;

	nauInstance = (Nau*)Nau::GetInstance();
	int w = 640, h = 360;

	try {

		//printf("%s\n%s\n%s<n", appPath.c_str(),cleanAppPath.c_str(), full.c_str());
		nauInstance->init(false);
		nauInstance->setWindowSize(w, h);

		if (argc > 1) {
			s = std::string(argv[1]);
			std::string appPath = nau::system::File::GetAppFolder();
			std::string cleanAppPath = nau::system::File::CleanFullPath(appPath);
			std::string full = nau::system::File::GetFullPath(appPath, s);
			time(&nauProjectStartTime);
			traceResult.clear();
			nauInstance->readProjectFile(full, &w, &h);
			nauLoaded = true;

			if (h != 0) {
				glfwSetWindowSize(window, w, h);
				changeSize(window, w, h);
			}
		}
	}
	catch (std::string s) {
		showMessageBox = true;
		messageBoxTitle = "Project loader message";
		messageBoxMessage = s;
		printf("%s\n", s.c_str());
		//exit(0);
	}



	// entrar no ciclo do GLUT 
	while (!glfwWindowShouldClose(window)) {
		{
			glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
			PROFILE("Frame");
			if (!passFlowControl_nauPaused)
				renderScene();
			else {
				if (passFlowControl_stepPassRequest) {
					nauInstance->stepPass();
					passFlowControl_stepPassRequest = false;
				}
				else if (passFlowControl_framePassRequest) {
					if (passFlowControl_framePassStageCompleteFrame) {
						nauInstance->stepCompleteFrame();
						passFlowControl_framePassStageCompleteFrame = false;
					}
					else {
						nauInstance->stepPasses(passFlowControl_framePassEnd);
						passFlowControl_framePassRequest = false;
					}
				}
			}
		}
		if (RENDERER->getPropb(IRenderer::DEBUG_DRAW_CALL)) {
			RENDERER->setPropb(IRenderer::DEBUG_DRAW_CALL, false);
			EVENTMANAGER->notifyEvent("SHADER_DEBUG_INFO_AVAILABLE", "Renderer", "", NULL);
		}
		
		renderGUI(io);

		glfwSwapBuffers(window);

#if NAU_PROFILE == NAU_PROFILE_CPU_AND_GPU
		Profile::CollectQueryResults();
#endif
		
		glfwPollEvents();
		// must be called outside profiling area
		if (nauInstance->getProfileResetRequest())
			Profile::Reset();

	}

	glfwDestroyWindow(window);

	glfwTerminate();
	exit(EXIT_SUCCESS);
}


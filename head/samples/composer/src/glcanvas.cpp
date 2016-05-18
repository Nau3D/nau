#include <glcanvas.h>

#include <main.h>


#include <nau.h>
#include <nau/slogger.h>
#include <nau/debug/profile.h>
#include <nau/event/eventFactory.h>
#include <nau/event/cameraMotion.h>
#include <nau/event/cameraOrientation.h>
#include <nau/geometry/iBoundingVolume.h>
#include <nau/math/utils.h>
#include <nau/render/iAPISupport.h>
#include <nau/scene/sceneObject.h>




//#ifdef GLINTERCEPTDEBUG
//#include "..\..\GLIntercept\Src\MainLib\ConfigDataExport.h"
//#endif

#include <iostream>

using namespace std;
using namespace nau::math;
using namespace nau::scene;
using namespace nau::geometry;

#define VELOCITY 50.0f
#define NON_DYNAMIC_VELOCITY 0.05f  // 2 meters per second

bool isPaused = false;

GLCanvas::GLCanvas (wxWindow *parent,
                     const wxWindowID id,
					 const int * 	attribList,
					 const int * contextAttribs,
                     const wxPoint &pos,
                     const wxSize &size,
                     long style,
                     const wxString &name)
   : wxGLCanvas (parent, id, attribList, contextAttribs, pos, size, style, name), 
	init (false), m_pEngine (0), m_pCamera (0), m_CounterFps (0),
	m_Alpha (3.14159f), m_AlphaAux (0.0f), m_Beta (0.0f), m_BetaAux (0.0f), m_OldX(0), m_OldY(0),
	m_tracking(0), /*m_RiverAnimation (0), */m_WaterState (true), m_Stereo (false)
{
	init = true;
	m_Timer.Start();
	p_GLC = new wxGLContext(this);
	SetCurrent(*p_GLC);
	step = 0;
}


GLCanvas::GLCanvas (wxWindow *parent,
                     const GLCanvas &other,
                     const wxWindowID id,
					 const int * 	attribList,
					 const int * contextAttribs,
                     const wxPoint &pos,
                     const wxSize &size,
                     long style,
                     const wxString &name)
   : wxGLCanvas (parent, id, attribList , contextAttribs, pos, size, style, name), init (false)
{
	p_GLC = new wxGLContext(this);
	SetCurrent(*p_GLC);
	step = 0;
}


GLCanvas::~GLCanvas() {

	delete 	p_GLC;
}

void
GLCanvas::setEngine (nau::Nau* engine) {

	m_pEngine = engine;
}


void 
GLCanvas::setCamera () {

	m_pCamera = NAU->getActiveCamera();
	if (m_pCamera) {
		vec4 v = m_pCamera->getPropf4(Camera::VIEW_VEC);
		m_Beta = asin(v.y);
		m_Alpha = atan2(v.x,v.z);
	}
}


void
GLCanvas::_setCamera() {

	m_pCamera = NAU->getActiveCamera();
}


void 
GLCanvas::OnPaint (wxPaintEvent &event) {

	if (!m_pEngine)
		return;

	{
		PROFILE("Nau");
		wxPaintDC dc(this);


		if (!isPaused) {
			this->Render();
			if (APISupport->apiSupport(IAPISupport::BUFFER_ATOMICS))
				DlgAtomics::Instance()->update();
		}
		//	if(!isPaused){
		//
		//		Render();
		//
		//		if (step != 0){
		//			DlgTrace::Instance()->loadLog();
		//
		//#ifdef GLINTERCEPTDEBUG
		//			gliSetIsGLIActive(false);
		//			if (gliIsLogPerFrame()){
		//				DlgTrace::Instance()->clear();
		//			}
		//
		//			DlgDbgStep::Instance()->updateDlg();
		//#endif
		//			step = 0;
		//		}
		//	}
		event.Skip();
	}
	if (m_pEngine && m_pEngine->getProfileResetRequest())
		Profile::Reset();
}


void
GLCanvas::OnIdle(wxIdleEvent& event) {

	//Refresh();
	if (!m_pEngine)
		return;
	{
		PROFILE("Nau");

		if (!isPaused) {
			this->Render();
			if (APISupport->apiSupport(IAPISupport::BUFFER_ATOMICS))
				DlgAtomics::Instance()->update();
		}
		event.RequestMore();
	}
	// moved to nau
	if (m_pEngine->getProfileResetRequest())
		Profile::Reset();
}


void 
GLCanvas::OnSize (wxSizeEvent &event) {

	int width;
	int height;

	GetClientSize (&width, &height);

	if (0 != m_pEngine) {
		std::shared_ptr<nau::event_::IEventData> e3 = nau::event_::EventFactory::Create("Vec3");
		vec3 *v = new vec3(width, height, 0);
		e3->setData(v);
		delete v;
		EVENTMANAGER->notifyEvent("WINDOW_SIZE_CHANGED","Canvas","",e3);

	}
}


void 
GLCanvas::OnEraseBackground (wxEraseEvent &event) {

   event.Skip ();
}


void 
GLCanvas::OnEnterWindow (wxMouseEvent &event) {

   SetFocus ();
   event.Skip ();
}


void 
GLCanvas::Render () {

	PROFILE ("Main cicle");
	try {
		if (0 != m_pEngine) {
			m_pEngine->step();
		}
	}
	catch (std::string s) {
			wxMessageBox(wxString (s.c_str()));
			exit (1);
	}

	{
		PROFILE ("Zi swap");
		SwapBuffers();
	}
	Profile::CollectQueryResults();

	//  FPS Counter
	m_CounterFps++;

	long t = m_Timer.Time();
	if (t > 1000) {
		m_Timer.Start();
		char fps[256];

		vec4 v;
		if (m_pCamera)
			v = m_pCamera->getPropf4(Camera::VIEW_VEC);

		sprintf(fps, "%s (FPS: %d - Triangles: %u)", NAU->getProjectName().c_str(), m_CounterFps, RENDERER->getCounter(nau::render::IRenderer::TRIANGLE_COUNTER));
		wxTopLevelWindow *tlw = (wxTopLevelWindow *)(this->GetParent());

		tlw->SetTitle(wxString::FromAscii(fps));

		m_CounterFps = 0;
	}
	if (RENDERER->getPropb(IRenderer::DEBUG_DRAW_CALL)) {
		RENDERER->setPropb(IRenderer::DEBUG_DRAW_CALL, false);
		EVENTMANAGER->notifyEvent("SHADER_DEBUG_INFO_AVAILABLE", "Renderer", "", NULL);
	}
}


void 
GLCanvas::OnKeyUp(wxKeyEvent & event) {

	//if (m_pCamera && true == m_pCamera->isDynamic()) {
	//	vec3 v3;
	//	v3.set(0.0f, 0.0f, 0.0f);
	//	m_pEngine->getWorld().setVelocity ("testCamera", v3);
	//}
}


void 
GLCanvas::OnKeyDown(wxKeyEvent & event) {

	int mod = 0;
	if (true == event.ShiftDown()) 
		mod |= Nau::KEY_MOD_SHIFT;
	if (true == event.AltDown())
		mod |= Nau::KEY_MOD_ALT;
	if (true == event.ControlDown())
		mod |= Nau::KEY_MOD_CTRL;

	if (m_pEngine->keyPressed(event.GetKeyCode(), mod))
		return;


	static bool physics = true;

	_setCamera();
	if (0 == m_pCamera) {
		return;
	}


	vec4 camPosition = m_pCamera->getPropf4(Camera::POSITION);
	vec4 camUp = m_pCamera->getPropf4(Camera::UP_VEC);
	vec4 camView = m_pCamera->getPropf4(Camera::VIEW_VEC);

	//if ('K' == event.GetKeyCode()) {
	//	m_pEngine->sendKeyToEngine (event.GetKeyCode());	
	//}
	//if ('T' == event.GetKeyCode()) {
	//	if (true == event.ShiftDown())
	//		m_pEngine->setTrace(1);
	//	else
	//		m_pEngine->setTrace(-1);
	//}
	//if ('Y' == event.GetKeyCode()) {
	//	m_pEngine->setTrace(0);
	//}
	//if ('I' == event.GetKeyCode()) {
	//	m_pEngine->sendKeyToEngine('I');
	//}
	if ('M' == event.GetKeyCode()) {
		EVENTMANAGER->notifyEvent("NEXT_POSE", "MainCanvas", "", NULL);
	}

	//if ('9' >= event.GetKeyCode() && '0' <= event.GetKeyCode()) {
	//	//m_pEngine->sendKeyToEngine (event.GetKeyCode());
	//	Camera *aCam = RENDERMANAGER->getCamera ("MainCamera");

	//	vec3 v;

	//	switch (event.GetKeyCode()) {
	//		case '1':	
	//			v.set (-14.486f * cos (DegToRad (-137.0)) - 59.256 * -sin (DegToRad (-137.0)), 
	//				13.266f, 
	//				-59.256 * cos (DegToRad(-137.0)) + -14.486f * sin (DegToRad (-137.0)));

	//			aCam->setCamera (vec3 (-14.486f, 13.266f, -59.256f), v, vec3 (0.0f, 1.0f, 0.0f));
	//			break;
	//		case '2':
	//			v.set (0.0f, 0.0f, -1.0f * cos (DegToRad (-141.4f)));

	//			aCam->setCamera (vec3 (7.930f, 16.135f, -38.392f), v, vec3 (0.0f, 1.0f, 0.0f));
	//			break;

	//		case '3':
	//			v.set (0.0f, 0.0f, -1.0f * cos (DegToRad (-81.8f)));

	//			aCam->setCamera (vec3 (7.374f, 14.465f, -58.637f), v, vec3 (0.0f, 1.0f, 0.0f));
	//			break;

	//		case '4':
	//			v.set (0.0f, 0.0f, -1.0f * cos (DegToRad (-17.0f)));

	//			aCam->setCamera (vec3 (13.363f, 13.977f, -47.436f), v, vec3 (0.0f, 1.0f, 0.0f));
	//			break;

	//		case '5':
	//			v.set (0.0f, 0.0f, -1.0f * cos (DegToRad (135.58f)));

	//			aCam->setCamera (vec3 (-131.176f, 9.555f, 188.927f), v, vec3 (0.0f, 1.0f, 0.0f));
	//			break;
	//	}

	//}

	//if ('0' == event.GetKeyCode()) {
	//	RENDERER->saveScreenShot();
	//}

	//if ('B' == event.GetKeyCode()) {
	//	m_pEngine->resetFrameCount();
	//}

	float direction;
	if (true == m_pCamera->isDynamic()) {
		direction = VELOCITY;
	}
	else {
		direction = NON_DYNAMIC_VELOCITY;
	}

	if (true == event.ShiftDown()) {	// SHIFT = fast motion
		direction *= 10.0f;
	}
	else if (true == event.ControlDown()) { // CTRL = very fast motion. note: shift has precedence over control!
		direction *= 100.0f;
	}

	if ('S' == event.GetKeyCode()) {
		//if (true == m_pCamera->isDynamic()) {
		//	vec4 vel (camView);

		//	vel.y = 0.0f;
		//	vel.normalize();
		//	vel *= -direction;
		//	vec3 v3;
		//	v3.set(vel.x,vel.y,vel.z);
		//	m_pEngine->getWorld().setVelocity ("testCamera", v3);
		//} 
		//else
		{
			nau::event_::CameraMotion c("BACKWARD", direction);
			std::shared_ptr<IEventData> e= nau::event_::EventFactory::Create("Camera Motion");
			e->setData(&c);
			EVENTMANAGER->notifyEvent("CAMERA_MOTION", "MainCanvas", "", e);
		}
		DlgCameras::Instance()->updateInfo(m_pCamera->getName());
	}

	if ('W' == event.GetKeyCode()) {
		//if (true == m_pCamera->isDynamic()) {
		//	vec4 vel (camView);
		//	
		//	vel.y = 0.0f;
		//	vel.normalize();
		//	vel *= direction; 
		//	
		//	vec3 v3;
		//	v3.set(vel.x,vel.y,vel.z);
		//	m_pEngine->getWorld().setVelocity ("testCamera", v3);
		//} 
		//else 
		{

			nau::event_::CameraMotion c("FORWARD", direction);
			std::shared_ptr<IEventData> e = nau::event_::EventFactory::Create("Camera Motion");
			e->setData(&c);
			EVENTMANAGER->notifyEvent("CAMERA_MOTION", "MainCanvas", "", e);
		}
		DlgCameras::Instance()->updateInfo(m_pCamera->getName());
	}

	if ('A' == event.GetKeyCode()){
		//if (true == m_pCamera->isDynamic()) {
		//	vec4 vel (camView.cross (camUp));

		//	vel *= -direction;
		//	vel.y = 0.0f;
		//	vec3 v3;
		//	v3.set(vel.x,vel.y,vel.z);
		//	m_pEngine->getWorld().setVelocity ("testCamera", v3);

		//} 
		//else 
		{

			nau::event_::CameraMotion c("LEFT", direction);
			std::shared_ptr<IEventData> e = nau::event_::EventFactory::Create("Camera Motion");
			e->setData(&c);
			EVENTMANAGER->notifyEvent("CAMERA_MOTION", "MainCanvas", "", e);
		}
		DlgCameras::Instance()->updateInfo(m_pCamera->getName());

	}
	if ('D' == event.GetKeyCode()){
		//if (true == m_pCamera->isDynamic()) {
		//	vec4 vel (camView.cross (camUp));

		//	vel *= direction;
		//	vel.y = 0.0f;
		//	vec3 v3;
		//	v3.set(vel.x,vel.y,vel.z);
		//	m_pEngine->getWorld().setVelocity ("testCamera", v3);
		//} 
		//else 
		{
			nau::event_::CameraMotion c("RIGHT", direction);
			std::shared_ptr<IEventData> e = nau::event_::EventFactory::Create("Camera Motion");
			e->setData(&c);
			EVENTMANAGER->notifyEvent("CAMERA_MOTION", "MainCanvas", "", e);
		}
		DlgCameras::Instance()->updateInfo(m_pCamera->getName());
	}
	if ('Q' == event.GetKeyCode()){
		//if (false == m_pCamera->isDynamic()) 
		{
			nau::event_::CameraMotion c("UP", direction);
			std::shared_ptr<IEventData> e = nau::event_::EventFactory::Create("Camera Motion");
			e->setData(&c);
			EVENTMANAGER->notifyEvent("CAMERA_MOTION", "MainCanvas", "", e);
		}
		DlgCameras::Instance()->updateInfo(m_pCamera->getName());
	}
	if ('Z' == event.GetKeyCode()){
		//if (false == m_pCamera->isDynamic()) 
		{
			nau::event_::CameraMotion c("DOWN", direction);
			std::shared_ptr<IEventData> e = nau::event_::EventFactory::Create("Camera Motion");
			e->setData(&c);
			EVENTMANAGER->notifyEvent("CAMERA_MOTION", "MainCanvas", "", e);
		}
		DlgCameras::Instance()->updateInfo(m_pCamera->getName());
	}

	//if (m_pCamera->isDynamic()){
	//	EVENTMANAGER->notifyEvent("DYNAMIC_CAMERA", "MainCanvas", "", NULL);
	//}

	if ('N' == event.GetKeyCode()) {
		RENDERER->setPropb(IRenderer::DEBUG_DRAW_CALL, true);
	}

	//if ('+' == event.GetKeyCode() || WXK_NUMPAD_ADD == event.GetKeyCode()) {
	//	SceneObject *aObject = RENDERMANAGER->getScene ("MainScene")->getSceneObject ("pPlane1");

	//	aObject->_getTransformPtr()->translate (0.0f, 0.5f, 0.0f);
	//}

	//if ('-' == event.GetKeyCode() || WXK_NUMPAD_SUBTRACT == event.GetKeyCode()) {
	//	SceneObject *aObject = RENDERMANAGER->getScene ("MainScene")->getSceneObject ("pPlane1");

	//	aObject->_getTransformPtr()->translate (0.0f, -0.5f, 0.0f);
	//}

	//std::shared_ptr<Camera> &cam = RENDERMANAGER->getCamera ("testCamera");

	if ('L' == event.GetKeyCode()){
		
		if (false == physics) {
			m_pEngine->enablePhysics();
			//cam->setDynamic(true);
		} else {
			m_pEngine->disablePhysics();
			//cam->setDynamic(false);
		}
		physics = !physics;
	}
	event.Skip();
}


void 
GLCanvas::OnMouseMove (wxMouseEvent& event) {

	if (m_pEngine->mouseMotion(event.GetX(), event.GetY()))
		return;

	static bool first = true; // Can't be here!

	if (event.ButtonIsDown (wxMOUSE_BTN_LEFT)) {
		if (!m_tracking)
			return;

		_setCamera();
		if (0 == m_pCamera) {
			return;
		}
	
		vec4 camView = m_pCamera->getPropf4(Camera::VIEW_VEC);
		vec4 camPosition = m_pCamera->getPropf4(Camera::POSITION);
		vec4 camRight = m_pCamera->getPropf4(Camera::NORMALIZED_RIGHT_VEC);
		vec4 camUp = m_pCamera->getPropf4(Camera::NORMALIZED_UP_VEC);

		newX = event.GetX();
		newY = event.GetY();

		float m_ScaleFactor = 1.0f / 100.0f;

		m_Alpha = m_AlphaAux - (float)(newX - m_OldX) * m_ScaleFactor;
		m_Beta = m_BetaAux + (float)(m_OldY - newY) * m_ScaleFactor;

		//nau::event_::CameraOrientation c(m_Alpha, m_Beta);
		//nau::event_::IEventData *e= nau::event_::EventFactory::create("Camera Orientation");
		//e->setData(&c);
		//EVENTMANAGER->notifyEvent("CAMERA_ORIENTATION", "MainCanvas", "", e);
		//delete e;
		//vec4 camView;

		if (m_Beta > M_PI * 0.48)
			m_Beta = (float)M_PI * 0.48f;
		else if (m_Beta < -M_PI * 0.48)
			m_Beta =  -(float)M_PI * 0.48f;
		camView.x = cos(m_Beta) * sin(m_Alpha);;
		camView.y = sin(m_Beta);
		camView.z = cos(m_Beta) * cos(m_Alpha);
		camView.w = 0;
//		vec4 up = camRight.cross(camView);
		m_pCamera->setCamera(vec3(camPosition.x, camPosition.y, camPosition.z),
							vec3(camView.x, camView.y, camView.z),
							vec3(0,1,0));

		DlgCameras::Instance()->updateInfo(m_pCamera->getName());

		event.Skip ();
	}
	else if (event.ButtonIsDown(wxMOUSE_BTN_RIGHT)) {
		if (!m_tracking)
			return;

		_setCamera();
		if (0 == m_pCamera) {
			return;
		}
		newX = event.GetX();
		newY = event.GetY();

		float m_ScaleFactor = 1.0f / 100.0f;

		m_Alpha = m_AlphaAux + (float)(newX - m_OldX) * m_ScaleFactor;
		m_Beta = m_BetaAux - (float)(m_OldY - newY) * m_ScaleFactor;

		//nau::event_::CameraOrientation c(m_Alpha, m_Beta);
		//nau::event_::IEventData *e= nau::event_::EventFactory::create("Camera Orientation");
		//e->setData(&c);
		//EVENTMANAGER->notifyEvent("CAMERA_ORIENTATION", "MainCanvas", "", e);
		//delete e;
		//vec4 camView;

		vec4 dir;
		if (m_Beta > M_PI * 0.48)
			m_Beta = (float)M_PI * 0.48f;
		else if (m_Beta < -M_PI * 0.48)
			m_Beta = -(float)M_PI * 0.48f;
		dir.x = cos(m_Beta) * sin(m_Alpha);;
		dir.y = sin(m_Beta);
		dir.z = cos(m_Beta) * cos(m_Alpha);
		dir.scale(-m_Radius);
		vec4 camPos = m_Center;
		camPos += dir;
		//		vec4 up = camRight.cross(camView);
		m_pCamera->setCamera(vec3(camPos.x, camPos.y, camPos.z),
			vec3(m_Center.x - camPos.x, m_Center.y - camPos.y, m_Center.z - camPos.z),
			vec3(0, 1, 0));

		DlgCameras::Instance()->updateInfo(m_pCamera->getName());

		event.Skip();

	}

}


void
GLCanvas::OnMiddleUp(wxMouseEvent &event) {

	if (m_pEngine->mouseButton(Nau::MouseAction::PRESSED, Nau::MouseButton::MIDDLE, event.GetX(), event.GetY()))
		event.Skip();
	event.Skip();
}


void
GLCanvas::OnRightUp(wxMouseEvent &event) {

	if (m_pEngine->mouseButton(Nau::MouseAction::RELEASED, Nau::MouseButton::RIGHT, event.GetX(), event.GetY()))
		return;

	m_tracking = false;
	event.Skip();
}


void
GLCanvas::OnRightDown(wxMouseEvent &event) {

	if (m_pEngine->mouseButton(Nau::MouseAction::PRESSED, Nau::MouseButton::RIGHT, event.GetX(), event.GetY()))
		return;
	_setCamera();
	if (m_pCamera) {
		m_OldX = event.GetX();
		m_OldY = event.GetY();

		m_AlphaAux = 0.0f;
		m_BetaAux = 0.0f;
		m_tracking = true;

		m_Radius = NAU->getDepthAtCenter();
		vec4 camPos = m_pCamera->getPropf4(Camera::POSITION);
		vec4 camDir = m_pCamera->getPropf4(Camera::VIEW_VEC);
		vec4 aux = camDir;
		aux.scale(m_Radius);
		m_Center = camPos;
		m_Center -= aux;

		m_AlphaAux = m_pCamera->getPropf(Camera::ZX_ANGLE)- M_PI;
		m_BetaAux = -m_pCamera->getPropf(Camera::ELEVATION_ANGLE);
	}
	event.Skip();
}


void
GLCanvas::OnLeftDown (wxMouseEvent& event) {

	if (m_pEngine->mouseButton(Nau::MouseAction::PRESSED, Nau::MouseButton::LEFT, event.GetX(), event.GetY()))
		event.Skip();

	_setCamera();
	if (m_pCamera) {
		m_OldX = event.GetX();
		m_OldY = event.GetY();

		m_Beta = m_pCamera->getPropf(Camera::ELEVATION_ANGLE);
		m_Alpha = m_pCamera->getPropf(Camera::ZX_ANGLE);


		m_AlphaAux = m_Alpha;
		m_BetaAux = m_Beta;
		m_tracking = true;
	}
	event.Skip ();
}


void
GLCanvas::OnLeftUp (wxMouseEvent &event) {
	
	if (m_pEngine->mouseButton(Nau::MouseAction::RELEASED, Nau::MouseButton::LEFT, event.GetX(), event.GetY()))
		event.Skip();

	m_tracking = false;


	//static int last = 0;

	//m_RiverAnimation = m_pEngine->getAnimation ("RiverAnimation");

	//if (abs (OldX - event.GetX()) < 10 && abs (OldY - event.GetY()) < 10 && 0 == m_RiverAnimation ) {
	//	std::vector<ISceneObject*> objects;
	//	ISceneObject *flood1 = RENDERMANAGER->getScene ("MainScene")->getSceneObject ("cheia1");
	//	ISceneObject *flood2 = RENDERMANAGER->getScene ("MainScene")->getSceneObject ("cheia2");
	//	ISceneObject *riber = RENDERMANAGER->getScene ("MainScene")->getSceneObject ("pPlane1");
	//	
	//	objects.push_back (RENDERMANAGER->getScene ("MainScene")->getSceneObject ("preto"));
	//	objects.push_back (flood1);
	//	objects.push_back (flood2);
	//	
	//	Camera *cam = RENDERMANAGER->getCamera ("MainCamera");
	//	
	//	int mx, my, wx, wy;

	//	wx = GetParent()->GetPosition().x;
	//	wy = GetParent()->GetPosition().y;

	//	mx = event.GetPosition().x;
	//	my = event.GetPosition().y;

	//	int pickedObject = m_pEngine->picking (mx, my, objects, *cam);

	//	if (pickedObject > 0) {
	//		ITransform *transf = riber->_getTransformPtr(); /***MARK***/ //This should be done inside IBoundingVolume
	//		const IBoundingVolume *bounding = riber->getBoundingVolume();
	//		
	//		vec3 center;

	//		center.set (bounding->getCenter().x, bounding->getCenter().y, bounding->getCenter().z);

	//		transf->getMat44().transform (center);

	//		switch (pickedObject) {


	//			case 1: {
	//				vec3 delta (flood1->getBoundingVolume()->getCenter());

	//				if (last == 1) {
	//					last = 0;

	//					delta -= vec3 (0.0f, 4.228f, 0.0f);					
	//					m_RiverAnimation = new nau::animation::LinearAnimation (riber, vec3 (0.0f, -delta.y, 0.0f));
	//					m_pEngine->addAnimation ("RiverAnimation", m_RiverAnimation);
	//				} else {
	//					last = 1;

	//					delta -= center;
	//					m_RiverAnimation = new nau::animation::LinearAnimation (riber, vec3 (0.0f, delta.y, 0.0f));
	//					m_pEngine->addAnimation ("RiverAnimation", m_RiverAnimation);
	//				}
	//				break;
	//			}
	//			case 2: {
	//				vec3 delta (flood2->getBoundingVolume()->getCenter());

	//				if (last == 2) {
	//					last = 0;

	//					delta -= vec3 (0.0f, 4.228f, 0.0f);					
	//					m_RiverAnimation = new nau::animation::LinearAnimation (riber, vec3 (0.0f, -delta.y, 0.0f));
	//					m_pEngine->addAnimation ("RiverAnimation", m_RiverAnimation);
	//				} else {
	//					last = 2;

	//					delta -= center;
	//					m_RiverAnimation = new nau::animation::LinearAnimation (riber, vec3 (0.0f, delta.y, 0.0f));
	//					m_pEngine->addAnimation ("RiverAnimation", m_RiverAnimation);
	//				}
	//				break;
	//			}
	//		}
	//	}
	//}
}


void
GLCanvas::BreakResume ()
{
	if (isPaused)
		m_pEngine->stepCompleteFrame();

	isPaused=!isPaused;
}


void 
GLCanvas::StepPass() {

	if (isPaused) {
		m_pEngine->stepPass();

		if (RENDERMANAGER->getActivePipeline()->getPassCounter() == 0)
			SwapBuffers();
	}
}


void 
GLCanvas::StepToEndOfFrame() {

	if (isPaused) {
		m_pEngine->stepCompleteFrame();
		SwapBuffers();
	}
}


void
GLCanvas::StepUntilSamePassNextFrame() {

	if (isPaused) {
		int n = RENDERMANAGER->getActivePipeline()->getPassCounter();
		m_pEngine->stepCompleteFrame();
		SwapBuffers();
		m_pEngine->stepPasses(n);
	}
}


bool
GLCanvas::IsPaused() {

	return isPaused;
}


void
GLCanvas::MultiStep(int stepSize) {

	step = stepSize;
}



//bool
//GLCanvas::changeWaterState (bool state)
//{
//	Camera *cam = RENDERMANAGER->getCamera ("MainCamera");
//	ISceneObject *river = RENDERMANAGER->getScene ("MainScene")->getSceneObject ("pPlane1");	
//	
//	const IBoundingVolume *volume2 = river->getBoundingVolume();
//	ITransform *transf = river->_getTransformPtr();
//
//	vec3 centerOfWater = volume2->getCenter();
//	transf->getMat44().transform (centerOfWater);
//
//
//	if (true == state) {
//		m_pEngine->getWorld().disableObject ("water");
//
//		if (cam->getPosition().y < centerOfWater.y) {
//			return false;
//		} else {
//			return true;
//		}
//	}
//
//	if (cam->getPosition().y - 1.75 > centerOfWater.y) {
//		m_pEngine->getWorld().enableObject ("water");
//		return true;
//	}
//
//	return false;
//}

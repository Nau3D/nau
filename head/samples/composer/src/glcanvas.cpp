#include <glcanvas.h>

#include <main.h>

#include <iostream>

#include <nau.h>
#include <nau/math/utils.h>

#include <nau/debug/profile.h>

#include <nau/scene/sceneobject.h>
#include <nau/geometry/iboundingvolume.h>

#include <nau/event/eventFactory.h>
#include <nau/event/cameraMotion.h>
#include <nau/event/cameraOrientation.h>

#include <nau/slogger.h>



using namespace std;
using namespace nau::math;
using namespace nau::scene;
using namespace nau::geometry;

#define VELOCITY 50.0f
#define NON_DYNAMIC_VELOCITY 0.05f  // 2 meters per second

bool isPaused = false;

GlCanvas::GlCanvas (wxWindow *parent,
                     const wxWindowID id,
					 const int * 	attribList,
					 const int * contextAttribs,
                     const wxPoint &pos,
                     const wxSize &size,
                     long style,
                     const wxString &name)
   : wxGLCanvas (parent, id, attribList, contextAttribs, pos, size, style, name), 
	init (false), m_pEngine (0), m_pCamera (0), m_CounterFps (0),
	m_Alpha (3.14159f), m_AlphaAux (0.0f), m_Beta (0.0f), m_BetaAux (0.0f), OldX(0),OldY(0),
	m_tracking(0), /*m_RiverAnimation (0), */m_WaterState (true), m_Stereo (false)
{
	//ctor
	init = true;
	m_Timer.Start();
	p_GLC = new wxGLContext(this);
	SetCurrent(*p_GLC);
}

GlCanvas::GlCanvas (wxWindow *parent,
                     const GlCanvas &other,
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
}


GlCanvas::~GlCanvas()
{
   //dtor

}

void
GlCanvas::setEngine (nau::Nau* engine)
{
	m_pEngine = engine;
}

void 
GlCanvas::setCamera ()
{
	m_pCamera = NAU->getActiveCamera();
	if (m_pCamera) {
		vec4 v = m_pCamera->getPropf4(Camera::NORMALIZED_VIEW_VEC);
		m_Beta = asin(v.y);
		m_Alpha = atan2(v.x,v.z);
	}
}

void
GlCanvas::_setCamera() {

	m_pCamera = NAU->getActiveCamera();
	
}


void 
GlCanvas::OnPaint (wxPaintEvent &event)
{
	PROFILE("Composer");
	wxPaintDC dc(this);

	if(!isPaused){
		Render();
	}
	event.Skip ();
}

void 
GlCanvas::OnSize (wxSizeEvent &event)
{
//   wxGLCanvas::OnSize(event);

   int width;
   int height;

   GetClientSize (&width, &height);

//#ifndef __WXMOTIF__
//   if (GetContext ())
//#endif
//   {
//		SetCurrent ();
		if (0 != m_pEngine) {
			EVENTMANAGER->notifyEvent("WINDOW_SIZE_CHANGED","Canvas","",new EventVec3(vec3(width,height,0)));
			//m_pEngine->setWindowSize (width, height);
		}
   //}
}

void 
GlCanvas::OnEraseBackground (wxEraseEvent &event)
{
   event.Skip ();
}


void 
GlCanvas::OnEnterWindow (wxMouseEvent &event)
{
   SetFocus ();
   event.Skip ();
}


void 
GlCanvas::Render ()
{

	PROFILE ("Main cicle");

	if (0 != m_pEngine) {
		m_pEngine->step();
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

		sprintf(fps, "FPS: %d - Triangles: %d", m_CounterFps, RENDERER->getCounter(nau::render::IRenderer::TRIANGLE_COUNTER));
		wxTopLevelWindow *tlw = (wxTopLevelWindow *)(this->GetParent());

		tlw->SetTitle(wxString::FromAscii(fps));

		m_CounterFps = 0;

	}
}


void
GlCanvas::OnIdle (wxIdleEvent& event)
{
	PROFILE("Composer");

    if(!isPaused){
		this->Render();
    }
	event.RequestMore();
	DlgAtomics::Instance()->update();
}

/*void 
 GlCanvas::InitGL () 
{
   SetCurrent ();
   
   glClearColor (0.8f, 0.8f, 0.8f, 0.0f);
   
   //glEnable (GL_LIGHT0);
	//glEnable (GL_LIGHTING); 
   glShadeModel (GL_SMOOTH);
   glEnable (GL_DEPTH_TEST);
   glEnable (GL_CULL_FACE);
}*/ 


void 
GlCanvas::OnKeyUp(wxKeyEvent & event)
{
	if (m_pCamera && true == m_pCamera->isDynamic()) {
		vec3 v3;
		v3.set(0.0f, 0.0f, 0.0f);
		m_pEngine->getWorld().setVelocity ("testCamera", v3);
	}
}

void 
GlCanvas::OnKeyDown(wxKeyEvent & event) 
{

	static bool physics = true;

	_setCamera();
	if (0 == m_pCamera) {
		return;
	}


	vec4 camPosition = m_pCamera->getPropf4(Camera::POSITION);
	vec4 camUp = m_pCamera->getPropf4(Camera::NORMALIZED_UP_VEC);
	vec4 camView = m_pCamera->getPropf4(Camera::NORMALIZED_VIEW_VEC);
//	vec3& camLookAt = m_pCamera->getLookAtPoint();

	if ('K' == event.GetKeyCode()) {
		m_pEngine->sendKeyToEngine (event.GetKeyCode());	
	}
	if ('I' == event.GetKeyCode()) {
		m_pEngine->sendKeyToEngine('I');
	}
	if ('M' == event.GetKeyCode()) {
		EVENTMANAGER->notifyEvent("NEXT_POSE", "MainCanvas", "", NULL);
	}

	if ('9' >= event.GetKeyCode() && '0' <= event.GetKeyCode()) {
		//m_pEngine->sendKeyToEngine (event.GetKeyCode());
		Camera *aCam = RENDERMANAGER->getCamera ("MainCamera");

		vec3 v;

		switch (event.GetKeyCode()) {
			case '1':	
				v.set (-14.486f * cos (DegToRad (-137.0)) - 59.256 * -sin (DegToRad (-137.0)), 
					13.266f, 
					-59.256 * cos (DegToRad(-137.0)) + -14.486f * sin (DegToRad (-137.0)));

				aCam->setCamera (vec3 (-14.486f, 13.266f, -59.256f), v, vec3 (0.0f, 1.0f, 0.0f));
				break;
			case '2':
				v.set (0.0f, 0.0f, -1.0f * cos (DegToRad (-141.4f)));

				aCam->setCamera (vec3 (7.930f, 16.135f, -38.392f), v, vec3 (0.0f, 1.0f, 0.0f));
				break;

			case '3':
				v.set (0.0f, 0.0f, -1.0f * cos (DegToRad (-81.8f)));

				aCam->setCamera (vec3 (7.374f, 14.465f, -58.637f), v, vec3 (0.0f, 1.0f, 0.0f));
				break;

			case '4':
				v.set (0.0f, 0.0f, -1.0f * cos (DegToRad (-17.0f)));

				aCam->setCamera (vec3 (13.363f, 13.977f, -47.436f), v, vec3 (0.0f, 1.0f, 0.0f));
				break;

			case '5':
				v.set (0.0f, 0.0f, -1.0f * cos (DegToRad (135.58f)));

				aCam->setCamera (vec3 (-131.176f, 9.555f, 188.927f), v, vec3 (0.0f, 1.0f, 0.0f));
				break;
		}

	}

	if ('B' == event.GetKeyCode()) {
		m_pEngine->sendKeyToEngine (event.GetKeyCode());
	}

	if ('P' == event.GetKeyCode()) {
		m_pEngine->sendKeyToEngine (event.GetKeyCode());
	}

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
		if (true == m_pCamera->isDynamic()) {
			vec4 vel (camView);

			//vel *= 2.0f;

			//vel -= m_OldCamView;

			//m_OldCamView.set (vel.x, vel.y, vel.z);


			vel.y = 0.0f;
			vel.normalize();
			vel *= -direction;
			vec3 v3;
			v3.set(vel.x,vel.y,vel.z);
			m_pEngine->getWorld().setVelocity ("testCamera", v3);
		} 
		else {
			nau::event_::CameraMotion c("BACKWARD", direction);
			nau::event_::IEventData *e= nau::event_::EventFactory::create("Camera Motion");
			e->setData(&c);
			EVENTMANAGER->notifyEvent("CAMERA_MOTION", "MainCanvas", "", e);
			delete e;
		}
		DlgCameras::Instance()->updateInfo(m_pCamera->getName());
	}

	if ('W' == event.GetKeyCode()) {
		if (true == m_pCamera->isDynamic()) {
			vec4 vel (camView);
			
			//vel *= 2.0f;

			//vel -= m_OldCamView;

			//m_OldCamView.set (camView.x, camView.y, camView.z);
			
			vel.y = 0.0f;
			vel.normalize();
			vel *= direction; 
			
			vec3 v3;
			v3.set(vel.x,vel.y,vel.z);
			m_pEngine->getWorld().setVelocity ("testCamera", v3);
		} 
		else {

			nau::event_::CameraMotion c("FORWARD", direction);
			nau::event_::IEventData *e= nau::event_::EventFactory::create("Camera Motion");
			e->setData(&c);
			EVENTMANAGER->notifyEvent("CAMERA_MOTION", "MainCanvas", "", e);
			delete e;

		}
		DlgCameras::Instance()->updateInfo(m_pCamera->getName());
	}

	if ('A' == event.GetKeyCode()){
		if (true == m_pCamera->isDynamic()) {
			vec4 vel (camView.cross (camUp));

			vel *= -direction;
			vel.y = 0.0f;
			vec3 v3;
			v3.set(vel.x,vel.y,vel.z);
			m_pEngine->getWorld().setVelocity ("testCamera", v3);

		} 
		else {

			nau::event_::CameraMotion c("LEFT", direction);
			nau::event_::IEventData *e= nau::event_::EventFactory::create("Camera Motion");
			e->setData(&c);
			EVENTMANAGER->notifyEvent("CAMERA_MOTION", "MainCanvas", "", e);
			delete e;
		}
		DlgCameras::Instance()->updateInfo(m_pCamera->getName());

	}
	if ('D' == event.GetKeyCode()){
		if (true == m_pCamera->isDynamic()) {
			vec4 vel (camView.cross (camUp));

			vel *= direction;
			vel.y = 0.0f;
			vec3 v3;
			v3.set(vel.x,vel.y,vel.z);
			m_pEngine->getWorld().setVelocity ("testCamera", v3);
		} 
		else {
			nau::event_::CameraMotion c("RIGHT", direction);
			nau::event_::IEventData *e= nau::event_::EventFactory::create("Camera Motion");
			e->setData(&c);
			EVENTMANAGER->notifyEvent("CAMERA_MOTION", "MainCanvas", "", e);
			delete e;
		}
		DlgCameras::Instance()->updateInfo(m_pCamera->getName());
	}
	if ('Q' == event.GetKeyCode()){
		if (false == m_pCamera->isDynamic()) {
			nau::event_::CameraMotion c("UP", direction);
			nau::event_::IEventData *e= nau::event_::EventFactory::create("Camera Motion");
			e->setData(&c);
			EVENTMANAGER->notifyEvent("CAMERA_MOTION", "MainCanvas", "", e);
			delete e;
		}
		DlgCameras::Instance()->updateInfo(m_pCamera->getName());
	}
	if ('Z' == event.GetKeyCode()){
		if (false == m_pCamera->isDynamic()) {
			nau::event_::CameraMotion c("DOWN", direction);
			nau::event_::IEventData *e= nau::event_::EventFactory::create("Camera Motion");
			e->setData(&c);
			EVENTMANAGER->notifyEvent("CAMERA_MOTION", "MainCanvas", "", e);
			delete e;
		}
		DlgCameras::Instance()->updateInfo(m_pCamera->getName());
	}

	if (m_pCamera->isDynamic()){
		EVENTMANAGER->notifyEvent("DYNAMIC_CAMERA", "MainCanvas", "", NULL);
	}

/*	if (WXK_SPACE == event.GetKeyCode()) {
		if (true == m_pCamera->m_IsDynamic) {
			vec3 vel (camUp);

			vel *= direction * 3;
			vel.x = 0.0f;
			vel.z = 0.0f;
			
			m_pEngine->getWorld().setVelocity ("MainCamera", vel);

			m_WaterState = changeWaterState (m_WaterState);

			
		}
	}
*/
/*	if ('N' == event.GetKeyCode()){
		m_pCamera->setNearPlane (m_pCamera->getNearPlane() + 0.5f);
	}

	if ('M' == event.GetKeyCode()){
		m_pCamera->setNearPlane (m_pCamera->getNearPlane() - 0.5f);
	}


	if ('F' == event.GetKeyCode()){
		m_pCamera->setFarPlane (m_pCamera->getFarPlane() + 1.0f);
	}

	if ('G' == event.GetKeyCode()){
		m_pCamera->setFarPlane (m_pCamera->getFarPlane() - 1.0f);
	}
*/
	if ('+' == event.GetKeyCode() || WXK_NUMPAD_ADD == event.GetKeyCode()) {
		SceneObject *aObject = RENDERMANAGER->getScene ("MainScene")->getSceneObject ("pPlane1");

		aObject->_getTransformPtr()->translate (0.0f, 0.5f, 0.0f);
	}

	if ('-' == event.GetKeyCode() || WXK_NUMPAD_SUBTRACT == event.GetKeyCode()) {
		SceneObject *aObject = RENDERMANAGER->getScene ("MainScene")->getSceneObject ("pPlane1");

		aObject->_getTransformPtr()->translate (0.0f, -0.5f, 0.0f);
	}

	nau::scene::Camera *cam = RENDERMANAGER->getCamera ("testCamera");

	if ('L' == event.GetKeyCode()){
		
		if (false == physics) {
			m_pEngine->enablePhysics();
			cam->setDynamic(true);
		} else {
			m_pEngine->disablePhysics();
			cam->setDynamic(false);
		}
		physics = !physics;
	}

	//if ('Y' == event.GetKeyCode()){
	//	
	//	m_Stereo = !m_Stereo;

	//	if (true == m_Stereo) {
	//		m_pEngine->enableStereo();
	//	} else {
	//		m_pEngine->disableStereo();
	//	}
	//}
	event.Skip();
}




void 
GlCanvas::OnMouseMove (wxMouseEvent& event)
{

	static bool first = true; // Can't be here!

	if (!event.ButtonIsDown (wxMOUSE_BTN_LEFT)) {
		return;
	}

	if (!m_tracking)
		return;

	_setCamera();
	if (0 == m_pCamera) {
		return;
	}
	
	vec4 camView = m_pCamera->getPropf4(Camera::NORMALIZED_VIEW_VEC);
	vec4 camPosition = m_pCamera->getPropf4(Camera::POSITION);

	if (true == m_tracking) {
		m_OldCamView.set(camView.x, camView.y, camView.z);
	}

	newX = event.GetX();
	newY = event.GetY();

	float m_ScaleFactor = 1.0f / 100.0f;

	m_Alpha = m_AlphaAux - (float)(newX - OldX) * m_ScaleFactor;
	m_Beta = m_BetaAux + (float)(OldY - newY) * m_ScaleFactor;
/*	if (m_Beta > 1.5f) 
		m_Beta = 1.5f;
	else if (m_Beta < -1.5f)
		m_Beta = -1.5f;
*/
	nau::event_::CameraOrientation c(m_Alpha, m_Beta);
	nau::event_::IEventData *e= nau::event_::EventFactory::create("Camera Orientation");
	e->setData(&c);
	EVENTMANAGER->notifyEvent("CAMERA_ORIENTATION", "MainCanvas", "", e);
	delete e;

	/*
	camView.x = cos(m_Beta) * sin(m_Alpha);
	camView.z = cos(m_Beta) * cos(m_Alpha);
	camView.y = sin(m_Beta);
	*/

	camV.x = cos(m_Beta) * sin(m_Alpha);;
	camV.y = sin(m_Beta);
	camV.z = cos(m_Beta) * cos(m_Alpha);

	if (!(true == m_tracking)) {
//		m_OldCamView.set (camView.x, camView.y, camView.z);
		m_tracking = true;
	}
	DlgCameras::Instance()->updateInfo(m_pCamera->getName());

	event.Skip ();
}

void
GlCanvas::OnLeftDown (wxMouseEvent& event)
{

	_setCamera();
	if (m_pCamera) {
		OldX = event.GetX();
		OldY = event.GetY();

		m_Beta = m_pCamera->getPropf(Camera::ELEVATION_ANGLE);//getElevationAngle();
		m_Alpha = m_pCamera->getPropf(Camera::ZX_ANGLE);//getZXAngle();


		m_AlphaAux = m_Alpha;
		m_BetaAux = m_Beta;
		m_tracking = true;
	}
	event.Skip ();
}

void
GlCanvas::OnLeftUp (wxMouseEvent &event) {
	
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
GlCanvas::BreakResume ()
{
   
   isPaused=!isPaused;
}

bool
GlCanvas::IsPaused ()
{
	return isPaused;
}

//bool
//GlCanvas::changeWaterState (bool state)
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

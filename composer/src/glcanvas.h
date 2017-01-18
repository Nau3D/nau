#ifndef GlCanvas_H
#define GlCanvas_H



#include <glbinding/gl/gl.h>
using namespace gl;

#include <nau.h>
#include <wx/wx.h>
#include <wx/glcanvas.h>

class GLCanvas : public wxGLCanvas
{
public:
	GLCanvas(wxWindow * parent,
               const wxWindowID id = -1,
			   const int * 	attribList = NULL,
			  // const int *contextAttribs = NULL,
                const wxPoint &pos = wxDefaultPosition,
               const wxSize &size = wxDefaultSize,
               long style = 0,
               const wxString &name = _("GLCanvas"));
	GLCanvas(wxWindow * parent,
               const GLCanvas &other,
               const wxWindowID id = -1,
			   const int * 	attribList = NULL,
			  // const int *contextAttribs = NULL,
                const wxPoint &pos = wxDefaultPosition,
               const wxSize &size = wxDefaultSize,
               long style = 0,
               const wxString &name = _("GLCanvas"));

	void DoSetCurrent();

	void OnPaint(wxPaintEvent & event);
	void BreakResume ();
	void StepPass();
	void StepToEndOfFrame();
	void StepUntilSamePassNextFrame();
	void OnSize(wxSizeEvent & event);
	void OnEraseBackground(wxEraseEvent & event);
	void OnEnterWindow(wxMouseEvent & event);
	void OnKeyDown(wxKeyEvent & event);
	void OnKeyUp(wxKeyEvent & event);
	void OnIdle (wxIdleEvent& event);
	void OnMouseMove (wxMouseEvent& event);
	void OnLeftDown (wxMouseEvent& event);
	void OnLeftUp (wxMouseEvent& event);
	void OnRightUp(wxMouseEvent& event);
	void OnRightDown(wxMouseEvent& event);
	void OnMiddleUp(wxMouseEvent& event);

	bool IsPaused ();
	void MultiStep(int stepSize = -1);

   void Render(void);
  // void InitGL(void);

	void setEngine (nau::Nau* engine);
	void setCamera ();

	virtual ~GLCanvas();

   DECLARE_EVENT_TABLE()
protected:

private:
	bool changeWaterState (bool state);
	void _setCamera();

	bool inited;
	wxWindow *mParent;

private:

	wxGLContext *p_GLC;
	// ONLY FOR DEBUG PURPOSES
	long int newX,newY;
	nau::math::vec3 camV;
	

	bool init;
	nau::Nau* m_pEngine;
	nau::scene::Camera* m_pCamera;

	long int m_OldX;
	long int m_OldY;


	float m_BetaAux;
	float m_Beta;
	float m_AlphaAux;
	float m_Alpha;
	float m_Radius;
	bool m_tracking;
	vec4 m_Center;

	bool m_WaterState;

	wxStopWatch m_Timer;
	int m_CounterFps;

	bool m_Stereo;
	
	nau::math::vec4 m_OldCamView;
	//nau::animation::IAnimation *m_RiverAnimation;

	int step;

};

#endif // GlCanvas_H

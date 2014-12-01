#ifndef GlCanvas_H
#define GlCanvas_H

#include <GL/glew.h>
#include <nau.h>
#include <wx/wx.h>
#include <wx/glcanvas.h>

class GlCanvas : public wxGLCanvas
{
public:
   GlCanvas (wxWindow * parent,
               const wxWindowID id = -1,
			   const int * 	attribList = NULL,
			   const int *contextAttribs = NULL,
                const wxPoint &pos = wxDefaultPosition,
               const wxSize &size = wxDefaultSize,
               long style = 0,
               const wxString &name = _("GlCanvas"));
   GlCanvas (wxWindow * parent,
               const GlCanvas &other,
               const wxWindowID id = -1,
			   const int * 	attribList = NULL,
			   const int *contextAttribs = NULL,
                const wxPoint &pos = wxDefaultPosition,
               const wxSize &size = wxDefaultSize,
               long style = 0,
               const wxString &name = _("GlCanvas"));

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

	bool IsPaused ();
	void MultiStep(int stepSize = -1);

   void Render(void);
  // void InitGL(void);

	void setEngine (nau::Nau* engine);
	void setCamera ();

	virtual ~GlCanvas();

   DECLARE_EVENT_TABLE()
protected:

private:
	bool changeWaterState (bool state);
	void _setCamera();

private:

	wxGLContext *p_GLC;
	// ONLY FOR DEBUG PURPOSES
	long int newX,newY;
	nau::math::vec3 camV;
	

	bool init;
	nau::Nau* m_pEngine;
	nau::scene::Camera* m_pCamera;

	long int OldX;
	long int OldY;


	float m_BetaAux;
	float m_Beta;
	float m_AlphaAux;
	float m_Alpha;
	bool m_tracking;

	bool m_WaterState;

	wxStopWatch m_Timer;
	int m_CounterFps;

	bool m_Stereo;
	
	nau::math::vec4 m_OldCamView;
	//nau::animation::IAnimation *m_RiverAnimation;

	int step;

};

#endif // GlCanvas_H

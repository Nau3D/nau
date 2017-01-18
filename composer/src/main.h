#ifndef MAIN_H
#define MAIN_H



#include <wx/wxprec.h>

#include <wx/string.h>
#include "dialogs/dlgOGL.h"
#include "dialogs/dlgLog.h"
#include "dialogs/dlgAtomics.h"
#include "dialogs/dlgTextureLib.h"
#include "dialogs/dlgCameras.h"
#include "dialogs/dlgMaterials.h"
#include "dialogs/dlgLights.h"
#include "dialogs/dlgShaders.h"
#include "dialogs/dlgScenes.h"
#include "dialogs/dlgPhysics.h"
#include "dialogs/dlgPass.h"
#include "dialogs/dlgTrace.h"
#include "dialogs/dlgDbgPrograms.h"
#include "dialogs/dlgBuffers.h"
#include "dialogs/dlgDbgStep.h"
#include "dialogs/dlgRenderTarget.h"
#include "dialogs/dlgViewports.h"

#include "glcanvas.h"

#include <nau.h>


#ifdef __BORLANDC__
    #pragma hdrstop
#endif

#ifndef WX_PRECOMP
    #include <wx/wx.h>
#endif

class WndComposer : public wxApp
{
	public:
		virtual bool OnInit();
};

class FrmMainFrame: public wxFrame
{
public:
	FrmMainFrame (wxFrame *frame, const wxString& title);
	~FrmMainFrame ();

	void init();

private:
   GLCanvas *m_Canvas;
	wxMenu *fileMenu, *renderMenu, *assetsMenu, *materialsMenu, *physicsMenu,
		*debugMenu, *aboutMenu;
	wxMenuBar *mbar;
	bool m_Inited, m_Tracing;

	nau::Nau *m_pRoot; 

	float m_Width, m_Height;

	void updateDlgs();


	void OnProjectLoad(wxCommandEvent& event);
	void OnModelLoad (wxCommandEvent& event);
	void OnDirectoryLoad (wxCommandEvent& event);
	void OnProjectSave(wxCommandEvent& event);
	void OnProcess (wxCommandEvent& event);
    void OnQuit(wxCommandEvent& event);

	void OnDlgPass(wxCommandEvent& event);
	void OnResetFrameCount(wxCommandEvent& event);
	void OnRenderMode(wxCommandEvent& event);
	void OnSetRenderFlags(wxCommandEvent& event);

	void OnDlgScenes(wxCommandEvent& event);
	void OnDlgViewports(wxCommandEvent& event);
	void OnDlgCameras(wxCommandEvent& event);
	void OnDlgLights(wxCommandEvent& event);

	void OnDlgMaterials(wxCommandEvent& event);
	void OnDlgTextures(wxCommandEvent& event);
	void OnDlgAtomics(wxCommandEvent& event);
	void OnDlgShaders(wxCommandEvent& event);
	void OnDlgBuffers(wxCommandEvent& event);
	void OnDlgRenderTargets(wxCommandEvent& event);

	void OnDlgPhysics(wxCommandEvent& event);

	void OnDlgLog(wxCommandEvent& event);
	void OnBreakResume(wxCommandEvent& event);
	void OnDlgDbgStep(wxCommandEvent& event);
	void OnDlgDbgTraceRead(wxCommandEvent& event);
	void OnDlgDbgProgram(wxCommandEvent& event);
	void OnScreenShot(wxCommandEvent& event);
	void OnTraceSingleFrame(wxCommandEvent& event);
	void OnTrace(wxCommandEvent& event);
	void OnProfileReset(wxCommandEvent& event);


	
	void OnDlgOGL(wxCommandEvent& event);
    void OnAbout(wxCommandEvent& event);

	void OnKeyDown(wxKeyEvent & event);
	void OnKeyUp(wxKeyEvent & event);

	void OnClose(wxCloseEvent& event);

	void startStandAlone (void);

	void loadProject(const char *s);

   DECLARE_EVENT_TABLE();

	//void buildPhysics (void);
	//void OnPhysicsBuild (wxCommandEvent &event);
	void OnPhysicsMode (wxCommandEvent &event);
	
};

#endif // MAIN_H

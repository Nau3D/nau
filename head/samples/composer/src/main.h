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
#include "dialogs/dlgPass.h"

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
private:

	nau::Nau *m_pRoot; 

	float m_Width, m_Height;

public:
	FrmMainFrame (wxFrame *frame, const wxString& title);
	~FrmMainFrame ();

	void compile(IScene *scene); 


private:
   GlCanvas *m_Canvas;
	wxMenu *materialsMenu, *helpMenu;
	bool m_Inited;

	void updateDlgs();

	void OnDlgLog(wxCommandEvent& event);
	void OnDlgOGL(wxCommandEvent& event);
	void OnDlgAtomics(wxCommandEvent& event);
	void OnDlgTextures(wxCommandEvent& event);
	void OnDlgCameras(wxCommandEvent& event);
	void OnDlgMaterials(wxCommandEvent& event);
	void OnDlgLights(wxCommandEvent& event);
	void OnDlgShaders(wxCommandEvent& event);
	void OnDlgScenes(wxCommandEvent& event);
	void OnDlgPass(wxCommandEvent& event);
    void OnProjectLoad(wxCommandEvent& event);
	void OnDirectoryLoad (wxCommandEvent& event);
	void OnModelLoad (wxCommandEvent& event);
	void OnModelAppend (wxCommandEvent& event);

    void OnQuit(wxCommandEvent& event);
    void OnAbout(wxCommandEvent& event);

    void OnProcess (wxCommandEvent& event);
	void OnRenderMode(wxCommandEvent& event);
	void OnSetRenderFlags(wxCommandEvent& event);
	void OnSetProfileMaterial(wxCommandEvent& event);
	void OnOctreeBuild (wxCommandEvent& event);
	void OnOctreeCompile (wxCommandEvent& event);
	void OnOctreeWrite (wxCommandEvent& event);
	void OnPhysicsBuild (wxCommandEvent &event);
	void OnPhysicsMode (wxCommandEvent &event);
	void OnKeyDown(wxKeyEvent & event);
	void OnKeyUp(wxKeyEvent & event);
	
	void startStandAlone (void);
	void buildPhysics (void);

   DECLARE_EVENT_TABLE();
	//void initScene (void);
};

#endif // MAIN_H

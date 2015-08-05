#ifndef OGLPANELS_H
#define OGLPANELS_H

#ifdef __GNUG__
#pragma implementation
#pragma interface
#endif


// For compilers that support precompilation, includes "wx/wx.h".
#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif

#include <wx/notebook.h>
#include <wx/grid.h>
#include <wx/colordlg.h>
#include <wx/propgrid/propgrid.h>
#include <wx/propgrid/advprops.h>
#include <wx/propgrid/manager.h>

#include <nau/material/material.h>

//#include "ogl.h"


class DlgOGLPanels {


public:

	nau::material::IState *m_glState;
//	GLCanvas *m_glCanvas;

	wxPropertyGridManager *pg;	

	enum {
		PG
	};

	DlgOGLPanels();
	~DlgOGLPanels();

	void setState(nau::material::IState *aState);
//	void setCanvas(GLCanvas *glCanvas);
	void updatePanel();

	void setPanel(wxSizer *siz, wxWindow *parent);
	void OnProcessPanelChange(wxPropertyGridEvent& e);
};


#endif
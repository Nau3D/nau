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


class DlgMatStatePanels {


public:

	nau::material::IState *m_glState;

	wxPropertyGridManager *m_PG;

	enum {
		PG
	};

	DlgMatStatePanels();
	~DlgMatStatePanels();

	void setState(nau::material::IState *aState);
	void updatePanel();

	void setPanel(wxSizer *siz, wxWindow *parent);
	void resetPropGrid();
	void OnProcessPanelChange(wxPropertyGridEvent& e);
};


#endif
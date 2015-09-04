#ifndef MATBUFFERPANELS_H
#define MATBUFFERPANELS_H

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


class DlgMatBufferPanels {


public:


	wxPropertyGridManager *pg;	
	wxComboBox *itemList;

	enum {
		PG = 100,
		DLG_ITEM_COMBO
	};

	DlgMatBufferPanels();
	~DlgMatBufferPanels();

	void setMaterial(nau::material::Material *aMat);

	void setPanel(wxSizer *siz, wxWindow *parent);
	void onProcessPanelChange(wxPropertyGridEvent& e);
	void onItemListSelect(wxCommandEvent& event);

private:
	nau::material::Material *m_Material;
	int m_CurrentBinding;
	std::vector<int> m_MaterialBindings;

	void updatePanel();
};


#endif
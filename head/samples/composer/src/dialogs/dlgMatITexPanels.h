#ifndef MATITEXPANELS_H
#define MATITEXPANELS_H

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


class DlgMatImageTexturePanels {


public:


	wxPropertyGridManager *pg;	
	wxComboBox *itemList;

	enum {
		PG = 100,
		DLG_ITEM_COMBO
	};

	DlgMatImageTexturePanels();
	~DlgMatImageTexturePanels();

	void setMaterial(std::shared_ptr<nau::material::Material> &aMat);

	void setPanel(wxSizer *siz, wxWindow *parent);
	void resetPropGrid();
	void onProcessPanelChange(wxPropertyGridEvent& e);
	void onItemListSelect(wxCommandEvent& event);

private:
	std::shared_ptr<nau::material::Material> m_Material;
	int m_CurrentUnit;
	std::vector<unsigned int> m_ImageTextureUnits;

	void updatePanel();
};


#endif
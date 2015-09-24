#include "dlgMatStatePanels.h"

#include "propertyManager.h"

#include <nau/material/iState.h>

#include <glbinding/gl/gl.h>
using namespace gl;
//#include <GL/glew.h>





DlgMatStatePanels::DlgMatStatePanels() {
	
}


DlgMatStatePanels::~DlgMatStatePanels(){

}


void DlgMatStatePanels::setState(nau::material::IState *aState) {

	m_glState = aState;
}


void DlgMatStatePanels::setPanel(wxSizer *siz, wxWindow *parent){

	wxStaticBox *sf = new wxStaticBox(parent,-1,wxT("State"));
	wxSizer *sizerf = new wxStaticBoxSizer(sf,wxVERTICAL);

	pg = new wxPropertyGridManager(parent, PG,
				wxDefaultPosition, wxDefaultSize,
				// These and other similar styles are automatically
				// passed to the embedded wxPropertyGrid.
				wxPG_BOLD_MODIFIED | wxPG_SPLITTER_AUTO_CENTER |
				// Plus defaults.
				wxPGMAN_DEFAULT_STYLE
           );
	pg->AddPage(wxT("Standard Items"));

	std::vector<std::string> order = {"DEPTH_FUNC", "DEPTH_TEST", "DEPTH_MASK", "CULL_TYPE", "CULL_FACE",
		"ORDER", "BLEND", "BLEND_SRC", "BLEND_DST", "BLEND_EQUATION", "BLEND_COLOR", "COLOR_MASK_B4"};
	PropertyManager::createOrderedGrid(pg, nau::material::IState::Attribs, order);

	pg->SetSplitterLeft(true);

	sizerf->Add(pg,1,wxEXPAND);
	siz->Add(sizerf,1,wxGROW|wxEXPAND|wxALL,5);
}


void DlgMatStatePanels::OnProcessPanelChange(wxPropertyGridEvent& e){


	const wxString& name = e.GetPropertyName();

	PropertyManager::updateProp(pg, name.ToStdString(), nau::material::IState::Attribs, (AttributeValues *)m_glState);
}


void DlgMatStatePanels::updatePanel(){

	PropertyManager::updateGrid(pg, nau::material::IState::Attribs, (AttributeValues *)m_glState);
	pg->Refresh();
}



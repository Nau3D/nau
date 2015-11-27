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

	m_PG = new wxPropertyGridManager(parent, PG,
				wxDefaultPosition, wxDefaultSize,
				// These and other similar styles are automatically
				// passed to the embedded wxPropertyGrid.
				wxPG_BOLD_MODIFIED | wxPG_SPLITTER_AUTO_CENTER |
				// Plus defaults.
				wxPGMAN_DEFAULT_STYLE
           );
	m_PG->AddPage(wxT("Standard Items"));

	std::vector<std::string> order = {"DEPTH_FUNC", "DEPTH_TEST", "DEPTH_MASK", "CULL_TYPE", "CULL_FACE",
		"ORDER", "BLEND", "BLEND_SRC", "BLEND_DST", "BLEND_EQUATION", "BLEND_COLOR", "COLOR_MASK_B4"};
	PropertyManager::createOrderedGrid(m_PG, nau::material::IState::Attribs, order);

	m_PG->SetSplitterLeft(true);

	sizerf->Add(m_PG,1,wxEXPAND);
	siz->Add(sizerf,1,wxGROW|wxEXPAND|wxALL,5);
}


void 
DlgMatStatePanels::resetPropGrid() {

	m_PG->Clear();
	m_PG->AddPage(wxT("Standard Items"));

	std::vector<std::string> order = { "DEPTH_FUNC", "DEPTH_TEST", "DEPTH_MASK", "CULL_TYPE", "CULL_FACE",
		"ORDER", "BLEND", "BLEND_SRC", "BLEND_DST", "BLEND_EQUATION", "BLEND_COLOR", "COLOR_MASK_B4" };
	PropertyManager::createOrderedGrid(m_PG, nau::material::IState::Attribs, order);

	m_PG->SetSplitterLeft(true);

}


void DlgMatStatePanels::OnProcessPanelChange(wxPropertyGridEvent& e){


	const wxString& name = e.GetPropertyName();

	PropertyManager::updateProp(m_PG, name.ToStdString(), nau::material::IState::Attribs, (AttributeValues *)m_glState);
}


void DlgMatStatePanels::updatePanel(){

	PropertyManager::updateGrid(m_PG, nau::material::IState::Attribs, (AttributeValues *)m_glState);
	m_PG->Refresh();
}



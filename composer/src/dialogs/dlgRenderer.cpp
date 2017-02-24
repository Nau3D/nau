#include "dialogs/dlgRenderer.h"

#include "dialogs/propertyManager.h"

#include "nau.h"
#include "nau/attribute.h"
#include "nau/event/eventFactory.h" 


BEGIN_EVENT_TABLE(DlgRenderer, wxDialog)

	EVT_PG_CHANGED( DLG_PROPS, DlgRenderer::OnPropsChange )
	EVT_BUTTON(DLG_BUTTON_UPDATE, DlgRenderer::OnUpdate)
END_EVENT_TABLE()

wxWindow *DlgRenderer::Parent = NULL;
DlgRenderer *DlgRenderer::Inst = NULL;


void DlgRenderer::SetParent(wxWindow *p) {

	Parent = p;
}


DlgRenderer* DlgRenderer::Instance () {

	if (Inst == NULL)
		Inst = new DlgRenderer();

	return Inst;
}


DlgRenderer::DlgRenderer()
	: wxDialog(DlgRenderer::Parent, -1, wxT("Nau - Renderer"),wxDefaultPosition,wxDefaultSize,wxRESIZE_BORDER|wxDEFAULT_DIALOG_STYLE)
                
{
	wxBoxSizer *sizer = new wxBoxSizer(wxVERTICAL);

	setupPanel(sizer,Parent);

	SetAutoLayout(TRUE);
	sizer->SetSizeHints(this);
	sizer->Fit(this);
    SetSizer(sizer);

	this->SetTitle(wxT("Nau - Renderer"));
	this->SetSize(350,200);
}


void
DlgRenderer::notifyUpdate(Notification aNot, std::string lightName, std::string value) {

	// sends events on behalf of renderer
}


void 
DlgRenderer::updateDlg() {

	setupGrid();
	update();
	m_parent->Refresh();	
}


void 
DlgRenderer::setupPanel(wxSizer *siz, wxWindow *parent) {

	m_PG = new wxPropertyGridManager(this, DLG_PROPS,
				wxDefaultPosition, wxDefaultSize,
				// These and other similar styles are automatically
				// passed to the embedded wxPropertyGrid.
				wxPG_BOLD_MODIFIED|wxPG_SPLITTER_AUTO_CENTER|
				// Plus defaults.
				wxPGMAN_DEFAULT_STYLE
           );

	m_PG->AddPage(wxT("Renderer"));

	m_PG->SetSplitterLeft(true);
	siz->Add(m_PG,1, wxEXPAND|wxALL,5);

	wxBoxSizer *sizH3 = new wxBoxSizer(wxHORIZONTAL);

	m_BUpdate = new wxButton(this, DLG_BUTTON_UPDATE, wxT("Update"));
	sizH3->Add(m_BUpdate, 0, wxALL | wxGROW | wxHORIZONTAL, 5);

	siz->Add(sizH3, 0, wxALL | wxGROW | wxHORIZONTAL, 15);
}


void
DlgRenderer::setupGrid() {

	m_PG->Clear();
	m_PG->AddPage(wxT("properties"));

	std::vector<std::string> order = {};

	PropertyManager::createOrderedGrid(m_PG, IRenderer::GetAttribs(), order);
	m_PG->SetSplitterLeft(true, true);
}


void 
DlgRenderer::update() {

	m_PG->ClearSelection();

	PropertyManager::updateGrid(m_PG, IRenderer::GetAttribs(), (AttributeValues *)RENDERER);

}



void 
DlgRenderer::OnPropsChange( wxPropertyGridEvent& e) {

	const wxString& name = e.GetPropertyName();
	unsigned int dotLocation = name.find_first_of(wxT("."),0);
	std::string topProp = std::string(name.substr(0,dotLocation).mb_str());

	PropertyManager::updateProp(m_PG, topProp, IRenderer::GetAttribs(), (AttributeValues *)RENDERER);
	notifyUpdate(PROPS_CHANGED,"",topProp);
}


void 
DlgRenderer::updateInfo(std::string name) {

	update();
}


void 
DlgRenderer::OnUpdate(wxCommandEvent& event) {

	update();
}
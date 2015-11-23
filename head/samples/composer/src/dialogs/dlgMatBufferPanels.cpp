#include "dlgMatBufferPanels.h"

#include "propertyManager.h"

#include <nau/material/iBuffer.h>


DlgMatBufferPanels::DlgMatBufferPanels() {
	
}


DlgMatBufferPanels::~DlgMatBufferPanels(){

}


void 
DlgMatBufferPanels::setMaterial(std::shared_ptr<nau::material::Material> &aMat) {

	m_Material = aMat;

	if (!aMat)
		return;

	m_MaterialBindings.clear();
	m_Material->getBufferBindings(&m_MaterialBindings);
	if (m_MaterialBindings.size() == 0) {
		pg->Hide();
		itemList->Clear();
		itemList->Disable();
	}
	else {
		itemList->Enable();
		itemList->SetSelection(0);
		m_CurrentBinding = 0;
		updatePanel();
		pg->Show();
	}
}


void 
DlgMatBufferPanels::setPanel(wxSizer *siz, wxWindow *parent) {


	// Item List
	wxStaticBox *sl = new wxStaticBox(parent,-1,wxT("Buffers"));
	wxSizer *sizerL = new wxStaticBoxSizer(sl,wxVERTICAL);
	wxBoxSizer *sizH1 = new wxBoxSizer(wxHORIZONTAL);
	wxStaticText *stg1 =  new wxStaticText(parent,-1,wxT("Binding Point: "));
	itemList = new wxComboBox(parent,DLG_ITEM_COMBO,wxT(""),wxDefaultPosition,wxDefaultSize,0,NULL,wxCB_READONLY );

	sizH1->Add(stg1, 5, wxGROW|wxHORIZONTAL,5);
	sizH1->Add(itemList, 5, wxGROW|wxHORIZONTAL,5);
	sizerL->Add(sizH1,1, wxEXPAND|wxALL|wxALIGN_CENTER,5);


	wxStaticBox *sf2 = new wxStaticBox(parent,-1,wxT("Buffer Props"));
	wxSizer *sizerf = new wxStaticBoxSizer(sf2,wxVERTICAL);;

	// Item properties
	pg = new wxPropertyGridManager(parent, PG,
				wxDefaultPosition, wxDefaultSize,
				// These and other similar styles are automatically
				// passed to the embedded wxPropertyGrid.
				wxPG_BOLD_MODIFIED | wxPG_SPLITTER_AUTO_CENTER |
				// Plus defaults.
				wxPGMAN_DEFAULT_STYLE
           );
	pg->AddPage(wxT("Standard Items"));

	std::vector<std::string> order2 = {};
	PropertyManager::createOrderedGrid(pg, nau::material::IMaterialBuffer::Attribs, order2);

	std::vector<std::string> order = {"ID", "SIZE", "CLEAR"};
	PropertyManager::createOrderedGrid(pg, nau::material::IBuffer::Attribs, order);

	PropertyManager::setAllReadOnly(pg, nau::material::IBuffer::Attribs);
	PropertyManager::setAllReadOnly(pg, nau::material::IMaterialBuffer::Attribs);

	pg->SetSplitterLeft(true);

	sizerf->Add(pg,1,wxEXPAND);

	siz->Add(sizerL,0,wxGROW|wxEXPAND|wxALL,5);
	siz->Add(sizerf,1,wxGROW|wxEXPAND|wxALL,5);
}


void 
DlgMatBufferPanels::onProcessPanelChange(wxPropertyGridEvent& e) {

	nau::material::IMaterialBuffer *buffer;

	buffer = m_Material->getBuffer(m_CurrentBinding);

	const wxString& name = e.GetPropertyName();
	PropertyManager::updateProp(pg, name.ToStdString(), nau::material::IMaterialBuffer::Attribs, (AttributeValues *)buffer);
	PropertyManager::updateProp(pg, name.ToStdString(), nau::material::IBuffer::Attribs, (AttributeValues *)buffer->getBuffer());
}


void 
DlgMatBufferPanels::updatePanel() {

	nau::material::IMaterialBuffer *buffer;

	itemList->Clear();
	for(auto i:m_MaterialBindings)  {
		wxString s;
		s << i;
		itemList->Append(wxString(s));
	}
	itemList->SetSelection(m_CurrentBinding);

	buffer = m_Material->getBuffer(m_MaterialBindings[m_CurrentBinding]);

	PropertyManager::updateGrid(pg, nau::material::IMaterialBuffer::Attribs, (AttributeValues *)buffer);
	PropertyManager::updateGrid(pg, nau::material::IBuffer::Attribs, (AttributeValues *)buffer->getBuffer());
	pg->Refresh();
}


void
DlgMatBufferPanels::onItemListSelect(wxCommandEvent& event) {

	int sel = event.GetSelection();

	m_CurrentBinding = sel;
	updatePanel();
}
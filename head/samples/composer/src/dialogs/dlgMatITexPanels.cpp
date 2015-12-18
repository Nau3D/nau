#include "dlgMatITexPanels.h"

#include "propertyManager.h"

#include <nau/material/iImageTexture.h>


DlgMatImageTexturePanels::DlgMatImageTexturePanels() {
	
}


DlgMatImageTexturePanels::~DlgMatImageTexturePanels(){

}


void 
DlgMatImageTexturePanels::setMaterial(std::shared_ptr<nau::material::Material> &aMat) {

	m_Material = aMat;

	if (!aMat)
		return;

	m_ImageTextureUnits.clear();
	m_Material->getImageTextureUnits(&m_ImageTextureUnits);
	if (m_ImageTextureUnits.size() == 0) {
		pg->Hide();
		itemList->Clear();
		itemList->Disable();
	}
	else {
		itemList->Enable();
		itemList->SetSelection(0);
		m_CurrentUnit = 0;
		updatePanel();
		pg->Show();
	}
}


void 
DlgMatImageTexturePanels::setPanel(wxSizer *siz, wxWindow *parent) {


	// Item List
	wxStaticBox *sl = new wxStaticBox(parent,-1,wxT("Image Textures"));
	wxSizer *sizerL = new wxStaticBoxSizer(sl,wxVERTICAL);
	wxBoxSizer *sizH1 = new wxBoxSizer(wxHORIZONTAL);
	wxStaticText *stg1 =  new wxStaticText(parent,-1,wxT("Unit: "));
	itemList = new wxComboBox(parent,DLG_ITEM_COMBO,wxT(""),wxDefaultPosition,wxDefaultSize,0,NULL,wxCB_READONLY );

	sizH1->Add(stg1, 5, wxGROW|wxHORIZONTAL,5);
	sizH1->Add(itemList, 5, wxGROW|wxHORIZONTAL,5);
	sizerL->Add(sizH1,1, wxEXPAND|wxALL|wxALIGN_CENTER,5);


	wxStaticBox *sf2 = new wxStaticBox(parent,-1,wxT("Image Texture Props"));
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

	std::vector<std::string> order = {"UNIT", "TEX_ID"};
	PropertyManager::createOrderedGrid(pg, nau::material::IImageTexture::Attribs, order);
	PropertyManager::setAllReadOnly(pg, nau::material::IImageTexture::Attribs);

	pg->SetSplitterLeft(true);

	sizerf->Add(pg,1,wxEXPAND);

	siz->Add(sizerL,0,wxGROW|wxEXPAND|wxALL,5);
	siz->Add(sizerf,1,wxGROW|wxEXPAND|wxALL,5);
}


void 
DlgMatImageTexturePanels::resetPropGrid() {

	pg->Clear();
	pg->AddPage(wxT("Standard Items"));

	std::vector<std::string> order = { "UNIT", "TEX_ID" };
	PropertyManager::createOrderedGrid(pg, nau::material::IImageTexture::Attribs, order);
	PropertyManager::setAllReadOnly(pg, nau::material::IImageTexture::Attribs);

	pg->SetSplitterLeft(true);

}


void 
DlgMatImageTexturePanels::onProcessPanelChange(wxPropertyGridEvent& e) {

	nau::material::IImageTexture *it;

	it = m_Material->getImageTexture(m_CurrentUnit);

	const wxString& name = e.GetPropertyName();
	PropertyManager::updateProp(pg, name.ToStdString(), nau::material::IImageTexture::Attribs, (AttributeValues *)it);
}


void 
DlgMatImageTexturePanels::updatePanel() {

	nau::material::IImageTexture *it;

	itemList->Clear();
	for(auto i:m_ImageTextureUnits)  {
		wxString s;
		s << i;
		itemList->Append(wxString(s));
	}
	itemList->SetSelection(m_CurrentUnit);

	it = m_Material->getImageTexture(m_ImageTextureUnits[m_CurrentUnit]);

	PropertyManager::updateGrid(pg, nau::material::IImageTexture::Attribs, (AttributeValues *)it);

	pg->Refresh();
}


void
DlgMatImageTexturePanels::onItemListSelect(wxCommandEvent& event) {

	int sel = event.GetSelection();

	m_CurrentUnit = sel;
	updatePanel();
}
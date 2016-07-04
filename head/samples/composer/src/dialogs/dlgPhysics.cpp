#include "dialogs/dlgPhysics.h"

#include "dialogs/propertyManager.h"

#include <nau/attribute.h>
#include <nau/attributeValues.h>
#include <nau/event/eventFactory.h>
#include <nau/math/utils.h>

using namespace nau::physics;

BEGIN_EVENT_TABLE(DlgPhysics, wxDialog)
	EVT_COMBOBOX(DLG_COMBO, DlgPhysics::OnListSelect)
	//EVT_PG_CHANGED( DLG_PROPS_GLOBAL, DlgPhysics::OnGlobalPropsChange )
	EVT_PG_CHANGED(DLG_PROPS_MAT, DlgPhysics::OnMaterialPropsChange)
	END_EVENT_TABLE()


wxWindow *DlgPhysics::Parent = NULL;
DlgPhysics *DlgPhysics::Inst = NULL;


void 
DlgPhysics::SetParent(wxWindow *p) {

	Parent = p;
}


DlgPhysics* DlgPhysics::Instance () {

	if (Inst == NULL)
		Inst = new DlgPhysics();

	return Inst;
}


DlgPhysics::DlgPhysics()
	: wxDialog(DlgPhysics::Parent, -1, wxT("Nau - Physics"),wxDefaultPosition,wxDefaultSize,wxRESIZE_BORDER|wxDEFAULT_DIALOG_STYLE),
	m_Name("Physics Dialog")
                
{
	wxBoxSizer *sizer = new wxBoxSizer(wxVERTICAL);

	setupPanel(sizer,Parent);

	SetAutoLayout(TRUE);
    sizer->SetSizeHints(this);
    sizer->Fit(this);
    SetSizer(sizer);

	this->SetTitle(wxT("Nau - Physics"));
	this->SetSize(350,400);
}


std::string &
DlgPhysics::getName() {

	return m_Name;
}


void 
DlgPhysics::updateDlg() {

	setupGrid();
	updateList();
	update();
	m_parent->Refresh();	
}


void 
DlgPhysics::setupPanel(wxSizer *siz, wxWindow *parent) {

	/* TOP: COMBO with material names */

	wxStaticBox *sbG = new wxStaticBox(this, -1, wxT(" Global Properties "));

	wxBoxSizer *sizerGlobal = new wxStaticBoxSizer(sbG, wxVERTICAL);

	/* MIDDLE: Property grid */

	m_PGGlobal = new wxPropertyGridManager(this, DLG_PROPS_MAT,
		wxDefaultPosition, wxDefaultSize,
		// These and other similar styles are automatically
		// passed to the embedded wxPropertyGrid.
		wxPG_BOLD_MODIFIED | wxPG_SPLITTER_AUTO_CENTER |
		// Plus defaults.
		wxPGMAN_DEFAULT_STYLE
		);

	m_PGGlobal->AddPage(wxT("properties"));
	sizerGlobal->Add(m_PGGlobal, 1, wxEXPAND, 5);


	wxStaticBox *sb = new wxStaticBox(this, -1, wxT(" Material Properties "));

	wxBoxSizer *sizerMat = new wxStaticBoxSizer(sb, wxVERTICAL);

	wxSizer *sizerH = new wxBoxSizer(wxHORIZONTAL); 

	wxStaticText *stg1 =  new wxStaticText(this,-1,wxT("Material: "));
	m_List = new wxComboBox(this,DLG_COMBO,wxT(""),wxDefaultPosition,wxDefaultSize,0,NULL,wxCB_READONLY );
	updateList();

	sizerH->Add(stg1,0, wxGROW | wxHORIZONTAL, 5);
	sizerH->Add(m_List, 1, wxGROW | wxHORIZONTAL, 5);

	wxSizer *sizerH2 = new wxBoxSizer(wxHORIZONTAL);

	/* MIDDLE: Property grid */

	m_PGMat = new wxPropertyGridManager(this, DLG_PROPS_MAT,
				wxDefaultPosition, wxDefaultSize,
				// These and other similar styles are automatically
				// passed to the embedded wxPropertyGrid.
				wxPG_BOLD_MODIFIED|wxPG_SPLITTER_AUTO_CENTER|
				// Plus defaults.
				wxPGMAN_DEFAULT_STYLE
           );

	m_PGMat->AddPage(wxT("properties"));

	//m_PGMat->Append(new wxBoolProperty(wxT("dummy"), wxPG_LABEL));

	sizerH2->Add(m_PGMat, 1, wxEXPAND , 5);

	sizerMat->Add(sizerH, 0, wxEXPAND | wxLEFT | wxRIGHT, 5);
	sizerMat->Add(sizerH2, 1, wxEXPAND | wxALL, 5);

	siz->Add(sizerGlobal, 1, wxEXPAND | wxALL, 5);
	siz->Add(sizerMat, 1, wxEXPAND | wxALL, 5);

}


void
DlgPhysics::setupGrid() {

	m_PGMat->Clear();
	m_PGMat->AddPage(wxT("properties"));
	std::vector<std::string> order = { "SCENE_TYPE" };
	PropertyManager::createOrderedGrid(m_PGMat, PhysicsMaterial::Attribs, order);
	m_PGMat->SetSplitterLeft(true, true);
	m_PGGlobal->Clear();
	m_PGGlobal->AddPage(wxT("properties"));
	PropertyManager::createGrid(m_PGGlobal, PhysicsManager::Attribs);
}


void 
DlgPhysics::updateList() {

	std::vector<std::string> names;
	PhysicsManager::GetInstance()->getMaterialNames(&names);
	int num = names.size();

	m_List->Clear();

	if (names.size()) {

		for (int i = 0; i < num; i++) {
			wxString s;
			s << i;
			m_List->Append(wxString(names[i].c_str()));
		}
		m_Active = names[0];
		m_List->SetSelection(0);
		m_List->Enable();
	}
	else {

		m_List->Disable();
		m_Active = "";
	}
}


void 
DlgPhysics::update() {

	if (m_Active != "") {

		PhysicsMaterial &mat = PhysicsManager::GetInstance()->getMaterial(m_Active);
		m_PGMat->ClearModifiedStatus();
		PropertyManager::updateGrid(m_PGMat, PhysicsMaterial::Attribs, (AttributeValues *)&mat);
		wxPGProperty *p = m_PGMat->GetPropertyByName("SCENE_TYPE");
		p->Enable(false);
		
		m_PGMat->Refresh();

		PropertyManager::updateGrid(m_PGGlobal, PhysicsManager::Attribs, (AttributeValues *)PhysicsManager::GetInstance());
	}
}


void 
DlgPhysics::OnMaterialPropsChange( wxPropertyGridEvent& e) {

	PhysicsMaterial &mat = PhysicsManager::GetInstance()->getMaterial(m_Active);

	const wxString& name = e.GetPropertyName();
	unsigned int dotLocation = name.find_first_of(wxT("."),0);
	std::string topProp = std::string(name.substr(0,dotLocation).mb_str());
	std::string prop = std::string(name.substr(dotLocation+1,name.size()-dotLocation-1).mb_str());

	PropertyManager::updateProp(m_PGMat, topProp, PhysicsMaterial::Attribs, (AttributeValues *)&mat);

	update();
}


void 
DlgPhysics::updateInfo(std::string name) {

	if (name == m_Active) {
		update();
	}
}


void 
DlgPhysics::OnListSelect(wxCommandEvent& event){

	int sel;

	sel = event.GetSelection();
	m_Active = std::string(m_List->GetString(sel).mb_str());
	update();
}

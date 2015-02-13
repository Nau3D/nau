#include "dialogs/dlgViewports.h"

#include "dialogs/propertyManager.h"

#include "nau.h"
#include "nau/attribute.h"
#include "nau/attributeValues.h"
#include "nau/event/eventFactory.h"

#include "nau/math/utils.h"
#include "nau/render/viewport.h"



BEGIN_EVENT_TABLE(DlgViewports, wxDialog)
EVT_COMBOBOX(DLG_COMBO, DlgViewports::OnListSelect)
EVT_PG_CHANGED(DLG_PROPS, DlgViewports::OnPropsChange)
EVT_BUTTON(DLG_BUTTON_ADD, DlgViewports::OnAdd)
END_EVENT_TABLE()


wxWindow *DlgViewports::Parent = NULL;
DlgViewports *DlgViewports::Inst = NULL;


void DlgViewports::SetParent(wxWindow *p) {

	Parent = p;
}


DlgViewports* DlgViewports::Instance() {

	if (Inst == NULL)
		Inst = new DlgViewports();

	return Inst;
}



DlgViewports::DlgViewports()
	: wxDialog(DlgViewports::Parent, -1, wxT("Nau - Viewports"), wxDefaultPosition, wxDefaultSize, wxRESIZE_BORDER | wxDEFAULT_DIALOG_STYLE),
	m_Name("Viewport Dialog")
                
{
	wxBoxSizer *sizer = new wxBoxSizer(wxVERTICAL);

	setupPanel(sizer,Parent);

	SetAutoLayout(TRUE);
    sizer->SetSizeHints(this);
    sizer->Fit(this);
    SetSizer(sizer);
	this->SetSize(350, 400);
}


void
DlgViewports::eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt) {

	if (eventType == "NEW_VIEWPORT") 
		updateList();
	else if (eventType == "VIEWPORT_CHANGED")
		update();
}


std::string &
DlgViewports::getName() {

	return m_Name;
}


void
DlgViewports::notifyUpdate(Notification aNot, std::string vpName, std::string value) {

	// sends events on behalf of the viewport
	nau::event_::IEventData *e= nau::event_::EventFactory::create("String");
	if (aNot == NEW_VIEWPORT) {
		e->setData(&value);
		EVENTMANAGER->notifyEvent("NEW_VIEWPORT", vpName,"", e);
	}
	else  {
		e->setData(&value);
		EVENTMANAGER->notifyEvent("VIEWPORT_CHANGED", vpName, "", e);
	}
	delete e;
}


void DlgViewports::updateDlg() {

	EVENTMANAGER->addListener("NEW_VIEWPORT", this);
	EVENTMANAGER->addListener("VIEWPORT_CHANGED", this);

	setupGrid();
	updateList();
	m_List->SetSelection(0);
	update();
	m_parent->Refresh();	
}


void DlgViewports::setupPanel(wxSizer *siz, wxWindow *Parent) {

	/* TOP: COMBO with viewport names */
	wxBoxSizer *sizH1 = new wxBoxSizer(wxHORIZONTAL);

	wxStaticText *stg1 =  new wxStaticText(this,-1,wxT("Viewport: "));
	m_List = new wxComboBox(this,DLG_COMBO,wxT(""),wxDefaultPosition,wxDefaultSize,0,NULL,wxCB_READONLY );

	updateList();
	m_List->SetSelection(0);

	sizH1->Add(stg1, 0, wxGROW|wxHORIZONTAL,5);
	sizH1->Add(m_List, 1, wxGROW|wxHORIZONTAL,5);
	siz->Add(sizH1, 0, wxGROW|wxALL, 15);

	/* MIDDLE: Property grid */

	m_PG = new wxPropertyGridManager(this, DLG_PROPS,
				wxDefaultPosition, wxDefaultSize,
				// These and other similar styles are automatically
				// passed to the embedded wxPropertyGrid.
				wxPG_BOLD_MODIFIED|
				// Plus defaults.
				wxPGMAN_DEFAULT_STYLE
           );

	m_PG->AddPage(wxT("properties"));
	m_PG->Append(new wxBoolProperty(wxT("dummy"), wxPG_LABEL));

	siz->Add(m_PG,1, wxEXPAND|wxALL,5);

	/* BOTTOM: Add Element Button */

	wxBoxSizer *sizH3 = new wxBoxSizer(wxHORIZONTAL);

	m_BAdd = new wxButton(this,DLG_BUTTON_ADD,wxT("Add Viewport"));
	sizH3-> Add(m_BAdd,0,wxALL |wxGROW |wxHORIZONTAL|wxALIGN_CENTER_HORIZONTAL,5);

	siz->Add(sizH3,0,wxALL |wxGROW |wxHORIZONTAL,15);
}


void
DlgViewports::setupGrid() {

	m_PG->Clear();
	m_PG->AddPage(wxT("properties"));

	PropertyManager::createGrid(m_PG, Viewport::Attribs);
	m_PG->SetSplitterLeft(true, true);
}


void 
DlgViewports::updateList() {

	std::vector<std::string> *names = NAU->getViewportNames();
	int num = names->size();

	m_List->Clear();

	for(int i = 0; i < num; i++)  {
		wxString s;
		s << i;
		m_List->Append(wxString(names->at(i).c_str()));
	}
	m_Active = names->at(0);
	delete names;
}


void DlgViewports::update() {

	nau::render::Viewport *elem;		
	elem = NAU->getViewport(m_Active);

	m_PG->ClearModifiedStatus();

	PropertyManager::updateGrid(m_PG, Viewport::Attribs, (AttributeValues *)elem);

}


void DlgViewports::OnPropsChange(wxPropertyGridEvent& e) {

	nau::render::Viewport *elem = NAU->getViewport(m_Active);
	const wxString& name = e.GetPropertyName();
	unsigned int dotLocation = name.find_first_of(wxT("."),0);
	std::string topProp = std::string(name.substr(0,dotLocation).mb_str());
	std::string prop = std::string(name.substr(dotLocation+1,name.size()-dotLocation-1).mb_str());

	PropertyManager::updateProp(m_PG, name.ToStdString(), Viewport::Attribs, (AttributeValues *)elem);
	notifyUpdate(PROPS_CHANGED,m_Active,topProp);
}


void DlgViewports::updateInfo(std::string name) {

	if (name == m_Active) {
		update();
	}
}


void DlgViewports::OnAdd(wxCommandEvent& event) {

	int result;
	bool nameUnique,exit = false;
	std::string name;

	do {
		wxTextEntryDialog dialog(this,
								 _T("(the name must be unique)\n")
								 _T("Please Input a Viewport Name"),
								 _T("Viewport Name\n"),
								 _T(""),
								 wxOK | wxCANCEL);

		result = dialog.ShowModal();
		name = std::string(dialog.GetValue().mb_str());
		nameUnique =  !NAU->hasViewport(name); 

		if (!nameUnique && (result == wxID_OK)){
			wxMessageBox(_T("Viewport name must be unique") , _T("Viewport Name Error"), wxOK | wxICON_INFORMATION, this);
		}
		if (name == "" && (result == wxID_OK)){
			wxMessageBox(_T("Viewport name must not be void"), _T("Viewport Name Error"), wxOK | wxICON_INFORMATION, this);
		}

		if (result == wxID_CANCEL) {
			exit = true;
		}
		else if (nameUnique && name != "") {
			exit = true;
		}

	} while (!exit);

	if (result == wxID_OK) {
		NAU->createViewport(name);
		updateList();
		m_List->Select(m_List->FindString((wxString)name.c_str()));
		m_Active = name;
		update();
		notifyUpdate(NEW_VIEWPORT,name,"");
	}
}


void DlgViewports::OnListSelect(wxCommandEvent& event) {

	int sel;

	sel = event.GetSelection();
	m_Active = std::string(m_List->GetString(sel).mb_str());
	update();
}

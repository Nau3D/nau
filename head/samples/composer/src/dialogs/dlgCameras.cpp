#include "dialogs/dlgCameras.h"

#include "dialogs/propertyManager.h"

#include <nau/attribute.h>
#include <nau/attributeValues.h>
#include <nau/event/eventFactory.h>
#include <nau/math/utils.h>


BEGIN_EVENT_TABLE(DlgCameras, wxDialog)
	EVT_COMBOBOX(DLG_COMBO, DlgCameras::OnListSelect)
	EVT_PG_CHANGED( DLG_PROPS, DlgCameras::OnPropsChange )
	EVT_BUTTON(DLG_BUTTON_ADD, DlgCameras::OnAdd)
	EVT_BUTTON(DLG_BUTTON_ACTIVATE, DlgCameras::OnActivate)
END_EVENT_TABLE()


wxWindow *DlgCameras::Parent = NULL;
DlgCameras *DlgCameras::Inst = NULL;


void 
DlgCameras::SetParent(wxWindow *p) {

	Parent = p;
}


DlgCameras* DlgCameras::Instance () {

	if (Inst == NULL)
		Inst = new DlgCameras();

	return Inst;
}


DlgCameras::DlgCameras()
	: wxDialog(DlgCameras::Parent, -1, wxT("Nau - Cameras"),wxDefaultPosition,wxDefaultSize,wxRESIZE_BORDER|wxDEFAULT_DIALOG_STYLE),
	m_Name("Camera Dialog")
                
{
	wxBoxSizer *sizer = new wxBoxSizer(wxVERTICAL);

	setupPanel(sizer,Parent);

	SetAutoLayout(TRUE);
    sizer->SetSizeHints(this);
    sizer->Fit(this);
    SetSizer(sizer);

	this->SetTitle(wxT("Nau - Cameras"));
	this->SetSize(350,400);
}


void
DlgCameras::eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt) {

	if (eventType == "NEW_VIEWPORT")
		updateList();
	else if (eventType == "VIEWPORT_CHANGED")
		update();
}


void
DlgCameras::notifyUpdate(Notification aNot, std::string camName, std::string value) {

	// sends events on behalf of the camera
	nau::event_::IEventData *e= nau::event_::EventFactory::create("String");
	if (aNot == NEW_CAMERA) {
		e->setData(&value);
		EVENTMANAGER->notifyEvent("NEW_CAMERA", camName,"", e);
	}
	else  {
		e->setData(&value);
		EVENTMANAGER->notifyEvent("CAMERA_CHANGED", camName ,"", e);
	}
	delete e;
}


std::string &
DlgCameras::getName() {

	return m_Name;
}


void 
DlgCameras::updateDlg() {

	EVENTMANAGER->addListener("NEW_VIEWPORT", this);
	EVENTMANAGER->addListener("VIEWPORT_CHANGED", this);
	setupGrid();
	updateList();
	updateViewportLabels();
	m_List->SetSelection(0);
	update();
	m_parent->Refresh();	
}


void 
DlgCameras::updateViewportLabels() {

	m_ViewportLabels.RemoveAt(0, m_ViewportLabels.GetCount());
	m_ViewportLabels.Add(wxT("None"), -1);

	std::vector<std::string> viewports;
	RENDERMANAGER->getViewportNames(&viewports);
	for (auto &v:viewports)
		m_ViewportLabels.Add(wxString(v.c_str()));

	wxPGProperty *p = m_PG->GetProperty(wxT("Viewport"));
	p->SetChoices(m_ViewportLabels);
}


void 
DlgCameras::setupPanel(wxSizer *siz, wxWindow *parent) {

	/* TOP: COMBO with camera names */
	wxBoxSizer *sizH1 = new wxBoxSizer(wxHORIZONTAL);

	wxStaticText *stg1 =  new wxStaticText(this,-1,wxT("Camera: "));
	m_List = new wxComboBox(this,DLG_COMBO,wxT(""),wxDefaultPosition,wxDefaultSize,0,NULL,wxCB_READONLY );

	updateList();
	m_List->SetSelection(0);

	sizH1->Add(stg1, 0, wxGROW|wxHORIZONTAL,5);
	sizH1->Add(m_List, 1, wxGROW | wxHORIZONTAL, 5);
	siz->Add(sizH1, 0, wxGROW|wxALL, 15);

	/* MIDDLE: Property grid */

	m_PG = new wxPropertyGridManager(this, DLG_PROPS,
				wxDefaultPosition, wxDefaultSize,
				// These and other similar styles are automatically
				// passed to the embedded wxPropertyGrid.
				wxPG_BOLD_MODIFIED|wxPG_SPLITTER_AUTO_CENTER|
				// Plus defaults.
				wxPGMAN_DEFAULT_STYLE
           );

	m_PG->AddPage(wxT("properties"));

	m_PG->Append(new wxBoolProperty(wxT("dummy"), wxPG_LABEL));

	siz->Add(m_PG, 1, wxEXPAND | wxALL, 5);

	/* BOTTOM: Add Camera Button */

	wxBoxSizer *sizH3 = new wxBoxSizer(wxHORIZONTAL);

	m_BAdd = new wxButton(this, DLG_BUTTON_ADD, wxT("Add Camera"));
	m_BActivate = new wxButton(this,DLG_BUTTON_ACTIVATE,wxT("Activate Camera"));
	sizH3->Add(m_BAdd, 0, wxALL | wxGROW | wxHORIZONTAL | wxALIGN_CENTER_HORIZONTAL, 5);
	sizH3->Add(m_BActivate, 0, wxALL | wxGROW | wxHORIZONTAL | wxALIGN_CENTER_HORIZONTAL, 5);

	siz->Add(sizH3,0,wxALL |wxGROW |wxHORIZONTAL,15);
}


void
DlgCameras::setupGrid() {

	m_PG->Clear();
	m_PG->AddPage(wxT("properties"));

	m_ViewportLabels.Add(wxT("None"));
	m_PG->Append(new wxEnumProperty(wxT("Viewport"), wxPG_LABEL, m_ViewportLabels));

	std::vector<std::string> order = { "POSITION", "VIEW", "UP", "LOOK_AT_POINT", "TYPE",
		"FOV", "NEAR", "FAR", "LEFT", "RIGHT", "BOTTOM", "TOP" };

	PropertyManager::createOrderedGrid(m_PG, Camera::Attribs, order);
	m_PG->SetSplitterLeft(true, true);

}


void 
DlgCameras::updateList() {

	std::vector<std::string> names;
	RENDERMANAGER->getCameraNames(&names);
	int num = names.size();

	m_List->Clear();

	for(int i = 0; i < num; i++)  {
		wxString s;
		s << i;
		m_List->Append(wxString(names[i].c_str()));
	}
	m_Active = names[0];
}


void DlgCameras::update() {

	std::shared_ptr<nau::scene::Camera> &cam = RENDERMANAGER->getCamera(m_Active);

	m_PG->ClearModifiedStatus();

	PropertyManager::updateGrid(m_PG, Camera::Attribs, (AttributeValues *)cam.get());

	updateViewportLabels();
	std::shared_ptr<Viewport> vp = cam->getViewport();
	std::string vpname;
	if (vp)
		vpname = vp->getName();
	else
		vpname = "None";
	m_PG->SetPropertyValueString(wxT("Viewport"), wxString(vpname.c_str()));

	m_PG->Refresh();
}


void DlgCameras::OnPropsChange( wxPropertyGridEvent& e) {

	std::shared_ptr<nau::scene::Camera> &cam = RENDERMANAGER->getCamera(m_Active);
	const wxString& name = e.GetPropertyName();
	unsigned int dotLocation = name.find_first_of(wxT("."),0);
	std::string topProp = std::string(name.substr(0,dotLocation).mb_str());
	std::string prop = std::string(name.substr(dotLocation+1,name.size()-dotLocation-1).mb_str());

	if (prop == "Viewport") {
		int index = e.GetPropertyValue().GetInteger();
		if (index != -1) {
			wxString s = m_ViewportLabels.GetLabel(index);
			cam->setViewport(RENDERMANAGER->getViewport(s.ToStdString()));
		}
		else
			cam->setViewport(NULL);
	}
	else
		PropertyManager::updateProp(m_PG, topProp, Camera::Attribs, (AttributeValues *)cam.get());

	notifyUpdate(PROPS_CHANGED,m_Active,topProp);
	update();
}


void DlgCameras::updateInfo(std::string name) {

	if (name == m_Active) {
		update();
	}
}


void DlgCameras::OnActivate(wxCommandEvent& event) {

	NAU->setActiveCameraName(m_Active);
}    


void DlgCameras::OnAdd(wxCommandEvent& event) {

	int result;
	bool nameUnique,exit = false;
	std::string name;

	do {
		wxTextEntryDialog dialog(this,
								 _T("(the name must be unique)\n")
								 _T("Please Input a Camera Name"),
								 _T("Camera Name\n"),
								 _T(""),
								 wxOK | wxCANCEL);

		result = dialog.ShowModal();
		name = std::string(dialog.GetValue().mb_str());
		nameUnique =  !RENDERMANAGER->hasCamera(name); 

		if (!nameUnique && (result == wxID_OK)){
			wxMessageBox(_T("Camera name must be unique") , _T("Camera Name Error"), wxOK | wxICON_INFORMATION, this);
		}
		if (name == "" && (result == wxID_OK)){
			wxMessageBox(_T("Camera name must not be void"), _T("Camera Name Error"), wxOK | wxICON_INFORMATION, this);
		}

		if (result == wxID_CANCEL) {
			exit = true;
		}
		else if (nameUnique && name != "") {
			exit = true;
		}

	} while (!exit);

	if (result == wxID_OK) {
		RENDERMANAGER->getCamera(name);
		updateList();
		m_List->Select(m_List->FindString((wxString)name.c_str()));
		m_Active = name;
		update();
		notifyUpdate(NEW_CAMERA,name,"");
	}
}


void DlgCameras::OnListSelect(wxCommandEvent& event){

	int sel;

	sel = event.GetSelection();
	m_Active = std::string(m_List->GetString(sel).mb_str());
	update();
//	parent->Refresh();
}

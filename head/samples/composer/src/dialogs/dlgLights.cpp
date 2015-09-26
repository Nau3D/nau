#include "dialogs/dlgLights.h"

#include "dialogs/propertyManager.h"

#include "nau.h"
#include "nau/attribute.h"
#include "nau/event/eventFactory.h" 


BEGIN_EVENT_TABLE(DlgLights, wxDialog)

	EVT_COMBOBOX(DLG_COMBO, DlgLights::OnListSelect)
	EVT_PG_CHANGED( DLG_PROPS, DlgLights::OnPropsChange )
	EVT_BUTTON(DLG_BUTTON_ADD, DlgLights::OnAdd)
	
END_EVENT_TABLE()

wxWindow *DlgLights::Parent = NULL;
DlgLights *DlgLights::Inst = NULL;


void DlgLights::SetParent(wxWindow *p) {

	Parent = p;
}

DlgLights* DlgLights::Instance () {

	if (Inst == NULL)
		Inst = new DlgLights();

	return Inst;
}


DlgLights::DlgLights()
	: wxDialog(DlgLights::Parent, -1, wxT("Nau - Lights"),wxDefaultPosition,wxDefaultSize,wxRESIZE_BORDER|wxDEFAULT_DIALOG_STYLE)
                
{
	wxBoxSizer *sizer = new wxBoxSizer(wxVERTICAL);

	setupPanel(sizer,Parent);

	SetAutoLayout(TRUE);
	sizer->SetSizeHints(this);
	sizer->Fit(this);
    SetSizer(sizer);

	this->SetTitle(wxT("Nau - Lights"));
	this->SetSize(350,200);
}


void
DlgLights::notifyUpdate(Notification aNot, std::string lightName, std::string value) {

	// sends events on behalf of the light
	nau::event_::IEventData *e= nau::event_::EventFactory::create("String");
	if (aNot == NEW_LIGHT) {
		e->setData(&lightName);
		EVENTMANAGER->notifyEvent("NEW_LIGHT", lightName,"", e);
	}
	else  {
		e->setData(&value);
		EVENTMANAGER->notifyEvent("LIGHT_CHANGED", lightName ,"", e);
	}
	delete e;
}


void 
DlgLights::updateDlg() {

	setupGrid();
	if (RENDERMANAGER->getNumLights()) {
		updateList();
		m_List->SetSelection(0);
		update();
		m_parent->Refresh();	
	}
	else {
		m_PG->Disable();
		m_List->Disable();
	}


}

void 
DlgLights::setupPanel(wxSizer *siz, wxWindow *parent) {

	int count = RENDERMANAGER->getNumLights();

	/* TOP: COMBO with camera names */
	wxBoxSizer *sizH1 = new wxBoxSizer(wxHORIZONTAL);

	wxStaticText *stg1 =  new wxStaticText(this,-1,wxT("Light: "));
	m_List = new wxComboBox(this,DLG_COMBO,wxT(""),wxDefaultPosition,wxDefaultSize,0,NULL,wxCB_READONLY );

	if (count) {
		updateList();
		m_List->SetSelection(0);
	}
	sizH1->Add(stg1, 0, wxGROW|wxHORIZONTAL,5);
	sizH1->Add(m_List, 1, wxGROW|wxHORIZONTAL,5);
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

	m_PG->AddPage(wxT("Lights"));
	//wxPropertyGridPage* pgPropsPage = m_PG->GetPage("Cameras");

//	const wxChar* lightType[] = { wxT("DIRECTIONAL"), wxT("POINT"), wxT("SPOT"), NULL};
//	const long lightTypeInd[] = { Light::DIRECTIONAL, Light::POSITIONAL, Light::SPOT_LIGHT};

	//m_PG->Append(new wxStringProperty(wxT("Name"),wxPG_LABEL,""));
 	//m_PG->DisableProperty(wxT("Name"));

//	m_PG->Append(new wxEnumProperty(wxT("TYPE"),wxPG_LABEL,lightType,lightTypeInd,Light::DIRECTIONAL));
 /*   m_PG->Append( new wxBoolProperty( wxT("ENABLED"), wxPG_LABEL ) );

	wxPGProperty* topId;
	
	topId= m_PG->Append( new wxStringProperty(wxT("POSITION"), wxPG_LABEL, wxT("<composed>")) );

    m_PG->AppendIn( topId, new wxFloatProperty( wxT("X"), wxPG_LABEL ) );
    m_PG->AppendIn( topId, new wxFloatProperty( wxT("Y"), wxPG_LABEL ) );
    m_PG->AppendIn( topId, new wxFloatProperty( wxT("Z"), wxPG_LABEL ) );

	topId= m_PG->Append( new wxStringProperty(wxT("DIRECTION"), wxPG_LABEL, wxT("<composed>")) );

    m_PG->AppendIn( topId, new wxFloatProperty( wxT("X"), wxPG_LABEL ) );
    m_PG->AppendIn( topId, new wxFloatProperty( wxT("Y"), wxPG_LABEL ) );
    m_PG->AppendIn( topId, new wxFloatProperty( wxT("Z"), wxPG_LABEL ) );

	topId= m_PG->Append( new wxStringProperty(wxT("COLOR"), wxPG_LABEL, wxT("<composed>")) );
	m_PG->AppendIn(topId,new wxColourProperty(wxT("RGB"),wxPG_LABEL,
					wxColour(255,255,255)));

	topId= m_PG->Append( new wxStringProperty(wxT("AMBIENT"), wxPG_LABEL, wxT("<composed>")) );
	m_PG->AppendIn(topId,new wxColourProperty(wxT("RGB"),wxPG_LABEL,
					wxColour(255,255,255)));

	topId= m_PG->Append( new wxStringProperty(wxT("SPECULAR"), wxPG_LABEL, wxT("<composed>")) );
	m_PG->AppendIn(topId,new wxColourProperty(wxT("RGB"),wxPG_LABEL,
					wxColour(255,255,255)));

	m_PG->Append( new wxFloatProperty(wxT("SPOT_CUTOFF"), wxPG_LABEL ));
	m_PG->Append( new wxFloatProperty(wxT("SPOT_EXPONENT"), wxPG_LABEL ));

	m_PG->Append( new wxFloatProperty(wxT("CONSTANT_ATT"), wxPG_LABEL));
	m_PG->Append( new wxFloatProperty(wxT("LINEAR_ATT"), wxPG_LABEL));
	m_PG->Append( new wxFloatProperty(wxT("QUADRATIC_ATT"), wxPG_LABEL));

	if (count)
		update();*/

	m_PG->SetSplitterLeft(true);
//	m_PG->SetSplitterPosition(100);
	siz->Add(m_PG,1, wxEXPAND|wxALL,5);

	//wxBoxSizer *sizH2 = new wxBoxSizer(wxHORIZONTAL);
	//sizH2->Add(m_PG,1,wxALIGN_CENTER_HORIZONTAL | wxGROW | wxALL  ,5);
	//siz->Add(sizH2,1,wxGROW|wxALL|wxEXPAND,5);

	/* BOTTOM: Add Light Button */

	wxBoxSizer *sizH3 = new wxBoxSizer(wxHORIZONTAL);

	m_BAdd = new wxButton(this,DLG_BUTTON_ADD,wxT("Add Light"));
	sizH3-> Add(m_BAdd,0,wxALL |wxGROW |wxHORIZONTAL|wxALIGN_CENTER_HORIZONTAL,5);

	siz->Add(sizH3,0,wxALL |wxGROW |wxHORIZONTAL,15);

	if (!count) {
		m_PG->Disable();
		m_List->Disable();
	}

}


void
DlgLights::setupGrid() {

	m_PG->Clear();
	m_PG->AddPage(wxT("properties"));

	std::vector<std::string> order = { "ENABLED", "POSITION", "DIRECTION", "AMBIENT", "COLOR" };

	PropertyManager::createOrderedGrid(m_PG, Light::Attribs, order);
	m_PG->SetSplitterLeft(true, true);

}


void 
DlgLights::updateList() {

	std::vector<std::string> *names = RENDERMANAGER->getLightNames();
	int num = names->size();

	m_List->Clear();

	for(int i = 0; i < num; i++)  {
		wxString s;
		s << i;
		m_List->Append(wxString(names->at(i).c_str()));
	}
	m_active = names->at(0);

	delete names;
}



void 
DlgLights::update() {


	nau::scene::Light *light;		
	light = RENDERMANAGER->getLight(m_active);
//	unsigned int type = light->getPrope(Light::TYPE);

	m_PG->ClearSelection();

	PropertyManager::updateGrid(m_PG, Light::Attribs, (AttributeValues *)light);
	//m_PG->SetPropertyValue(wxT("ENABLED"),light->getPropb(Light::ENABLED));

	//if (nau::scene::Light::DIRECTIONAL == type) {
	//	m_PG->SetPropertyValue(wxT("TYPE"), DLG_MI_DIRECTIONAL);
	//	m_PG->DisableProperty(wxT("POSITION"));
	//	m_PG->DisableProperty(wxT("SPOT_CUTOFF"));
	//	m_PG->DisableProperty(wxT("SPOT_EXPONENT"));
	//	//m_PG->DisableProperty(wxT("Attenuation"));
	//	m_PG->DisableProperty(wxT("CONSTANT_ATT"));
	//	m_PG->DisableProperty(wxT("LINEAR_ATT"));
	//	m_PG->DisableProperty(wxT("QUADRATIC_ATT"));
	//}
	//else if (nau::scene::Light::POSITIONAL == type) {
	//	m_PG->SetPropertyValue(wx vvvvvT("TYPE"), DLG_MI_POINT);
	//	m_PG->EnableProperty(wxT("POSITION"));
	//	m_PG->DisableProperty(wxT("SPOT_CUTOFF"));
	//	m_PG->DisableProperty(wxT("SPOT_EXPONENT"));
	//	//m_PG->EnableProperty(wxT("Attenuation"));
	//	m_PG->EnableProperty(wxT("CONSTANT_ATT"));
	//	m_PG->EnableProperty(wxT("LINEAR_ATT"));
	//	m_PG->EnableProperty(wxT("QUADRATIC_ATT"));
	//}
	//else if (nau::scene::Light::SPOT_LIGHT == type) {
	//	m_PG->SetPropertyValue(wxT("TYPE"), DLG_MI_SPOT);
	//	m_PG->EnableProperty(wxT("POSITION"));
	//	m_PG->EnableProperty(wxT("SPOT_CUTOFF"));
	//	m_PG->EnableProperty(wxT("SPOT_EXPONENT"));
	//	//m_PG->EnableProperty(wxT("Attenuation"));
	//	m_PG->EnableProperty(wxT("CONSTANT_ATT"));
	//	m_PG->EnableProperty(wxT("LINEAR_ATT"));
	//	m_PG->EnableProperty(wxT("QUADRATIC_ATT"));
	//}

	//nau::math::vec4 v = light->getPropf4(Light::POSITION);
	//m_PG->SetPropertyValue(wxT("POSITION.X"),v.x);
	//m_PG->SetPropertyValue(wxT("POSITION.Y"),v.y);
	//m_PG->SetPropertyValue(wxT("POSITION.Z"),v.z);

	//v = light->getPropf4(Light::DIRECTION);
	//m_PG->SetPropertyValue(wxT("DIRECTION.X"),v.x);
	//m_PG->SetPropertyValue(wxT("DIRECTION.Y"),v.y);
	//m_PG->SetPropertyValue(wxT("DIRECTION.Z"),v.z);

	//v = light->getPropf4(Light::COLOR);
	//m_PG->SetPropertyValue(wxT("COLOR.RGB"),
	//				wxColour(255*v.x,255*v.y,255*v.z));

	//v = light->getPropf4(Light::AMBIENT);
	//m_PG->SetPropertyValue(wxT("AMBIENT.RGB"),
	//				wxColour(255*v.x,255*v.y,255*v.z));

	//v = light->getPropf4(Light::SPECULAR);
	//m_PG->SetPropertyValue(wxT("SPECULAR.RGB"),
	//				wxColour(255*v.x,255*v.y,255*v.z));

	//float ff;

	//ff = light->getPropf(Light::SPOT_CUTOFF);
	//m_PG->SetPropertyValue(wxT("SPOT_CUTOFF"),ff);

	//ff = light->getPropf(Light::SPOT_EXPONENT);
	//m_PG->SetPropertyValue(wxT("SPOT_EXPONENT"),ff);

	//ff = light->getPropf(Light::CONSTANT_ATT);
	//m_PG->SetPropertyValue(wxT("CONSTANT_ATT"),ff);

	//ff = light->getPropf(Light::LINEAR_ATT);
	//m_PG->SetPropertyValue(wxT("LINEAR_ATT"),ff);

	//ff = light->getPropf(Light::QUADRATIC_ATT);
	//m_PG->SetPropertyValue(wxT("QUADRATIC_ATT"),ff);

}



void DlgLights::OnPropsChange( wxPropertyGridEvent& e) {

	nau::scene::Light *light = RENDERMANAGER->getLight(m_active);
	const wxString& name = e.GetPropertyName();
	unsigned int dotLocation = name.find_first_of(wxT("."),0);
	std::string topProp = std::string(name.substr(0,dotLocation).mb_str());
	std::string prop = std::string(name.substr(dotLocation+1,name.size()-dotLocation-1).mb_str());
	wxVariant variant;
	wxColour col;


	PropertyManager::updateProp(m_PG, name.ToStdString(), Light::Attribs, (AttributeValues *)light);
//	m_PG->ClearSelection();
	//nau::math::vec4 v1,v2;
	//if (topProp == "ENABLED")
	//	light->setPropb(Light::ENABLED, e.GetPropertyValue().GetBool() != 0);


	//if (topProp == "POSITION") {
	//	v1 = light->getPropf4(Light::POSITION);
	//	if      (prop == "X") {
	//		v2.set(e.GetPropertyValue().GetDouble(),v1.y,v1.z);
	//		light->setPropf4(Light::POSITION,v2);
	//	}
	//	else if (prop == "Y") {
	//		v2.set(v1.x,e.GetPropertyValue().GetDouble(),v1.z);
	//		light->setPropf4(Light::POSITION,v2);
	//	}
	//	else if (prop == "Z") {
	//		v2.set(v1.x,v1.y,e.GetPropertyValue().GetDouble());
	//		light->setPropf4(Light::POSITION,v2);
	//	}
	//}
	//else if (topProp == "DIRECTION") {
	//	v1 = light->getPropf4(Light::DIRECTION);
	//	if      (prop == "X") {
	//		v2.set(e.GetPropertyValue().GetDouble(),v1.y,v1.z, 0.0);
	//		light->setPropf4(Light::DIRECTION,v2);
	//	}
	//	else if (prop == "Y") {
	//		v2.set(v1.x,e.GetPropertyValue().GetDouble(),v1.z, 0.0);
	//		light->setPropf4(Light::DIRECTION,v2);
	//	}
	//	else if (prop == "Z") {
	//		v2.set(v1.x,v1.y,e.GetPropertyValue().GetDouble(), 0.0);
	//		light->setPropf4(Light::DIRECTION,v2);
	//	}
	//}	
	//	
	////else if (topProp == "TYPE") {

	////	int op = e.GetPropertyValue().GetInteger();
	////	light->setProp(Light::TYPE, (Light::EnumProperty)op);

	////	if (op == Light::DIRECTIONAL) {	
	////		m_PG->DisableProperty(wxT("POSITION"));
	////		m_PG->EnableProperty(wxT("DIRECTION"));
	////		m_PG->DisableProperty(wxT("SPOT_CUTOFF"));
	////		m_PG->DisableProperty(wxT("SPOT_EXPONENT"));
	////		m_PG->DisableProperty(wxT("CONSTANT_ATT"));
	////		m_PG->DisableProperty(wxT("LINEAR_ATT"));
	////		m_PG->DisableProperty(wxT("QUADRATIC_ATT"));
	////	}
	////	else if (op == Light::POSITIONAL){
	////		m_PG->EnableProperty(wxT("POSITION"));
	////		m_PG->DisableProperty(wxT("DIRECTION"));
	////		m_PG->DisableProperty(wxT("SPOT_CUTOFF"));
	////		m_PG->DisableProperty(wxT("SPOT_EXPONENT"));
	////		m_PG->EnableProperty(wxT("CONSTANT_ATT"));
	////		m_PG->EnableProperty(wxT("LINEAR_ATT"));
	////		m_PG->EnableProperty(wxT("QUADRATIC_ATT"));
	////	}
	////	else if (op == Light::SPOT_LIGHT){
	////		m_PG->EnableProperty(wxT("POSITION"));
	////		m_PG->EnableProperty(wxT("DIRECTION"));
	////		m_PG->EnableProperty(wxT("SPOT_CUTOFF"));
	////		m_PG->EnableProperty(wxT("SPOT_EXPONENT"));
	////		m_PG->EnableProperty(wxT("CONSTANT_ATT"));
	////		m_PG->EnableProperty(wxT("LINEAR_ATT"));
	////		m_PG->EnableProperty(wxT("QUADRATIC_ATT"));
	////	}
	////}

	//else if (topProp == "COLOR") {
	//	variant = m_PG->GetPropertyValue(wxT("COLOR.RGB"));
	//	col << variant;
	//	light->setPropf4(Light::COLOR, vec4(col.Red()/255.0f, col.Green()/255.0f, col.Blue()/255.0f, 1.0));
	//}
	//else if (topProp == "AMBIENT") {
	//	variant = m_PG->GetPropertyValue(wxT("AMBIENT.RGB"));
	//	col << variant;
	//	light->setPropf4(Light::AMBIENT, vec4(col.Red()/255.0f, col.Green()/255.0f, col.Blue()/255.0f, 1.0));
	//}
	//else if (topProp == "SPECULAR") {
	//	variant = m_PG->GetPropertyValue(wxT("SPECULAR.RGB"));
	//	col << variant;
	//	light->setPropf4(Light::SPECULAR, vec4(col.Red()/255.0f, col.Green()/255.0f, col.Blue()/255.0f, 1.0));
	//}
	//else if (topProp == "SPOT_CUTOFF") {
	//	light->setPropf(Light::SPOT_CUTOFF, e.GetPropertyValue().GetDouble());
	//}
	//else if (topProp == "SPOT_EXPONENT") {
	//	light->setPropf(Light::SPOT_EXPONENT, e.GetPropertyValue().GetDouble());
	//}
	//else if (topProp == "CONSTANT_ATT") {
	//	light->setPropf(Light::CONSTANT_ATT, e.GetPropertyValue().GetDouble());
	//}
	//else if (topProp == "LINEAR_ATT") {
	//	light->setPropf(Light::LINEAR_ATT, e.GetPropertyValue().GetDouble());
	//}
	//else if (topProp == "QUADRATIC_ATT") {
	//	light->setPropf(Light::QUADRATIC_ATT, e.GetPropertyValue().GetDouble());
	//}
	notifyUpdate(PROPS_CHANGED,m_active,topProp);
}


void DlgLights::updateInfo(std::string name) {

	if (name == m_active) {
		update();
	}
}


void DlgLights::OnAdd(wxCommandEvent& event) {

	int result;
	bool nameUnique,exit = false;
	std::string name;

	do {
		wxTextEntryDialog dialog(this,
								 _T("(the name must be unique)\n")
								 _T("Please Input a Light Name"),
								 _T("Light Name\n"),
								 _T(""),
								 wxOK | wxCANCEL);

		result = dialog.ShowModal();
		name = std::string(dialog.GetValue().mb_str());
		nameUnique =  !RENDERMANAGER->hasLight(name); 

		if (!nameUnique && (result == wxID_OK)){
			wxMessageBox(_T("Light name must be unique") , _T("Light Name Error"), wxOK | wxICON_INFORMATION, this);
		}
		if (name == "" && (result == wxID_OK)){
			wxMessageBox(_T("Light name must not be void"), _T("Light Name Error"), wxOK | wxICON_INFORMATION, this);
		}

		if (result == wxID_CANCEL) {
			exit = true;
		}
		else if (nameUnique && name != "") {
			exit = true;
		}

	} while (!exit);

	if (result == wxID_OK) {
		RENDERMANAGER->getLight(name);
		m_List->Enable();
		m_PG->Enable();
		updateList();
		m_List->Select(m_List->FindString((wxChar *)name.c_str()));
		m_active = name;
		update();
		notifyUpdate(NEW_LIGHT,name,"");
	}
}

void DlgLights::OnListSelect(wxCommandEvent& event){

	int sel;

	sel = event.GetSelection();
	m_active = std::string(m_List->GetString(sel).mb_str());
	update();
//	parent->Refresh();
}

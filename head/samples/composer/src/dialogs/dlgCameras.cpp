

#include "dialogs/dlgCameras.h"
#include <nau/event/eventFactory.h>
#include <nau/math/utils.h>

/*
Care must be taken when setting the camera vectors (same for light's direction).

The vector should be updated all at once, not component by component, due to the
normalization in every update.
*/


BEGIN_EVENT_TABLE(DlgCameras, wxDialog)
	EVT_COMBOBOX(DLG_COMBO, DlgCameras::OnListSelect)
	EVT_PG_CHANGED( DLG_PROPS, DlgCameras::OnPropsChange )
	EVT_BUTTON(DLG_BUTTON_ADD, DlgCameras::OnAdd)
	EVT_BUTTON(DLG_BUTTON_ACTIVATE, DlgCameras::OnActivate)
END_EVENT_TABLE()


wxWindow *DlgCameras::parent = NULL;
DlgCameras *DlgCameras::inst = NULL;


void DlgCameras::SetParent(wxWindow *p) {

	parent = p;
}


DlgCameras* DlgCameras::Instance () {

	if (inst == NULL)
		inst = new DlgCameras();

	return inst;
}



DlgCameras::DlgCameras()
	: wxDialog(DlgCameras::parent, -1, wxT("Nau - Cameras"),wxDefaultPosition,wxDefaultSize,wxRESIZE_BORDER|wxDEFAULT_DIALOG_STYLE)
                
{
	wxBoxSizer *sizer = new wxBoxSizer(wxVERTICAL);

	setupPanel(sizer,parent);

	SetAutoLayout(TRUE);
    sizer->SetSizeHints(this);
    sizer->Fit(this);
    SetSizer(sizer);

	this->SetTitle(wxT("Nau - Cameras"));
	this->SetSize(350,400);
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


void DlgCameras::updateDlg() {

	updateList();
	updateViewportLabels();
	list->SetSelection(0);
	update();
	m_parent->Refresh();	
}


void DlgCameras::updateViewportLabels() {

	viewportLabels.RemoveAt(0, viewportLabels.GetCount());
	viewportLabels.Add(wxT("None"), -1);

	std::vector<std::string>::iterator iter;
	std::vector<std::string> *viewports = NAU->getViewportNames();
	for (iter = viewports->begin(); iter != viewports->end(); ++iter)
		viewportLabels.Add(wxString(iter->c_str()));

	delete viewports;
}


void DlgCameras::setupPanel(wxSizer *siz, wxWindow *parent) {

	/* TOP: COMBO with camera names */
	wxBoxSizer *sizH1 = new wxBoxSizer(wxHORIZONTAL);

	wxStaticText *stg1 =  new wxStaticText(this,-1,wxT("Camera: "));
	list = new wxComboBox(this,DLG_COMBO,wxT(""),wxDefaultPosition,wxDefaultSize,0,NULL,wxCB_READONLY );

	updateList();
	list->SetSelection(0);

	sizH1->Add(stg1, 0, wxGROW|wxHORIZONTAL,5);
	sizH1->Add(list, 1, wxGROW|wxHORIZONTAL,5);
	siz->Add(sizH1, 0, wxGROW|wxALL, 15);

	/* MIDDLE: Property grid */

	pg = new wxPropertyGridManager(this, DLG_PROPS,
				wxDefaultPosition, wxDefaultSize,
				// These and other similar styles are automatically
				// passed to the embedded wxPropertyGrid.
				wxPG_BOLD_MODIFIED|wxPG_SPLITTER_AUTO_CENTER|
				// Plus defaults.
				wxPGMAN_DEFAULT_STYLE
           );

	pg->AddPage(wxT("Cameras"));

	viewportLabels.Add(wxT("None"));
	pg->Append( new wxEnumProperty(wxT("Viewport"),wxPG_LABEL,viewportLabels));
	updateViewportLabels();
	//wxPropertyGridPage* pgPropsPage = pg->GetPage("Cameras");

	const wxChar* cameraType[] = { wxT("PERSPECTIVE"), wxT("ORTHOGONAL"),NULL};
	const long cameraTypeInd[] = { DLG_MI_PERSPECTIVE, DLG_MI_ORTHOGONAL};

	//pg->Append(new wxStringProperty(wxT("Name"),wxPG_LABEL,""));
 	//pg->DisableProperty(wxT("Name"));

	wxPGProperty* topId,*pid2;
	
	topId= pg->Append( new wxStringProperty(wxT("Position"), wxPG_LABEL, wxT("<composed>")) );

    pg->AppendIn( topId, new wxFloatProperty( wxT("X"), wxPG_LABEL ) );
    pg->AppendIn( topId, new wxFloatProperty( wxT("Y"), wxPG_LABEL ) );
    pg->AppendIn( topId, new wxFloatProperty( wxT("Z"), wxPG_LABEL ) );

	topId= pg->Append( new wxStringProperty(wxT("Direction"), wxPG_LABEL, wxT("<composed>")) );

	pg->AppendIn( topId, new wxFloatProperty( wxT("Elevation Angle"), wxPG_LABEL));
	pg->AppendIn( topId, new wxFloatProperty( wxT("ZX Angle"), wxPG_LABEL));

	pid2 = pg->AppendIn(topId, new wxStringProperty(wxT("View"), wxPG_LABEL, wxT("<composed>")) );
    pg->AppendIn( pid2, new wxFloatProperty( wxT("X"), wxPG_LABEL ) );
    pg->AppendIn( pid2, new wxFloatProperty( wxT("Y"), wxPG_LABEL ) );
    pg->AppendIn( pid2, new wxFloatProperty( wxT("Z"), wxPG_LABEL ) );
	pg->DisableProperty(pid2);

	topId= pg->Append( new wxStringProperty(wxT("Up"), wxPG_LABEL, wxT("<composed>")) );
    pg->AppendIn( topId, new wxFloatProperty( wxT("X"), wxPG_LABEL ) );
    pg->AppendIn( topId, new wxFloatProperty( wxT("Y"), wxPG_LABEL ) );
    pg->AppendIn( topId, new wxFloatProperty( wxT("Z"), wxPG_LABEL ) );
	pg->DisableProperty(topId);

    pg->Append( new wxFloatProperty( wxT("Near"), wxPG_LABEL ) );
    pg->Append( new wxFloatProperty( wxT("Far"), wxPG_LABEL ) );

	pg->Append(new wxEnumProperty(wxT("Projection"),wxPG_LABEL,cameraType,cameraTypeInd,DLG_MI_PERSPECTIVE));
	//pg->DisableProperty(wxT("Camera Type"));

	topId= pg->Append( new wxStringProperty(wxT("Perspective Proj:"), wxPG_LABEL, wxT("<composed>")) );

    pg->AppendIn( topId, new wxFloatProperty( wxT("Y FOV"), wxPG_LABEL ) );

	topId= pg->Append( new wxStringProperty(wxT("Orthogonal Proj:"), wxPG_LABEL, wxT("<composed>")) );

    pg->AppendIn( topId, new wxFloatProperty( wxT("Left"), wxPG_LABEL ) );
    pg->AppendIn( topId, new wxFloatProperty( wxT("Right"), wxPG_LABEL ) );
    pg->AppendIn( topId, new wxFloatProperty( wxT("Bottom"), wxPG_LABEL ) );
    pg->AppendIn( topId, new wxFloatProperty( wxT("Top"), wxPG_LABEL ) );

	for (int i = 0; i < Camera::COUNT_MAT4PROPERTY; i++)
		addMatrix(pg, (Camera::Mat4Property)i);

	update();

	pg->SetSplitterLeft(true);
	//pg->SetSplitterPosition(100);

//	wxBoxSizer *sizH2 = new wxBoxSizer(wxHORIZONTAL);
//	sizH2->Add(pg,1,wxALIGN_CENTER_HORIZONTAL | wxGROW | wxALL  ,5);
	siz->Add(pg,1, wxEXPAND|wxALL,5);

	/* BOTTOM: Add Camera Button */

	wxBoxSizer *sizH3 = new wxBoxSizer(wxHORIZONTAL);

	bAdd = new wxButton(this,DLG_BUTTON_ADD,wxT("Add Camera"));
	bActivate = new wxButton(this,DLG_BUTTON_ACTIVATE,wxT("Activate Camera"));
	sizH3-> Add(bAdd,0,wxALL |wxGROW |wxHORIZONTAL|wxALIGN_CENTER_HORIZONTAL,5);
	sizH3-> Add(bActivate,0,wxALL |wxGROW |wxHORIZONTAL|wxALIGN_CENTER_HORIZONTAL,5);

	siz->Add(sizH3,0,wxALL |wxGROW |wxHORIZONTAL,15);
}


void 
DlgCameras::addMatrix(wxPropertyGridManager *pg, Camera::Mat4Property m)
{
	wxPGProperty* topId,*pid2;
	nau::scene::Camera *cam;		
	cam = RENDERMANAGER->getCamera(m_active);

	std::string name = Camera::Attribs.getName(m, Enums::DataType::MAT4); //Mat4String[m];
	topId = pg->Append(new wxStringProperty(wxString(name.c_str()), wxPG_LABEL, wxT("<composed>")));
	pg->DisableProperty(topId);
	pid2 = pg->AppendIn(topId, new wxStringProperty(wxT("Row 0"), wxPG_LABEL, wxT("<composed>")) );
	pg->DisableProperty(pid2);
    pg->AppendIn( pid2, new wxFloatProperty( wxT("m00"), wxPG_LABEL ) );
    pg->AppendIn( pid2, new wxFloatProperty( wxT("m01"), wxPG_LABEL ) );
    pg->AppendIn( pid2, new wxFloatProperty( wxT("m02"), wxPG_LABEL ) );
    pg->AppendIn( pid2, new wxFloatProperty( wxT("m03"), wxPG_LABEL ) );
	pg->Collapse(pid2);
	pid2 = pg->AppendIn(topId, new wxStringProperty(wxT("Row 1"), wxPG_LABEL, wxT("<composed>")) );
	pg->DisableProperty(pid2);
    pg->AppendIn( pid2, new wxFloatProperty( wxT("m10"), wxPG_LABEL ) );
    pg->AppendIn( pid2, new wxFloatProperty( wxT("m11"), wxPG_LABEL ) );
    pg->AppendIn( pid2, new wxFloatProperty( wxT("m12"), wxPG_LABEL ) );
    pg->AppendIn( pid2, new wxFloatProperty( wxT("m13"), wxPG_LABEL ) );
	pg->Collapse(pid2);
	pid2 = pg->AppendIn(topId, new wxStringProperty(wxT("Row 2"), wxPG_LABEL, wxT("<composed>")) );
	pg->DisableProperty(pid2);
    pg->AppendIn( pid2, new wxFloatProperty( wxT("m20"), wxPG_LABEL ) );
    pg->AppendIn( pid2, new wxFloatProperty( wxT("m21"), wxPG_LABEL ) );
    pg->AppendIn( pid2, new wxFloatProperty( wxT("m22"), wxPG_LABEL ) );
    pg->AppendIn( pid2, new wxFloatProperty( wxT("m23"), wxPG_LABEL ) );
	pg->Collapse(pid2);
	pid2 = pg->AppendIn(topId, new wxStringProperty(wxT("Row 3"), wxPG_LABEL, wxT("<composed>")) );
	pg->DisableProperty(pid2);
    pg->AppendIn( pid2, new wxFloatProperty( wxT("m30"), wxPG_LABEL ) );
    pg->AppendIn( pid2, new wxFloatProperty( wxT("m31"), wxPG_LABEL ) );
    pg->AppendIn( pid2, new wxFloatProperty( wxT("m32"), wxPG_LABEL ) );
    pg->AppendIn( pid2, new wxFloatProperty( wxT("m33"), wxPG_LABEL ) );
	pg->Collapse(pid2);
	pg->Collapse(topId);
}


void 
DlgCameras::updateList() {

	std::vector<std::string> *names = RENDERMANAGER->getCameraNames();
	int num = names->size();

	list->Clear();

	for(int i = 0; i < num; i++)  {
		wxString s;
		s << i;
		list->Append(wxString(names->at(i).c_str()));
	}
	m_active = names->at(0);
	delete names;
}


void DlgCameras::update() {

	nau::scene::Camera *cam;		
	cam = RENDERMANAGER->getCamera(m_active);
	unsigned int proj = cam->getPrope(Camera::PROJECTION_TYPE);

	pg->ClearSelection();

	Viewport *vp = cam->getViewport();
	std::string vpname;
	if (vp)
		vpname = vp->getName();
	else
		vpname = "None";
	pg->SetPropertyValueString(wxT("Viewport"),wxString(vpname.c_str()));
	vec4 v = cam->getPropf4(Camera::POSITION);
	pg->SetPropertyValue(wxT("Position.X"),v.x);
	pg->SetPropertyValue(wxT("Position.Y"),v.y);
	pg->SetPropertyValue(wxT("Position.Z"),v.z);

	pg->SetPropertyValue(wxT("Direction.Elevation Angle"), nau::math::RadToDeg(cam->getPropf(Camera::ELEVATION_ANGLE)));
	pg->SetPropertyValue(wxT("Direction.ZX Angle"), nau::math::RadToDeg(cam->getPropf(Camera::ZX_ANGLE)));

	v = cam->getPropf4(Camera::VIEW_VEC);
	pg->SetPropertyValue(wxT("Direction.View.X"),v.x);
	pg->SetPropertyValue(wxT("Direction.View.Y"),v.y);
	pg->SetPropertyValue(wxT("Direction.View.Z"),v.z);

	v = cam->getPropf4(Camera::UP_VEC);
	pg->SetPropertyValue(wxT("Up.X"),v.x);
	pg->SetPropertyValue(wxT("Up.Y"),v.y);
	pg->SetPropertyValue(wxT("Up.Z"),v.z);

	pg->SetPropertyValue(wxT("Near"),cam->getPropf(Camera::NEARP));
	pg->SetPropertyValue(wxT("Far"),cam->getPropf(Camera::FARP));

	if (nau::scene::Camera::PERSPECTIVE == proj) {
		pg->SetPropertyValue(wxT("Projection"), DLG_MI_PERSPECTIVE);
		pg->DisableProperty(wxT("Orthogonal Proj:"));
		pg->EnableProperty(wxT("Perspective Proj:"));
	}
	else {
		pg->SetPropertyValue(wxT("Projection"), DLG_MI_ORTHOGONAL);
		pg->DisableProperty(wxT("Perspective Proj:"));
		pg->EnableProperty(wxT("Orthogonal Proj:"));
	}
	pg->SetPropertyValue(wxT("Perspective Proj:.Y FOV"),cam->getPropf(Camera::FOV));

	pg->SetPropertyValue(wxT("Orthogonal Proj:.Left"),cam->getPropf(Camera::LEFT));
	pg->SetPropertyValue(wxT("Orthogonal Proj:.Right"),cam->getPropf(Camera::RIGHT));
	pg->SetPropertyValue(wxT("Orthogonal Proj:.Top"),cam->getPropf(Camera::TOP));
	pg->SetPropertyValue(wxT("Orthogonal Proj:.Bottom"),cam->getPropf(Camera::BOTTOM));

	for (int i = 0; i < Camera::COUNT_MAT4PROPERTY; i++)
		updateMatrix(cam, (Camera::Mat4Property)i);
}


void 
DlgCameras::updateMatrix(Camera *cam, Camera::Mat4Property m) 
{
	std::string name = Camera::Attribs.getName(m, Enums::DataType::MAT4); // Camera::Mat4String[m];
	const nau::math::mat4 mat = cam->getPropm4(m);
	char s[64];

	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4 ; j++) {
			sprintf(s,"%s.Row %d.m%d%d",name.c_str(), i, i, j);
			std::string s1(s);
			pg->SetPropertyValue(wxString(s1.c_str()),mat.at(i,j));
		}

}


void DlgCameras::OnPropsChange( wxPropertyGridEvent& e) {

	nau::scene::Camera *cam = RENDERMANAGER->getCamera(m_active);
	const wxString& name = e.GetPropertyName();
	unsigned int dotLocation = name.find_first_of(wxT("."),0);
	std::string topProp = std::string(name.substr(0,dotLocation).mb_str());
	std::string prop = std::string(name.substr(dotLocation+1,name.size()-dotLocation-1).mb_str());

//	pg->ClearSelection();
	nau::math::vec4 v1,v2;
	if (topProp == "Position") {
		nau::math::vec4 v1 = cam->getPropf4(Camera::POSITION);
		if      (prop == "X") {
			v2.set(e.GetPropertyValue().GetDouble(),v1.y,v1.z);
			cam->setProp(Camera::POSITION, v2.x, v2.y, v2.z, 1.0f);
		}
		else if (prop == "Y") {
			v2.set(v1.x,e.GetPropertyValue().GetDouble(),v1.z);
			cam->setProp(Camera::POSITION, v2.x, v2.y, v2.z, 1.0f);
		}
		else if (prop == "Z") {
			v2.set(v1.x,v1.y,e.GetPropertyValue().GetDouble());
			cam->setProp(Camera::POSITION, v2.x, v2.y, v2.z, 1.0f);
		}
	}
	else if (topProp == "Direction") {

		if (prop == "Elevation Angle") {
		
			float rad = nau::math::DegToRad(e.GetPropertyValue().GetDouble());
			cam->setProp(Camera::ELEVATION_ANGLE, rad);
		}
		else if (prop == "ZX Angle") {
		
			float rad = nau::math::DegToRad(e.GetPropertyValue().GetDouble());
			cam->setProp(Camera::ZX_ANGLE, rad);
		}
		vec4 vv = cam->getPropf4(Camera::VIEW_VEC);
		pg->SetPropertyValue(wxT("Direction.View.X"),vv.x);
		pg->SetPropertyValue(wxT("Direction.View.Y"),vv.y);
		pg->SetPropertyValue(wxT("Direction.View.Z"),vv.z);

		vv = cam->getPropf4(Camera::UP_VEC);
		pg->SetPropertyValue(wxT("Up.X"),vv.x);
		pg->SetPropertyValue(wxT("Up.Y"),vv.y);
		pg->SetPropertyValue(wxT("Up.Z"),vv.z);
		//nau::math::vec3 v1 = cam->getViewVector();
		//if      (prop == "X") {
		//	v2.set(e.GetPropertyValueAsDouble(),v1.y,v1.z);
		//	cam->setViewVector(v2);
		//}
		//else if (prop == "Y") {
		//	v2.set(v1.x,e.GetPropertyValueAsDouble(),v1.z);
		//	cam->setViewVector(v2);
		//}
		//else if (prop == "Z") {
		//	v2.set(v1.x,v1.y,e.GetPropertyValueAsDouble());
		//	cam->setViewVector(v2);
		//}

	}	
	//else if (topProp == "Up") {
	//	nau::math::vec3 v1 = cam->getUpVector();
	//	if      (prop == "X") {
	//		v2.set(e.GetPropertyValueAsDouble(),v1.y,v1.z);
	//		cam->setUpVector(v2);
	//	}
	//	else if (prop == "Y") {
	//		v2.set(v1.x,e.GetPropertyValueAsDouble(),v1.z);
	//		cam->setUpVector(v2);
	//	}
	//	else if (prop == "Z") {
	//		v2.set(v1.x,v1.y,e.GetPropertyValueAsDouble());
	//		cam->setUpVector(v2);
	//	}
	//}	
	else if (topProp == "Projection") {
		int op = e.GetPropertyValue().GetInteger();
		if (op == DLG_MI_PERSPECTIVE) {	
			pg->DisableProperty(wxT("Orthogonal Proj:"));
			pg->EnableProperty(wxT("Perspective Proj:"));
			cam->setProp(Camera::PROJECTION_TYPE, Camera::PERSPECTIVE);
		}
		else {
			pg->DisableProperty(wxT("Perspective Proj:"));
			pg->EnableProperty(wxT("Orthogonal Proj:"));
			cam->setProp(Camera::PROJECTION_TYPE, Camera::ORTHO);
		}
	}
	else if (prop == "Near") {
		cam->setProp(Camera::NEARP,e.GetPropertyValue().GetDouble());
	}
	else if (prop == "Far") {
		cam->setProp(Camera::FARP, e.GetPropertyValue().GetDouble());
	}
	else if (prop == "Y FOV") {
		cam->setProp(Camera::FOV, e.GetPropertyValue().GetDouble());
	}
	else if (prop == "Left") {
		cam->setProp(Camera::LEFT, e.GetPropertyValue().GetDouble());
	}
	else if (prop == "Right") {
		cam->setProp(Camera::RIGHT, e.GetPropertyValue().GetDouble());
	}
	else if (prop == "Top") {
		cam->setProp(Camera::TOP, e.GetPropertyValue().GetDouble());
	}
	else if (prop == "Bottom") {
		cam->setProp(Camera::BOTTOM, e.GetPropertyValue().GetDouble());
	}

	for (int i = 0; i < Camera::COUNT_MAT4PROPERTY; i++)
		updateMatrix(cam, (Camera::Mat4Property)i);

	notifyUpdate(PROPS_CHANGED,m_active,topProp);
}


void DlgCameras::updateInfo(std::string name) {

	if (name == m_active) {
		update();
	}
}


void DlgCameras::OnActivate(wxCommandEvent& event) {

	NAU->setActiveCameraName(m_active);
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
		list->Select(list->FindString((wxString)name.c_str()));
		m_active = name;
		update();
		notifyUpdate(NEW_CAMERA,name,"");
	}
}


void DlgCameras::OnListSelect(wxCommandEvent& event){

	int sel;

	sel = event.GetSelection();
	m_active = std::string(list->GetString(sel).mb_str());
	update();
//	parent->Refresh();
}

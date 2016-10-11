#include "dialogs/dlgShaders.h"

#include <nau/event/eventFactory.h>
#include <nau/material/uniformBlockManager.h>
#include <nau/render/iAPISupport.h>

BEGIN_EVENT_TABLE(DlgShaders, wxDialog)

	EVT_COMBOBOX(DLG_COMBO, DlgShaders::OnListSelect)
	EVT_PG_CHANGED( DLG_PROPS, DlgShaders::OnPropsChange )
	EVT_BUTTON(DLG_BUTTON_ADD, DlgShaders::OnAdd)

	EVT_BUTTON(DLG_SHADER_COMPILE_AND_LINK, DlgShaders::OnProcessCompileAndLinkShaders)
	EVT_BUTTON(DLG_SHADER_VALIDATE, DlgShaders::OnProcessValidateShaders)

END_EVENT_TABLE()

wxWindow *DlgShaders::parent = NULL;
DlgShaders *DlgShaders::inst = NULL;


void DlgShaders::SetParent(wxWindow *p) {

	parent = p;
}

DlgShaders* DlgShaders::Instance () {

	if (inst == NULL)
		inst = new DlgShaders();

	return inst;
}




DlgShaders::DlgShaders()
	: wxDialog(DlgShaders::parent, -1, wxT("Nau - Programs"),wxDefaultPosition,wxDefaultSize,wxRESIZE_BORDER|wxDEFAULT_DIALOG_STYLE),
	m_Name("Shaders Dialog")
                
{
	SetSizeHints( wxDefaultSize, wxDefaultSize );
	wxBoxSizer *sizer = new wxBoxSizer(wxVERTICAL);

	setupPanel(sizer,parent);

	//SetAutoLayout(TRUE);
    SetSizer(sizer);
	Layout();
	Centre(wxBOTH);

    sizer->SetSizeHints(this);
    sizer->Fit(this);

	this->SetTitle(wxT("Nau - Programs"));
}


void
DlgShaders::notifyUpdate(Notification aNot, std::string shaderName, std::string value) {

	// sends events on behalf of the shader
	std::shared_ptr<nau::event_::IEventData> e = nau::event_::EventFactory::Create("String");
	if (aNot == NEW_SHADER) {
		e->setData(&shaderName);
		EVENTMANAGER->notifyEvent("NEW_SHADER", shaderName,"", e);
	}
	else  {
		e->setData(&value);
		EVENTMANAGER->notifyEvent("SHADER_CHANGED", shaderName ,"", e);
	}
}


void 
DlgShaders::updateDlg() {

	
	updateList();
	m_List->SetSelection(0);
	update();
	EVENTMANAGER->addListener("MATERIAL_SHADER_USAGE_REPORT", this);

}


void
DlgShaders::eventReceived(const std::string &sender, const std::string &eventType,
	const std::shared_ptr<nau::event_::IEventData> &evt) {

	std::string *str;

	if (eventType == "MATERIAL_SHADER_USAGE_REPORT" && sender == m_Active) {

		str = (std::string *)evt->getData();
		updateLogAux(*str);
	}
}


void 
DlgShaders::setupPanel(wxSizer *siz, wxWindow *parent) {

	// Program identification
	wxBoxSizer *sizH1 = new wxBoxSizer(wxHORIZONTAL);

	wxStaticText *stg1 =  new wxStaticText(this,-1,wxT("Program: "));
	m_List = new wxComboBox(this,DLG_COMBO,wxT(""),wxDefaultPosition,wxDefaultSize,0,NULL,wxCB_READONLY );

	updateList();
	m_List->SetSelection(0);

	sizH1->Add(stg1, 0, wxALL,5);
	sizH1->Add(m_List, 1, wxALIGN_RIGHT|wxALL,5);
	siz->Add(sizH1, 0, wxEXPAND, 5);

	/* MIDDLE: Property grid */

	m_PG = new wxPropertyGridManager(this, DLG_PROPS,
				wxDefaultPosition, wxDefaultSize,
				// These and other similar styles are automatically
				// passed to the embedded wxPropertyGrid.
				wxPG_BOLD_MODIFIED|wxPG_SPLITTER_AUTO_CENTER|
				// Plus defaults.
				wxPGMAN_DEFAULT_STYLE
           );

	m_PG->AddPage(wxT("Programs"));

	m_PG->Append( new wxPropertyCategory(wxT("Files"),wxPG_LABEL) );	

	for (int i = 0; i < IProgram::SHADER_COUNT; ++i) {
		m_Shader[i] = new wxFileProperty( IProgram::ShaderNames[i].c_str(), IProgram::ShaderNames[i].c_str() );
		//m_Shader[i]->SetAttribute(wxPG_FILE_WILDCARD,wxT("Vertex Shader Files (*.vert)|*.vert"));
		m_PG->Append(m_Shader[i]);
	}
	m_PG->Append( new wxPropertyCategory(wxT("Program properties"),wxPG_LABEL) );	

    m_LinkStatus = new wxBoolProperty( wxT("Link Status"), wxPG_LABEL );
	m_PG->Append(m_LinkStatus);

	m_ValidateStatus = new wxBoolProperty( wxT("Validate Status"), wxPG_LABEL );
	m_PG->Append(m_ValidateStatus);

	m_ActiveAtomicBuffers = new wxIntProperty( wxT("Active Atomic Counter Buffers"), wxPG_LABEL );
	m_PG->Append(m_ActiveAtomicBuffers);
		
	m_ActiveAttributes = new wxIntProperty( wxT("Active Attributes"), wxPG_LABEL );
	m_PG->Append(m_ActiveAttributes);
		
	m_ActiveUniforms = new wxIntProperty( wxT("Active Uniforms"), wxPG_LABEL );
	m_PG->Append(m_ActiveUniforms);
		
	siz->Add(m_PG,1, wxEXPAND|wxALL,5);

	// Buttons

	wxSizer *sizerS_SB = new wxBoxSizer(wxHORIZONTAL);
		
	m_bValidate = new wxButton(this,DLG_SHADER_VALIDATE,wxT("Validate"));
	m_bCompileAndLink = new wxButton(this,DLG_SHADER_COMPILE_AND_LINK,wxT("Compile and Link"));

	sizerS_SB->Add(m_bValidate,0,wxALL,5);
	sizerS_SB->Add(m_bCompileAndLink,0,wxALL,5);

	siz->Add(sizerS_SB,0,wxALIGN_CENTER_HORIZONTAL,5);

	/* Logs */

	wxBoxSizer *sizerL = new wxBoxSizer(wxHORIZONTAL);

	m_Log = new wxListBox(this,DLG_SHADER_LOG,wxDefaultPosition,wxDefaultSize,0,NULL,0);
	//m_Log->SetMinSize( wxSize( -1,25 ) );

	sizerL->Add(m_Log,1, wxALL|wxEXPAND, 5);
	siz->Add(sizerL,1,wxALL|wxEXPAND,5);

	/* BOTTOM: Add Shader Button */

	wxBoxSizer *sizH3 = new wxBoxSizer(wxVERTICAL);

	bAdd = new wxButton(this,DLG_BUTTON_ADD,wxT("Add Program"));
	sizH3-> Add(bAdd,0,wxALL ,5);

	siz->Add(sizH3,0,wxEXPAND,5);
}


void 
DlgShaders::updateList() {

	std::vector<std::string> *names = RESOURCEMANAGER->getProgramNames();
	int num = names->size();

	m_List->Clear();

	for(int i = 0; i < num; i++)  {
		wxString s;
		s << i;
		m_List->Append(wxString(names->at(i).c_str()));
	}
	if (num > 0)
		m_Active = names->at(0);
	else
		m_Active = "";
	delete names;
}


void 
DlgShaders::OnPropsChange( wxPropertyGridEvent& e) {

	std::string fn;
	const wxString& name = e.GetPropertyName();
	wxString filename = m_PG->GetPropertyValueAsString(name);
	fn = std::string(filename.mb_str());
	
	IProgram *p = RESOURCEMANAGER->getProgram(m_Active);
	if (name == wxT("Vertex"))
		p->setShaderFile(IProgram::VERTEX_SHADER,fn);
//#if NAU_OPENGL_VERSION >= 320
	else if (name == wxT("Geometry"))
		p->setShaderFile(IProgram::GEOMETRY_SHADER, fn);
//#endif
	else if (name == wxT("Fragment"))
		p->setShaderFile(IProgram::FRAGMENT_SHADER,fn);
//#if NAU_OPENGL_VERSION >= 400
	else if (name == wxT("Tess Eval"))
		p->setShaderFile(IProgram::TESS_EVALUATION_SHADER,fn);
	else if (name == wxT("Tess Control"))
		p->setShaderFile(IProgram::TESS_CONTROL_SHADER,fn);
//#endif
//#if (NAU_OPENGL_VERSION >= 430)
	else if (name == wxT("Compute"))
		p->setShaderFile(IProgram::COMPUTE_SHADER,fn);
//#endif

	updateShaderAux();
	notifyUpdate(PROPS_CHANGED, m_Active,std::string(name.mb_str()));

}


void
DlgShaders::update(){

	if (m_List->GetCount() == 0) {
		m_List->Disable();
		m_PG->GetPageByName(wxT("Programs"));
		for (int i = 0; i < IProgram::SHADER_COUNT; ++i) {
			m_PG->DisableProperty(m_Shader[i]);
			m_PG->SetPropertyValue(m_Shader[i],"");
		}

		wxPGProperty *prop = m_PG->GetPropertyByName(wxT("Uniform Variables"));
		if (prop)
			m_PG->DeleteProperty(wxT("Uniform Variables"));

		m_bCompileAndLink->Disable();
		m_bValidate->Disable();
		m_Log->Disable();
		m_Log->Clear();
	}

	else {

		m_List->Enable();
		m_Log->Enable();
		m_Log->Clear();

		GLProgram *p = (GLProgram *)RESOURCEMANAGER->getProgram(m_Active);
		std::string f[IProgram::SHADER_COUNT];
		
		m_PG->ClearSelection();

		for (int i = 0 ; i < IProgram::SHADER_COUNT; ++i) {
		
			m_PG->SetPropertyValue(m_Shader[i],(char *)p->getShaderFile((IProgram::ShaderType)i).c_str());
			m_PG->EnableProperty(m_Shader[i]);
		}

		if (!APISupport->apiSupport(IAPISupport::GEOMETRY_SHADER))
			m_PG->DisableProperty(m_Shader[IProgram::GEOMETRY_SHADER]);

		if (!APISupport->apiSupport(IAPISupport::COMPUTE_SHADER))
			m_PG->DisableProperty(m_Shader[IProgram::COMPUTE_SHADER]);

		if (!APISupport->apiSupport(IAPISupport::TESSELATION_SHADERS)) {
			m_PG->DisableProperty(m_Shader[IProgram::TESS_CONTROL_SHADER]);
			m_PG->DisableProperty(m_Shader[IProgram::TESS_EVALUATION_SHADER]);
		}

		updateProgramProperties(p);
		updateShaderAux();
	}
}


void 
DlgShaders::updateShaderAux() {

	GLProgram *p = (GLProgram *)RESOURCEMANAGER->getProgram(m_Active);
	std::string vfn = p->getShaderFile(IProgram::VERTEX_SHADER),
				ffn = p->getShaderFile(IProgram::FRAGMENT_SHADER),
				gfn = p->getShaderFile(IProgram::GEOMETRY_SHADER),
				cfn = p->getShaderFile(IProgram::COMPUTE_SHADER);

	if (vfn != "" || ffn != "" || gfn != "" || cfn != "") {
		m_bCompileAndLink->Enable();
	}
	else {
		m_bCompileAndLink->Disable();
	}

/*	if (p->areCompiled() && p->isLinked())
		m_bLink->Enable();
	else
		m_bLink->Disable();
*/
	if (p->isLinked())
		m_bValidate->Enable();
	else
		m_bValidate->Disable();


	std::map<std::string, nau::material::ProgramValue> progValues;
	std::map<std::string, nau::material::ProgramValue>::iterator progValuesIter;

	GLUniform u;
	std::string s;
	m_PG->GetPage(0);
	updateProgramProperties(p);
	wxPGProperty *prop = m_PG->GetPropertyByName(wxT("Uniform Variables"));
	if (prop)
		m_PG->DeleteProperty(wxT("Uniform Variables"));

	if (p->isLinked()) {
		std::vector<std::string> blockNames, uniformNames;
		p->getUniformBlockNames(&blockNames);
		int uni = p->getNumberOfUniforms();
		if (uni + blockNames.size() != 0) {
			m_PG->Append(new wxPropertyCategory(wxT("Uniform Variables"), wxPG_LABEL));
		}
		//p->updateUniforms();

		wxPGProperty *pid = m_PG->GetPropertyByName(wxT("Default Block"));
		if (!pid) {
			pid = new wxStringProperty(wxT("Default Block"), wxPG_LABEL);
			m_PG->Append(pid);
		}
		for (int i = 0; i < uni; i++) {
			u = p->getUniform(i);
			addUniform(pid, wxString(u.getName().c_str()),
				wxString(u.getStringSimpleType().c_str()));
		}

		IUniformBlock *ub;
		for (auto b : blockNames) {
			pid = new wxStringProperty(wxString(b.c_str()), wxPG_LABEL);
			m_PG->Append(pid);
			ub = UNIFORMBLOCKMANAGER->getBlock(b);
			ub->getUniformNames(&uniformNames);
			for (auto n : uniformNames) {
				addUniform(pid, wxString(n.c_str()), 
					wxString(Enums::DataTypeToString[ub->getUniformType(n)].c_str()));
			}
			uniformNames.clear();
		}
	}
	m_PG->Refresh();
	m_Log->Clear();
	//DlgMaterials::Instance()->updateDlg();
}


void 
DlgShaders::addUniform(wxPGProperty *pid, wxString name, wxString type) {

	wxPGProperty *pid2;
	pid2 = m_PG->AppendIn(pid, new wxStringProperty(name,wxPG_LABEL,type));
	m_PG->DisableProperty(pid2);
}


wxString 
DlgShaders::getUniformType(int type) {

	std::string s;
	switch (type) {
		case Enums::FLOAT:s = "float";break;
		case Enums::VEC2:s = "vec2";break;
		case Enums::VEC3:s = "vec3";break;
		case Enums::VEC4:s = "vec4";break;

		case Enums::INT: s = "int";break;
		case Enums::IVEC2: s = "ivec2";break;
		case Enums::IVEC3: s = "ivec3";break;
		case Enums::IVEC4: s = "ivec4";break;

		case Enums::BOOL: s = "bool";break;
		case Enums::BVEC2: s = "bvec2";break;
		case Enums::BVEC3: s = "bvec3";break;
		case Enums::BVEC4: s = "bvec4";break;

		case Enums::SAMPLER: s = "sampler";break;

		case Enums::MAT2: s = "mat2";break;
		case Enums::MAT3: s = "mat3";break;
		case Enums::MAT4: s = "mat4";break;
	}
	return wxString(s.c_str());
}


void 
DlgShaders::updateProgramProperties(GLProgram *p) {

	m_PG->SetPropertyValue(m_LinkStatus, p->getPropertyb((int)GL_LINK_STATUS));
	m_PG->DisableProperty(m_LinkStatus);
	m_PG->SetPropertyValue(m_ValidateStatus, p->getPropertyb((int)GL_VALIDATE_STATUS));
	m_PG->DisableProperty(m_ValidateStatus);
	m_PG->SetPropertyValue(m_ActiveAtomicBuffers, p->getPropertyi((int)GL_ACTIVE_ATOMIC_COUNTER_BUFFERS));
	m_PG->DisableProperty(m_ActiveAtomicBuffers);
	m_PG->SetPropertyValue(m_ActiveAttributes, p->getPropertyi((int)GL_ACTIVE_ATTRIBUTES));
	m_PG->DisableProperty(m_ActiveAttributes);
	m_PG->SetPropertyValue(m_ActiveUniforms, p->getPropertyi((int)GL_ACTIVE_UNIFORMS));
	m_PG->DisableProperty(m_ActiveUniforms);


}


void 
DlgShaders::updateLogAux(std::string s) {

	if (s != "") {
		std::string aux = s;
		int loc0 = 0;
		int loc = aux.find(10);
		while (loc != aux.npos) {
			m_Log->AppendAndEnsureVisible(wxString(aux.substr(0,loc).c_str()));
			aux = aux.substr(loc+1);
			loc = aux.find(10);
		}
		//aux = s.su;
		m_Log->AppendAndEnsureVisible(wxString(aux.c_str()));
	}
}


void 
DlgShaders::OnProcessCompileAndLinkShaders(wxCommandEvent& event) {

	std::string infoLog;
	IProgram *p = RESOURCEMANAGER->getProgram(m_Active);

	// compile individual shaders
	for (int i = 0; i < IProgram::SHADER_COUNT; ++i) {

		if (p->getShaderFile((IProgram::ShaderType)i) != "" &&p->reloadShaderFile((IProgram::ShaderType)i)) {
			p->compileShader((IProgram::ShaderType)i);
		}
	}

	// if compiled, link program
	if (p->areCompiled())
		p->linkProgram();

	updateShaderAux();

	// Display logs
	m_Log->Clear();

	for (int i = 0; i < IProgram::SHADER_COUNT; ++i) {

		infoLog = p->getShaderInfoLog((IProgram::ShaderType)i);
		updateLogAux(infoLog);
	}
	infoLog = p->getProgramInfoLog();
	updateLogAux(infoLog);

	// notify  
	notifyUpdate(PROPS_CHANGED,m_Active,"Compiled");
	notifyUpdate(PROPS_CHANGED, m_Active, "Linked");
}


void 
DlgShaders::OnProcessValidateShaders(wxCommandEvent& event){

	std::string infoLog;
	IProgram *p = RESOURCEMANAGER->getProgram(m_Active);
	wxString s;

	if (p->programValidate())
		s = wxT("Validate: OK");
	else
		s = wxT("Not a Valid Program");
	m_Log->Clear();
	m_Log->Append(s);
	infoLog = p->getProgramInfoLog();
	updateLogAux(infoLog);
	if (p->isLinked())
		notifyUpdate(PROPS_CHANGED, m_Active, "Linked");
}


void 
DlgShaders::updateInfo(std::string name) {

	if (name == m_Active) {
		update();
	}
}


void 
DlgShaders::OnAdd(wxCommandEvent& event) {

	int result;
	bool nameUnique,exit = false;
	std::string name;

	do {
		wxTextEntryDialog dialog(this,
								 _T("(the name must be unique)\n")
								 _T("Please Input a Program Name"),
								 _T("Program Name\n"),
								 _T(""),
								 wxOK | wxCANCEL);

		result = dialog.ShowModal();
		name = std::string(dialog.GetValue().mb_str());
		nameUnique =  !RESOURCEMANAGER->hasProgram(name); 

		if (!nameUnique && (result == wxID_OK)){
			wxMessageBox(_T("Program name must be unique") , _T("Program Name Error"), wxOK | wxICON_INFORMATION, this);
		}
		if (name == "" && (result == wxID_OK)){
			wxMessageBox(_T("Program name must not be void"), _T("Program Name Error"), wxOK | wxICON_INFORMATION, this);
		}

		if (result == wxID_CANCEL) {
			exit = true;
		}
		else if (nameUnique && name != "") {
			exit = true;
		}

	} while (!exit);

	if (result == wxID_OK) {
		RESOURCEMANAGER->getProgram(name);
		updateList();
		m_List->Select(m_List->FindString(wxString(name.c_str())));
		m_Active = name;
		update();
		notifyUpdate(NEW_SHADER,name,"");

	}
}

void 
DlgShaders::OnListSelect(wxCommandEvent& event) {

	int sel;

	sel = event.GetSelection();
	m_Active = std::string(m_List->GetString(sel).mb_str());
	update();
}

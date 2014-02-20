
/* ----------------------------------------------------------------------

	Shaders Stuff

-----------------------------------------------------------------------*/



void DlgMaterials::setupShaderPanel(wxSizer *siz, wxWindow *parent) {

	wxStaticBox *shaderSB = new wxStaticBox(parent,-1,"Shaders");
	wxSizer *sizerS = new wxStaticBoxSizer(shaderSB,wxVERTICAL);

	pgShaderFiles = new wxPropertyGridManager(parent, DLG_SHADER_FILES,
				wxDefaultPosition, wxSize(300,50),
				// These and other similar styles are automatically
				// passed to the embedded wxPropertyGrid.
				wxPG_BOLD_MODIFIED|
				// Plus defaults.
				wxPGMAN_DEFAULT_STYLE
           );

	pgShaderFiles->AddPage(wxT("Shader Files"));

    m_vf = new wxFileProperty( wxT("Vertex Shader File"), wxT("VertexFile") );
    m_vf->SetAttribute(wxPG_FILE_WILDCARD,wxT("Vertex Shader Files (*.vert)|*.vert"));
	m_vf->SetAttribute(wxPG_FILE_SHOW_RELATIVE_PATH,wxT(m_project->m_path.c_str()));
	pgShaderFiles->Append(m_vf);


    m_ff = new wxFileProperty( wxT("Fragment Shader File"), wxT("FragmentFile") );
    m_ff->SetAttribute(wxPG_FILE_WILDCARD,wxT("Fragment Shader Files (*.frag)|*.frag"));
	m_ff->SetAttribute(wxPG_FILE_SHOW_RELATIVE_PATH,wxT(m_project->m_path.c_str()));
    pgShaderFiles->Append(m_ff);

	wxSizer *sizerS_SB = new wxBoxSizer(wxHORIZONTAL);
		
	m_bValidate = new wxButton(parent,DLG_SHADER_VALIDATE,"Validate");
	m_bCompile = new wxButton(parent,DLG_SHADER_COMPILE,"Compile");
	m_bLink = new wxButton(parent,DLG_SHADER_LINK,"Link");

	m_cbUseShader = new wxCheckBox(parent,DLG_SHADER_USE,"Use Shaders");
	m_cbUseShader->SetValue(0);

	sizerS_SB->Add(m_cbUseShader,0,wxALIGN_CENTER,5);
	sizerS_SB->Add(m_bValidate,0,wxGROW|wxALL|wxALIGN_CENTER,5);
	sizerS_SB->Add(m_bCompile,0,wxGROW|wxALL|wxALIGN_CENTER,5);
	sizerS_SB->Add(m_bLink,0,wxGROW|wxALL|wxALIGN_CENTER,5);

	sizerS->Add(pgShaderFiles,1, wxEXPAND|wxALL|wxALIGN_CENTER,5);
	sizerS->Add(sizerS_SB,0,wxGROW|wxALL|wxALIGN_CENTER,5);
	//sizerS->RecalcSizes();

	/* Uniforms */
	wxStaticBox *shaderU = new wxStaticBox(parent,-1,"Uniforms");
	wxSizer *sizerU = new wxStaticBoxSizer(shaderU,wxVERTICAL);

	pgShaderUniforms = new wxPropertyGridManager(parent, DLG_SHADER_UNIFORMS,
				wxDefaultPosition, wxDefaultSize,
				// These and other similar styles are automatically
				// passed to the embedded wxPropertyGrid.
				wxPG_BOLD_MODIFIED|wxPG_SPLITTER_AUTO_CENTER|
				// Plus defaults.
				wxPGMAN_DEFAULT_STYLE
           );

	pgShaderUniforms->AddPage(wxT("Shader Uniforms"));

	m_cbShowGlobalU = new wxCheckBox(parent,DLG_SHADER_SHOW_GLOBAL_UNIFORMS,"Show Global Uniforms");
	m_cbShowGlobalU->SetValue(0);


	sizerU->Add(pgShaderUniforms,0,wxGROW | wxALL|wxALIGN_CENTER,5);
	sizerU->Add(m_cbShowGlobalU,0,wxGROW | wxALL|wxALIGN_CENTER,5);
	/* Logs */

	wxStaticBox *shaderL = new wxStaticBox(parent,-1,"Log");
	wxSizer *sizerL = new wxStaticBoxSizer(shaderL,wxVERTICAL);

	m_log = new wxListBox(parent,DLG_SHADER_LOG,wxDefaultPosition,wxSize(100,150),0,NULL,wxLB_SINGLE | wxLB_HSCROLL | wxEXPAND);

	sizerL->Add(m_log,0,wxGROW|wxALL|wxALIGN_CENTER,5);



	siz->Add(sizerS,0,wxGROW|wxALL|wxALIGN_CENTER,5);
	siz->Add(sizerU,0,wxGROW|wxALL|wxALIGN_CENTER,5);
	siz->Add(sizerL,0,wxGROW|wxALL|wxALIGN_CENTER,5);

	updateShader(getModelMaterial());

}


void DlgMaterials::updateShader(CMaterial *m){

	CProgram p = m->shader;
	std::string vfn = p.getVertexFilename(),ffn = p.getFragmentFilename();

	pgShaderFiles->ClearSelection();

	pgShaderFiles->SetPropertyValue(m_vf,(char *)vfn.c_str());
	pgShaderFiles->SetPropertyValue(m_ff,(char *)ffn.c_str());

	updateShaderAux(m);
}

void DlgMaterials::updateShaderAux(CMaterial *m) {

	CProgram p = m->shader;
	std::string vfn = p.getVertexFilename(),ffn = p.getFragmentFilename();

	if (vfn != "" || ffn != "") {
		m_bCompile->Enable();
	}
	else {
		m_bCompile->Disable();
	}

	m_cbUseShader->SetValue(m->shader.m_useShader);


	if (p.m_vCompiled && p.m_fCompiled)
		m_bLink->Enable();
	else
		m_bLink->Disable();

	if (p.m_pLinked)
		m_bValidate->Enable();
	else
		m_bValidate->Disable();

	

	CUniform u;
	pgShaderUniforms->ClearPage(0);
	for (int i = 0; i < p.getNumberOfUniforms(); i++) {

		u = p.getUniform(i);
		addUniform(u,m->shader.m_showGlobalUniforms);
	}
	pgShaderUniforms->Refresh();

	m_log->Clear();

	m_parent->Refresh();
}


void DlgMaterials::addUniform(CUniform &u, int showGlobal) {

 //   wxFloatPropertyValidator float_validator(-1.0,1.0);

	int edit = strncmp("gl_",u.m_name.c_str(),3);
	if ((!showGlobal) && (edit == 0))
			return;

	wxPGId pid,pid2;
	const wxChar* units[] = {"0","1","2","3","4","5","6","7",NULL};
	const long unitsInd[] = {0,1,2,3,4,5,6,7};


	pid = pgShaderUniforms->Append(new wxParentProperty((u.m_name.c_str()),wxPG_LABEL));
	pgShaderUniforms->LimitPropertyEditing(pid);
	pid2 = pgShaderUniforms->AppendIn(pid,new wxStringProperty(wxT("type"),wxPG_LABEL,getUniformType(u.getType())));
	pgShaderUniforms->DisableProperty(pid2);
//	pgShaderUniforms->AppendIn(pid,wxIntProperty(wxT("loc"),wxPG_LABEL,u.m_loc));
	
	switch(u.getType()) {


		case GL_FLOAT:
			pgShaderUniforms->AppendIn(pid,new wxFloatProperty(wxT("fvalue"),wxPG_LABEL,u.m_values[0]));
			break;
		case GL_FLOAT_VEC2:
			pid2 = pgShaderUniforms->AppendIn(pid,new wxParentProperty("fvalues",wxPG_LABEL));
			pgShaderUniforms->LimitPropertyEditing(pid2);
			if (!edit)
				pgShaderUniforms->DisableProperty(pid2);
			pgShaderUniforms->AppendIn(pid2,new wxFloatProperty(wxT("v[0]"),wxPG_LABEL,u.m_values[0]));
			pgShaderUniforms->AppendIn(pid2,new wxFloatProperty(wxT("v[1]"),wxPG_LABEL,u.m_values[1]));
			break;
		case GL_FLOAT_VEC3:
			pid2 = pgShaderUniforms->AppendIn(pid,new wxParentProperty("fvalues",wxPG_LABEL));
			pgShaderUniforms->LimitPropertyEditing(pid2);
			pgShaderUniforms->AppendIn(pid2,new wxFloatProperty(wxT("v[0]"),wxPG_LABEL,u.m_values[0]));
			pgShaderUniforms->AppendIn(pid2,new wxFloatProperty(wxT("v[1]"),wxPG_LABEL,u.m_values[1]));
			pgShaderUniforms->AppendIn(pid2,new wxFloatProperty(wxT("v[2]"),wxPG_LABEL,u.m_values[2]));
			break;
		case GL_FLOAT_VEC4:
			pid2 = pgShaderUniforms->AppendIn(pid,new wxParentProperty("fvalues",wxPG_LABEL));
			pgShaderUniforms->LimitPropertyEditing(pid2);
			pgShaderUniforms->AppendIn(pid2,new wxFloatProperty(wxT("v[0]"),wxPG_LABEL,u.m_values[0]));
			pgShaderUniforms->AppendIn(pid2,new wxFloatProperty(wxT("v[1]"),wxPG_LABEL,u.m_values[1]));
			pgShaderUniforms->AppendIn(pid2,new wxFloatProperty(wxT("v[2]"),wxPG_LABEL,u.m_values[2]));
			pgShaderUniforms->AppendIn(pid2,new wxFloatProperty(wxT("v[3]"),wxPG_LABEL,u.m_values[3]));
			break;

		case GL_INT: 
		case GL_BOOL:
			pgShaderUniforms->AppendIn(pid,new wxIntProperty(wxT("ivalue"),wxPG_LABEL,u.m_values[0]));
			break;
		case GL_INT_VEC2:
		case GL_BOOL_VEC2:
			pid2 = pgShaderUniforms->AppendIn(pid,new wxParentProperty("ivalues",wxPG_LABEL));
			pgShaderUniforms->LimitPropertyEditing(pid2);
			pgShaderUniforms->AppendIn(pid2,new wxIntProperty(wxT("v[0]"),wxPG_LABEL,u.m_values[0]));
			pgShaderUniforms->AppendIn(pid2,new wxIntProperty(wxT("v[1]"),wxPG_LABEL,u.m_values[1]));
			break;
		case GL_INT_VEC3:
		case GL_BOOL_VEC3:
			pid2 = pgShaderUniforms->AppendIn(pid,new wxParentProperty("ivalues",wxPG_LABEL));
			pgShaderUniforms->LimitPropertyEditing(pid2);
			pgShaderUniforms->AppendIn(pid2,new wxIntProperty(wxT("v[0]"),wxPG_LABEL,u.m_values[0]));
			pgShaderUniforms->AppendIn(pid2,new wxIntProperty(wxT("v[1]"),wxPG_LABEL,u.m_values[1]));
			pgShaderUniforms->AppendIn(pid2,new wxIntProperty(wxT("v[2]"),wxPG_LABEL,u.m_values[2]));
			break;
		case GL_INT_VEC4:
		case GL_BOOL_VEC4:
			pid2 = pgShaderUniforms->AppendIn(pid,new wxParentProperty("ivalues",wxPG_LABEL));
			pgShaderUniforms->LimitPropertyEditing(pid2);
			pgShaderUniforms->AppendIn(pid2,new wxIntProperty(wxT("v[0]"),wxPG_LABEL,u.m_values[0]));
			pgShaderUniforms->AppendIn(pid2,new wxIntProperty(wxT("v[1]"),wxPG_LABEL,u.m_values[1]));
			pgShaderUniforms->AppendIn(pid2,new wxIntProperty(wxT("v[2]"),wxPG_LABEL,u.m_values[2]));
			pgShaderUniforms->AppendIn(pid2,new wxIntProperty(wxT("v[3]"),wxPG_LABEL,u.m_values[3]));
			break;
 		case GL_SAMPLER_1D:
		case GL_SAMPLER_2D:
		case GL_SAMPLER_3D:
		case GL_SAMPLER_CUBE:
			pgShaderUniforms->AppendIn(pid,new wxEnumProperty(wxT("texture unit"),wxPG_LABEL,units,unitsInd,(int)u.m_values[0]));
			break;




// FALTAM AS MATRIZES

	}
}

wxString DlgMaterials::getUniformType(int type) {

	wxString s;
	switch (type) {
		case GL_FLOAT:s = "float";break;
		case GL_FLOAT_VEC2:s = "vec2";break;
		case GL_FLOAT_VEC3:s = "vec3";break;
		case GL_FLOAT_VEC4:s = "vec4";break;

		case GL_INT: s = "int";break;
		case GL_INT_VEC2: s = "ivec2";break;
		case GL_INT_VEC3: s = "ivec3";break;
		case GL_INT_VEC4: s = "ivec4";break;

		case GL_BOOL: s = "bool";break;
		case GL_BOOL_VEC2: s = "bvec2";break;
		case GL_BOOL_VEC3: s = "bvec3";break;
		case GL_BOOL_VEC4: s = "bvec4";break;

		case GL_SAMPLER_2D: s = "sampler2D";break;
		case GL_SAMPLER_3D: s = "sampler3D";break;
		case GL_SAMPLER_1D: s = "sampler1D";break;
		case GL_SAMPLER_CUBE: s = "samplerCube";break;

		case GL_FLOAT_MAT2: s = "mat2";break;
		case GL_FLOAT_MAT3: s = "mat3";break;
		case GL_FLOAT_MAT4: s = "mat4";break;
	}
	return s;
}



void DlgMaterials::OnProcessShaderFileChange( wxPropertyGridEvent& e){

	std::string fn;
	const wxString& name = e.GetPropertyName();
	wxString filename = pgShaderFiles->GetPropertyValueAsString(name);
	fn = (char *)filename.c_str();
	
	CMaterial *m = getModelMaterial();
	if (name == "VertexFile")
		m->shader.setVertexShaderFile(fn);
	else
		m->shader.setFragmentShaderFile(fn);

	updateShaderAux(m);
}



void DlgMaterials::OnProcessCompileShaders(wxCommandEvent& event){

	CMaterial *m = getModelMaterial();

	if (m->shader.getVertexFilename() != "") {
		m->shader.reloadVertexShaderFile();
		m->shader.compileVertexShader();
	}
	
	if (m->shader.getFragmentFilename() != "") {
		m->shader.reloadFragmentShaderFile();
		m->shader.compileFragmentShader();
	}

	updateShaderAux(m);

	m_log->Append(m->shader.getVertexInfoLog());
	m_log->Append(m->shader.getFragmentInfoLog());
}




void DlgMaterials::OnProcessUseShader(wxCommandEvent& event){

	CMaterial *m = getModelMaterial();

	m->shader.m_useShader = event.IsChecked();

	m_parent->Refresh();
}


void DlgMaterials::OnProcessShowGlobalUniforms(wxCommandEvent& event){

	CMaterial *m = getModelMaterial();
	CUniform u;

	m->shader.m_showGlobalUniforms = event.IsChecked();
	pgShaderUniforms->ClearPage(0);

	for (int i = 0; i < m->shader.getNumberOfUniforms(); i++) {

		u = m->shader.getUniform(i);
		addUniform(u,m->shader.m_showGlobalUniforms);
	}
	pgShaderUniforms->Refresh();

}



void DlgMaterials::OnProcessLinkShaders(wxCommandEvent& event){

	CMaterial *m = getModelMaterial();

	m->shader.linkProgram();

	updateShaderAux(m);

	m_log->Append(m->shader.getProgramInfoLog());
}



void DlgMaterials::OnProcessValidateShaders(wxCommandEvent& event){

	CMaterial *m = getModelMaterial();
	int res;
	wxString s;

	res = m->shader.programValidate();
	
	if (res)
		s = "Validate: OK";
	else
		s = "Not a Valid Program";
	m_log->Clear();
	m_log->Append(s);
	m_log->Append(m->shader.getProgramInfoLog());

	updateUniforms(m);

}

void DlgMaterials::updateUniforms(CMaterial *m) {

	CUniform u;
	pgShaderUniforms->ClearPage(0);
	m->shader.updateUniforms();
	for (int i = 0; i < m->shader.getNumberOfUniforms(); i++) {

		u = m->shader.getUniform(i);
		addUniform(u,m->shader.m_showGlobalUniforms);
	}
	pgShaderUniforms->Refresh();
}
		

void DlgMaterials::OnProcessShaderUpdateUniforms( wxPropertyGridEvent& e){

	wxPGId id, id2;
	wxString s,name,parName;
	CMaterial *m = getModelMaterial(); 

	id = e.GetProperty();
	id2 = pgShaderUniforms->GetPropertyParent(id);

	name = e.GetPropertyName();
	parName = pgShaderUniforms->GetPropertyName(id2);
	
	m->shader.useProgram();
	if (name == "texture unit") {
		m->shader.setValueOfUniformByName((char *)parName.c_str(),e.GetPropertyValueAsInt());
	}
	else if (name == "fvalue") {
		m->shader.setValueOfUniformByName((char *)parName.c_str(),e.GetPropertyValueAsDouble());
	}
	else if (name == "ivalue") {
		m->shader.setValueOfUniformByName((char *)parName.c_str(),e.GetPropertyValueAsInt());
	}
	else if (parName == "ivalues"){ // change vecs or mats
		id2 = pgShaderUniforms->GetPropertyParent(id2);
		parName = pgShaderUniforms->GetPropertyName(id2);

		float values[16];
		CUniform uni = m->shader.getUniform((char *)parName.c_str());
		for (int j = 0; j < uni.getCardinality(); j++) {

			s.Printf("%s.ivalues.v[%d]",parName,j);
			values[j] = (float)pgShaderUniforms->GetPropertyValueAsInt(s);
		}
		m->shader.setValueOfUniformByName((char *)parName.c_str(),values);
	}
	else if (parName == "fvalues"){ // change vecs or mats
		id2 = pgShaderUniforms->GetPropertyParent(id2);
		parName = pgShaderUniforms->GetPropertyName(id2);

		float values[16];
		CUniform uni = m->shader.getUniform((char *)parName.c_str());
		for (int j = 0; j < uni.getCardinality(); j++) {

			s.Printf("%s.fvalues.v[%d]",parName,j);
			values[j] = (float)pgShaderUniforms->GetPropertyValueAsDouble(s);
		}
		m->shader.setValueOfUniformByName((char *)parName.c_str(),values);
	}

	m_parent->Refresh();

}


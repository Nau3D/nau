
/* ----------------------------------------------------------------------

	Materials Stuff

-----------------------------------------------------------------------*/



void DlgMaterials::setupColorPanel(wxSizer *siz, wxWindow *parent) {

	wxColour col;


	pgMaterial = new wxPropertyGridManager(parent, DLG_MI_PGMAT,
				wxDefaultPosition, wxDefaultSize,
				// These and other similar styles are automatically
				// passed to the embedded wxPropertyGrid.
				wxPG_BOLD_MODIFIED|
				// Plus defaults.
				wxPGMAN_DEFAULT_STYLE
           );

	pgMaterial->AddPage(wxT("Material Items"));

	wxPGId pid = pgMaterial->Append(new wxParentProperty(wxT("GL_DIFFUSE"),wxPG_LABEL));
	pgMaterial->AppendIn(pid,new wxColourProperty(wxT("RGB"),wxPG_LABEL,
					wxColour(255,255,255)));
	pgMaterial->AppendIn(pid,new wxFloatProperty(wxT("Alpha"),wxPG_LABEL,1.0));
	pgMaterial->Expand(pid);

	pid = pgMaterial->Append(new wxParentProperty(wxT("GL_AMBIENT"),wxPG_LABEL));
	pgMaterial->AppendIn(pid,new wxColourProperty(wxT("RGB"),wxPG_LABEL,
					wxColour(255,255,255)));
	pgMaterial->AppendIn(pid,new wxFloatProperty(wxT("Alpha"),wxPG_LABEL,1.0));
	pgMaterial->Expand(pid);

	pid = pgMaterial->Append(new wxParentProperty(wxT("GL_SPECULAR"),wxPG_LABEL));
	pgMaterial->AppendIn(pid,new wxColourProperty(wxT("RGB"),wxPG_LABEL,
					wxColour(255,255,255)));
	pgMaterial->AppendIn(pid,new wxFloatProperty(wxT("Alpha"),wxPG_LABEL,1.0));
	pgMaterial->Expand(pid);

	pid = pgMaterial->Append(new wxParentProperty(wxT("GL_EMISSION"),wxPG_LABEL));
	pgMaterial->AppendIn(pid,new wxColourProperty(wxT("RGB"),wxPG_LABEL,
					wxColour(255,255,255)));
	pgMaterial->AppendIn(pid,new wxFloatProperty(wxT("Alpha"),wxPG_LABEL,1.0));
	pgMaterial->Expand(pid);

	pgMaterial->Append(new wxFloatProperty(wxT("GL_SHININESS"),wxPG_LABEL,0));

	siz->Add(pgMaterial,1,wxEXPAND);
//	siz->Add(pgMaterial,0,wxGROW | wxALL,5);

	updateColors(getModelMaterial()); // what is default material?


}


void DlgMaterials::OnProcessColorChange( wxPropertyGridEvent& e){

	CMaterial *mm;
	wxColour *col;
	double f;
	wxArrayInt a;

	mm = getModelMaterial();

	col = (wxColour *)pgMaterial->GetPropertyValueAsWxObjectPtr("GL_DIFFUSE.RGB");
	f = pgMaterial->GetPropertyValueAsDouble("GL_DIFFUSE.Alpha");

	mm->color.m_diffuse[0] = col->Red()/255.0;
	mm->color.m_diffuse[1] = col->Green()/255.0;
	mm->color.m_diffuse[2] = col->Blue()/255.0;
	mm->color.m_diffuse[3] = f;

	col = (wxColour *)pgMaterial->GetPropertyValueAsWxObjectPtr("GL_AMBIENT.RGB");
	f = pgMaterial->GetPropertyValueAsDouble("GL_AMBIENT.Alpha");

	mm->color.m_ambient[0] = col->Red()/255.0;
	mm->color.m_ambient[1] = col->Green()/255.0;
	mm->color.m_ambient[2] = col->Blue()/255.0;
	mm->color.m_ambient[3] = f;

	col = (wxColour *)pgMaterial->GetPropertyValueAsWxObjectPtr("GL_SPECULAR.RGB");
	f = pgMaterial->GetPropertyValueAsDouble("GL_SPECULAR.Alpha");

	mm->color.m_specular[0] = col->Red()/255.0;
	mm->color.m_specular[1] = col->Green()/255.0;
	mm->color.m_specular[2] = col->Blue()/255.0;
	mm->color.m_specular[3] = f;

	col = (wxColour *)pgMaterial->GetPropertyValueAsWxObjectPtr("GL_EMISSION.RGB");
	f = pgMaterial->GetPropertyValueAsDouble("GL_EMISSION.Alpha");

	mm->color.m_emissive[0] = col->Red()/255.0;
	mm->color.m_emissive[1] = col->Green()/255.0;
	mm->color.m_emissive[2] = col->Blue()/255.0;
	mm->color.m_emissive[3] = f;

	mm->color.m_shininess = pgMaterial->GetPropertyValueAsDouble("GL_SHININESS");

	m_parent->Refresh();
}




void DlgMaterials::updateColors(CMaterial *mm) {

	pgMaterial->ClearSelection();
	pgMaterial->SetPropertyValue("GL_DIFFUSE.RGB",
					wxColour(255*mm->color.m_diffuse[0],255*mm->color.m_diffuse[1],255*mm->color.m_diffuse[2]));
	pgMaterial->SetPropertyValue("GL_DIFFUSE.Alpha",mm->color.m_diffuse[3]);

	pgMaterial->SetPropertyValue("GL_AMBIENT.RGB",
					wxColour(255*mm->color.m_ambient[0],255*mm->color.m_ambient[1],255*mm->color.m_ambient[2]));
	pgMaterial->SetPropertyValue("GL_AMBIENT.Alpha",mm->color.m_ambient[3]);

	pgMaterial->SetPropertyValue("GL_SPECULAR.RGB",
					wxColour(255*mm->color.m_specular[0],255*mm->color.m_specular[1],255*mm->color.m_specular[2]));
	pgMaterial->SetPropertyValue("GL_SPECULAR.Alpha",mm->color.m_specular[3]);

	pgMaterial->SetPropertyValue("GL_EMISSION.RGB",
					wxColour(255*mm->color.m_emissive[0],255*mm->color.m_emissive[1],255*mm->color.m_emissive[2]));
	pgMaterial->SetPropertyValue("GL_EMISSION.Alpha",mm->color.m_emissive[3]);

	pgMaterial->SetPropertyValue("GL_SHININESS",mm->color.m_shininess);
		
	pgMaterial->Refresh();

}















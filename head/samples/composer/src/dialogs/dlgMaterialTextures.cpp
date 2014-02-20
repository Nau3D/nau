/* ----------------------------------------------------------------------

	Textures Stuff

-----------------------------------------------------------------------*/


void DlgMaterials::setupTexturesPanel(wxSizer *siz, wxWindow *parent) {

	Material *mm = getModelMaterial();

	gridTextures = new wxGrid(parent,DLG_MI_TEXTURE_GRID,wxDefaultPosition,
                       wxSize( 420, 210 ));

	gridTextures->CreateGrid(2,4);
	gridTextures->SetColLabelSize(0);
	gridTextures->SetRowLabelSize(0);
	gridTextures->SetColSize(0,100);
	gridTextures->SetColSize(1,100);
	gridTextures->SetColSize(2,100);
	gridTextures->SetColSize(3,100);
	gridTextures->SetRowMinimalAcceptableHeight(100);
	gridTextures->SetColMinimalAcceptableWidth(100);

	for(int i = 0; i < 2; i++) { 
		gridTextures->SetRowHeight(i,100);
		for(int j = 0 ; j < 4 ; j++){

			Texture *texture = mm->m_texmat.getTexture(i);
	
			gridTextures->SetReadOnly(i,j,true);
			if (texture != NULL)
				imagesGrid[i*4+j] = new ImageGridCellRenderer(texture->bitmap);
			else
				imagesGrid[i*4+j] = new ImageGridCellRenderer(new wxBitmap(96,96));
			gridTextures->SetCellRenderer(i, j, imagesGrid[i*4+j]);
		}
	}

	wxStaticBox *texProp = new wxStaticBox(parent,-1,"Texture Properties");
	wxSizer *sizerTP = new wxStaticBoxSizer(texProp,wxVERTICAL);

	pgTextureProps = new wxPropertyGridManager(parent, DLG_MI_PGTEXTPROPS,
				wxDefaultPosition, wxDefaultSize,
				// These and other similar styles are automatically
				// passed to the embedded wxPropertyGrid.
				wxPG_BOLD_MODIFIED|
				// Plus defaults.
				wxPGMAN_DEFAULT_STYLE
           );

	pgTextureProps->AddPage(wxT("Texture"));

	const wxChar*  repeat[] =  {"GL_REPEAT","GL_CLAMP",NULL};
	const long repeatInd[] = {GL_REPEAT,GL_CLAMP}; 

	const wxChar* filtersMag[] =  {"GL_LINEAR","GL_NEAREST",NULL};
	const long filtersIndMag[] =  {GL_LINEAR,GL_NEAREST};
			
	const wxChar* filtersMin[] =  {"GL_LINEAR","GL_NEAREST",
		"GL_NEAREST_MIPMAP_NEAREST","GL_NEAREST_MIPMAP_LINEAR",
		"GL_LINEAR_MIPMAP_NEAREST","GL_LINEAR_MIPMAP_LINEAR",NULL};
	const long filtersIndMin[] =  {GL_LINEAR,GL_NEAREST,GL_NEAREST_MIPMAP_NEAREST,GL_NEAREST_MIPMAP_LINEAR,
		GL_LINEAR_MIPMAP_NEAREST,GL_LINEAR_MIPMAP_LINEAR};

			
	const wxChar* units[] = {"0","1","2","3","4","5","6","7",NULL};
	const long unitsInd[] = {0,1,2,3,4,5,6,7};

	const wxChar* texType[] = { "GL_TEXTURE_1D", "GL_TEXTURE_2D", "GL_TEXTURE_3D", "GL_TEXTURE_CUBE_MAP",NULL};
	const long texTypeInd[] = { GL_TEXTURE_1D, GL_TEXTURE_2D, GL_TEXTURE_3D, GL_TEXTURE_CUBE_MAP};

	pgTextureProps->Append( new wxEnumProperty(wxT("Texture Unit"),wxPG_LABEL,units,unitsInd,0));

	wxPGId pgid;
	textureLabels.Add("No Texture",-1);
	pgid = pgTextureProps->Append(new wxEnumProperty("Name",wxPG_LABEL,textureLabels));

/*	pgTextureProps->Append( wxStringProperty(wxT("Name"),wxPG_LABEL,""));
 	pgTextureProps->DisableProperty(wxT("Name"));
*/
	pgTextureProps->Append( new wxEnumProperty(wxT("Texture Type"),wxPG_LABEL,texType,texTypeInd,GL_TEXTURE_2D));
	pgTextureProps->DisableProperty(wxT("Texture Type"));

	wxString texDim;
	texDim.Printf("%d x %d x %d",0,0,0);
	pgTextureProps->Append( new wxStringProperty(wxT("Dimensions(WxHxD)"),wxPG_LABEL,texDim));
 	pgTextureProps->DisableProperty(wxT("Dimensions(WxHxD)"));

	pgTextureProps->Append( new wxEnumProperty(wxT("GL_TEXTURE_WRAP_S"),wxPG_LABEL,repeat,repeatInd,0));
	pgTextureProps->Append( new wxEnumProperty(wxT("GL_TEXTURE_WRAP_T"),wxPG_LABEL,repeat,repeatInd,0));
	pgTextureProps->Append( new wxEnumProperty(wxT("GL_TEXTURE_WRAP_R"),wxPG_LABEL,repeat,repeatInd,0));

	pgTextureProps->Append( new wxEnumProperty(wxT("GL_TEXTURE_MAG_FILTER"),wxPG_LABEL,filtersMag,filtersIndMag,0));
	pgTextureProps->Append( new wxEnumProperty(wxT("GL_TEXTURE_MIN_FILTER"),wxPG_LABEL,filtersMin,filtersIndMin,0));


	sizerTP->Add(pgTextureProps,0,wxGROW|wxALL|wxALIGN_CENTER_HORIZONTAL,15);

	siz->Add(gridTextures,0,wxALL|wxALIGN_CENTER,5);

	siz->Add(sizerTP,0,wxALL|wxGROW,5);

	updateTextures(getModelMaterial(),0);
}


void DlgMaterials::setTextureUnit(int index){


	CMaterial *mm = getModelMaterial();
	CTexture *texture = mm->m_texmat.getTexture(index);

	pgTextureProps->SetPropertyValue(wxT("Texture Unit"),(long)index);

	int row,col;
	row = index / 4;
	col = index % 4;
	gridTextures->SelectBlock(row,col,row,col,false);

	updateTextureList();

	if (texture == NULL) {
		pgTextureProps->SelectProperty(wxT("Texture Unit"));
//		pgTextureProps->SetPropertyValueString(wxT("Name"),"No Texture");
	 	pgTextureProps->DisableProperty(wxT("GL_TEXTURE_WRAP_S"));
	 	pgTextureProps->DisableProperty(wxT("GL_TEXTURE_WRAP_T"));
	 	pgTextureProps->DisableProperty(wxT("GL_TEXTURE_WRAP_R"));
	 	pgTextureProps->DisableProperty(wxT("GL_TEXTURE_MAG_FILTER"));
	 	pgTextureProps->DisableProperty(wxT("GL_TEXTURE_MIN_FILTER"));
		return;
	}

//	pgTextureProps->SetPropertyValueString(wxT("Name"),(char *)(texture->m_label.c_str()));
	pgTextureProps->SetPropertyValue(wxT("Texture Type"),(int)texture->getTarget());
	pgTextureProps->EnableProperty(wxT("GL_TEXTURE_MAG_FILTER"));
	pgTextureProps->EnableProperty(wxT("GL_TEXTURE_MIN_FILTER"));
	
	wxString texDim;
	texDim.Printf("%d x %d x %d",texture->getWidth(),
								texture->getHeight(),
								texture->getDepth());
	pgTextureProps->SetPropertyValue(wxT("Dimensions(WxHxD)"),texDim);

	pgTextureProps->SetPropertyValue(wxT("GL_TEXTURE_WRAP_S"),(long)texture->getSRepeat());
	pgTextureProps->SetPropertyValue(wxT("GL_TEXTURE_WRAP_T"),(long)texture->getTRepeat());
	pgTextureProps->SetPropertyValue(wxT("GL_TEXTURE_WRAP_R"),(long)texture->getRRepeat());

	if (texture->getTarget() == GL_TEXTURE_1D) {

		pgTextureProps->EnableProperty(wxT("GL_TEXTURE_WRAP_S"));
	 	pgTextureProps->DisableProperty(wxT("GL_TEXTURE_WRAP_T"));
	 	pgTextureProps->DisableProperty(wxT("GL_TEXTURE_WRAP_R"));
	}
	else if (texture->getTarget() == GL_TEXTURE_2D)  {
		pgTextureProps->EnableProperty(wxT("GL_TEXTURE_WRAP_S"));
		pgTextureProps->EnableProperty(wxT("GL_TEXTURE_WRAP_T"));
	 	pgTextureProps->DisableProperty(wxT("GL_TEXTURE_WRAP_R"));
	}
	else {
		pgTextureProps->EnableProperty(wxT("GL_TEXTURE_WRAP_S"));
		pgTextureProps->EnableProperty(wxT("GL_TEXTURE_WRAP_T"));
		pgTextureProps->EnableProperty(wxT("GL_TEXTURE_WRAP_R"));
	}

	pgTextureProps->SetPropertyValue(wxT("GL_TEXTURE_MAG_FILTER"),(long)texture->getMagFilter());
	pgTextureProps->SetPropertyValue(wxT("GL_TEXTURE_MIN_FILTER"),(long)texture->getMinFilter());

}



void DlgMaterials::OnProcessTexturePropsChange( wxPropertyGridEvent& e) {

	CMaterial *mm = getModelMaterial();

	const wxString& name = e.GetPropertyName();
	const int index = pgTextureProps->GetPropertyValueAsLong("Texture Unit");
	const int value = pgTextureProps->GetPropertyValueAsLong(name);

	CTexture *texture = mm->m_texmat.getTexture(index);

	if (name == "Texture Unit") {
		
		setTextureUnit(index);
	}
	else if (name == "GL_TEXTURE_WRAP_S") {
		
		texture->setSRepeat(value);
	}
	else if (name == "GL_TEXTURE_WRAP_T") {

		texture->setTRepeat(value);
	}
	else if (name == "GL_TEXTURE_WRAP_R") {

		texture->setRRepeat(value);
	}
	else if (name == "GL_TEXTURE_MAG_FILTER") {

		texture->setMagFilter(value);
	}
	else if (name == "GL_TEXTURE_MIN_FILTER") {

		texture->setMinFilter(value);
	}
	else if (name == "Name") {

		int value = pgTextureProps->GetPropertyValueAsInt(wxT("Name"));
		if (value == -1) {
			mm->m_texmat.unsetTexture(index);
			imagesGrid[index]->setBitmap(new wxBitmap(96,96));
		}
		else if (value < 100){
			CTextureDevIL *t = CTextureManager::Instance()->getTextureOrdered(value);
			mm->m_texmat.setTexture(index,t);
			imagesGrid[index]->setBitmap(t->bitmap);

		}
		else {
			int pass = (value) / 100;
			int rt = value - pass * 100;
			pass--;
			CProject *p = CProject::Instance();
			mm->m_texmat.setTexture(index,&(p->m_passes[pass]->m_renderTargets[rt]));
		}
		setTextureUnit(index);
	}

	m_parent->Refresh();

	if (name != "Name") {

		DlgTextureLib *dtl = DlgTextureLib::Instance();
		dtl->updateTexInfo(CTextureManager::Instance()->getTexturePosition(texture));
	}
}


void DlgMaterials::updateTexture(CTexture *tex) {

	CMaterial *mm = getModelMaterial();

	const int index = pgTextureProps->GetPropertyValueAsLong("Texture Unit");

	CTexture *texture = mm->m_texmat.getTexture(index);

	if (texture == tex) {
		updateTextures(mm,index);
	}

}

void DlgMaterials::updateActiveTexture() {

	const int index = pgTextureProps->GetPropertyValueAsLong("Texture Unit");
	CMaterial *mm = getModelMaterial();

	updateTextures(mm,index);

}

void DlgMaterials::updateTextures(CMaterial *mm, int index) {

	CTextureDevIL *texture;
	for(int i = 0; i < 2; i++) { 
		for(int j = 0 ; j < 4 ; j++) {
			texture = (CTextureDevIL *)mm->m_texmat.getTexture(i*4+j);
			if (texture != NULL)
				imagesGrid[i*4+j]->setBitmap(texture->bitmap);
			else
				imagesGrid[i*4+j]->setBitmap(new wxBitmap(96,96));
		}
	}
	setTextureUnit(index);
	gridTextures->Refresh(true);
	m_parent->Refresh();	

}



void DlgMaterials::updateTextureList() {

	CTextureManager *texman = CTextureManager::Instance();
	int i,num = texman->getNumTextures();

	pgTextureProps->ClearSelection();
	textureLabels.RemoveAt(0,textureLabels.GetCount());
	textureLabels.Add("No Texture",-1);

	for (i = 0; i < num; i++) {
		textureLabels.Add((char *)texman->getTextureOrdered(i)->m_label.c_str(),i);
	}

	CProject *p = CProject::Instance();

	for (int pass = 0; pass < p->m_passes.size(); pass++) {
		for (int rt = 0; rt < p->m_passes[pass]->m_color; rt++) {

			textureLabels.Add((char *)p->m_passes[pass]->m_renderTargets[rt].m_label.c_str(),(pass+1)*100 + rt);
		}
	}

	CMaterial *mm = getModelMaterial();

	const int index = pgTextureProps->GetPropertyValueAsLong("Texture Unit");
	CTexture *texture = mm->m_texmat.getTexture(index);

	if (texture == NULL)
		pgTextureProps->SetPropertyValueString(wxT("Name"),"No Texture");
	else
		pgTextureProps->SetPropertyValueString(wxT("Name"),(char *)texture->m_label.c_str());

	pgTextureProps->Refresh(true);

}



void DlgMaterials::loadTextureDialog(wxGridEvent &e) {

	int row,col,index;

	col = e.GetCol();
	row = e.GetRow();

	index = row*4+col;

	wxString filter;

	filter = "Texture Files (*.gif, *.jpg) | *.gif;*.jpg";

    wxFileDialog dialog
                 (
                    this,
                    _T("Open Texture Unit"),
                    _T(""),
                    _T(""),
                    filter
                 );

 //	dialog.SetDirectory(wxGetHomeDir());

    if (dialog.ShowModal() == wxID_OK)
    {
		CMaterial *mm = getModelMaterial();
		mm->createTexture(index,(char *)dialog.GetPath().c_str());

/*
		CTextureDevIL *texture;
		texture = (CTextureDevIL *)mm->m_texmat.getTexture(index);
		mm->m_texmat.setTextureFilename(index,(char *)dialog.GetPath().c_str());
*/
		updateTextures(mm,index);
		DlgTextureLib::Instance()->updateDlg();
		updateTextureList();
		
	}
}


void DlgMaterials::OnprocessDClickGrid(wxGridEvent &e) {


	switch(e.GetId()) {
		case DLG_MI_TEXTURE_GRID:loadTextureDialog(e);
			break;
	}
}

void DlgMaterials::OnprocessClickGrid(wxGridEvent &e) {

	wxString s;

	int row = e.GetRow();
	int col = e.GetCol();

	setTextureUnit(row*4+col);
	
}
#include "dialogs/dlgtexturelib.h"
#include <nau/event/eventFactory.h>
#include <nau/material/teximage.h>
#include <nau/loader/textureloader.h>

#include "GL/glew.h"

using namespace nau::render;
using namespace nau::material;

BEGIN_EVENT_TABLE(DlgTextureLib, wxDialog)

	EVT_PG_CHANGED( DLG_MI_PGTEXTPROPS, DlgTextureLib::OnProcessTexturePropsChange )
	EVT_BUTTON(DLG_MI_BUTTON_ADD_TEX, DlgTextureLib::OnAddTex)
	EVT_BUTTON(DLG_MI_BUTTON_SAVE_RAW, DlgTextureLib::OnSaveRaw)
	EVT_BUTTON(DLG_MI_BUTTON_SAVE_PNG, DlgTextureLib::OnSavePNG)

	EVT_GRID_CELL_LEFT_DCLICK(DlgTextureLib::OnprocessDClickGrid)
	EVT_GRID_CMD_CELL_LEFT_CLICK(DLG_MI_TEXTURE_GRID,DlgTextureLib::OnprocessClickGrid)
	
END_EVENT_TABLE()

wxWindow *DlgTextureLib::parent = NULL;
DlgTextureLib *DlgTextureLib::inst = NULL;


void DlgTextureLib::SetParent(wxWindow *p) {

	parent = p;
}


DlgTextureLib* DlgTextureLib::Instance () {

	if (inst == NULL)
		inst = new DlgTextureLib();

	return inst;
}



DlgTextureLib::DlgTextureLib()
	: wxDialog(DlgTextureLib::parent, -1, wxString(wxT("Nau - Materials")),
	wxDefaultPosition,wxDefaultSize,wxRESIZE_BORDER|wxDEFAULT_DIALOG_STYLE),
	m_Name("DlgTextureLib")
                
{

	m_activeTexture = 0;

	wxBoxSizer *sizer = new wxBoxSizer(wxVERTICAL);

	setupTexturesPanel(sizer,this);

	SetAutoLayout(TRUE);
    SetSizer(sizer);

    sizer->SetSizeHints(this);
    sizer->Fit(this);

	this->SetTitle(wxT("Nau - Textures"));
}


void
DlgTextureLib::notifyUpdate(Notification aNot, std::string texName, std::string value) {

	// sends events on behalf of the light
	if (aNot == NEW_TEXTURE) {
		nau::event_::IEventData *e= nau::event_::EventFactory::create("String");
		e->setData(&texName);
		EVENTMANAGER->notifyEvent("NEW_TEXTURE", texName,"", e);
		delete e;
	}
}

void 
DlgTextureLib::updateDlg() {

	updateTextures(0);
	gridTextures->Refresh(true);
	m_parent->Refresh();	
}

void 
DlgTextureLib::setupTexturesPanel(wxSizer *siz, wxWindow *parent) {

	int numTex = RESOURCEMANAGER->getNumTextures(); 

	gridTextures = new wxGrid(parent,DLG_MI_TEXTURE_GRID,wxDefaultPosition,wxSize( 420, 125 ));

	gridTextures->CreateGrid(1,0);
	gridTextures->SetColLabelSize(0);
	gridTextures->SetRowLabelSize(0);
	//gridTextures->SetColSize(0,100);
	//gridTextures->SetColSize(1,100);
	//gridTextures->SetColSize(2,100);
	//gridTextures->SetColSize(3,100);
	gridTextures->SetRowMinimalAcceptableHeight(110);
	gridTextures->SetColMinimalAcceptableWidth(104);
	gridTextures->SetRowSize(0,110);

	//for (j = 0; j < 4; j++) {
	//	imagesGrid.push_back(new ImageGridCellRenderer(new wxBitmap(96,96)));
	//	gridTextures->SetColSize(j,104);
	//	gridTextures->SetReadOnly(0,j,true);
	//	gridTextures->SetCellRenderer(0, j, imagesGrid[j]);
	//}


	//wxStaticBox *texProp = new wxStaticBox(parent,-1,"Texture Properties");
	//wxSizer *sizerTP = new wxStaticBoxSizer(texProp,wxVERTICAL);

	pgTextureProps = new wxPropertyGridManager(parent, DLG_MI_PGTEXTPROPS,
				wxDefaultPosition, wxDefaultSize,
				// These and other similar styles are automatically
				// passed to the embedded wxPropertyGrid.
				wxPG_BOLD_MODIFIED|wxPG_SPLITTER_AUTO_CENTER|
				// Plus defaults.
				wxPGMAN_DEFAULT_STYLE
           );

	pgTextureProps->AddPage(wxT("Texture"));
	//wxPropertyGridPage* pgTexturePropsPage = pgTextureProps->GetPage("Texture");
	std::vector<int> dims = Texture::Attribs.getListValues(Texture::DIMENSION);
	std::vector<std::string> strDims = Texture::Attribs.getListString(Texture::DIMENSION);;

	int size = dims.size();

	wxPGChoices texType;
	for(int i = 0; i < size; i++) {
		texType.Add(wxString::FromAscii(strDims[i].c_str()),dims[i]);
	}

	dims = Texture::Attribs.getListValues(Texture::INTERNAL_FORMAT);
	strDims = Texture::Attribs.getListString(Texture::INTERNAL_FORMAT);;

	size = dims.size();
	wxPGChoices texFormat;
	for(int i = 0; i < size; i++) {
		texFormat.Add(wxString::FromAscii(strDims[i].c_str()),dims[i]);
	}


	pgTextureProps->Append(new wxStringProperty(wxT("Name"),wxPG_LABEL,wxT("")));
 	pgTextureProps->DisableProperty(wxT("Name"));

	pgTextureProps->Append(new wxEnumProperty(wxT("Type"),wxPG_LABEL,texType));
	pgTextureProps->DisableProperty(wxT("Type"));

	pgTextureProps->Append(new wxEnumProperty(wxT("Internal Format"),wxPG_LABEL,texFormat));
	pgTextureProps->DisableProperty(wxT("Internal Format"));

	wxString texDim;
	texDim.Printf(wxT("%d x %d x %d"),0,0,0);
	pgTextureProps->Append(new wxStringProperty(wxT("Dimensions(WxHxD)"),wxPG_LABEL,texDim));
 	pgTextureProps->DisableProperty(wxT("Dimensions(WxHxD)"));

	pgTextureProps->SetSplitterLeft(true);
	updateTextures(0);

	siz->Add(gridTextures,0,wxALL|wxGROW|wxHORIZONTAL|wxALIGN_CENTER_HORIZONTAL,5);

	wxBoxSizer *sizH2 = new wxBoxSizer(wxHORIZONTAL);
	sizH2->Add(pgTextureProps,1,wxALIGN_CENTER_HORIZONTAL | wxGROW | wxALL  ,5);
	siz->Add(sizH2,1,wxGROW|wxALL|wxEXPAND,5);

	//sizerTP->Add(pgTextureProps,0,wxALL |wxGROW|wxHORIZONTAL|wxALIGN_CENTER_HORIZONTAL,15);
	//siz->Add(sizerTP,0,wxALL |wxGROW|wxHORIZONTAL,5);

	wxBoxSizer *sizH = new wxBoxSizer(wxHORIZONTAL);

	bAddTex = new wxButton(this,DLG_MI_BUTTON_ADD_TEX,wxT("Add Texture"));
	bSaveRaw = new wxButton(this, DLG_MI_BUTTON_SAVE_RAW,wxT("Save Raw"));
	bSavePNG = new wxButton(this, DLG_MI_BUTTON_SAVE_PNG,wxT("Save PNG"));

	sizH-> Add(bAddTex,0,wxALL |wxGROW |wxHORIZONTAL|wxALIGN_CENTER_HORIZONTAL,5);
	sizH-> Add(bSaveRaw,0,wxALL |wxGROW |wxHORIZONTAL|wxALIGN_CENTER_HORIZONTAL,5);
	sizH-> Add(bSavePNG,0,wxALL |wxGROW |wxHORIZONTAL|wxALIGN_CENTER_HORIZONTAL,5);


	siz->Add(sizH,0,wxALL |wxGROW |wxHORIZONTAL,5);

	



}

void DlgTextureLib::updateTextures(int index) {

	int j,numTex = RESOURCEMANAGER->getNumTextures(); 
	int gridSize = imagesGrid.size();

	if (numTex > gridSize) {
		for (j = gridSize; j < numTex; j++) {
			imagesGrid.push_back(new ImageGridCellRenderer());
			gridTextures->AppendCols();
			gridTextures->SetColSize(j,104);
			gridTextures->SetCellRenderer(0, j, imagesGrid[j]);
			gridTextures->SetReadOnly(0,j,true);
		}
	} else if (numTex < gridSize){
		gridTextures->DeleteCols(numTex,gridSize-numTex);
		for (j = gridSize; j > numTex; j--)
			imagesGrid.pop_back();
	}

	

	nau::render::Texture *texture;
		
	for(j = 0 ; j < numTex ; j++) {
		texture = RESOURCEMANAGER->getTexture(j);
		wxBitmap *bm = texture->getBitmap();
		if (bm == NULL)
			imagesGrid[j]->setBitmap(new wxBitmap(96,96));
		else
			imagesGrid[j]->setBitmap(bm);
		//gridTextures->SetCellRenderer(0, j, imagesGrid[j]);
	}

//	for (j = numTex; j < 4; j++) {
//		imagesGrid[j]->setBitmap(new wxBitmap(96,96));
	//	gridTextures->SetCellRenderer(0, j, imagesGrid[j]);
//	}

	setTextureProps(index);

}



void DlgTextureLib::setTextureProps(int index){

	nau::render::Texture *texture = RESOURCEMANAGER->getTexture(index);
	m_activeTexture = index;

	gridTextures->SelectBlock(0,index,0,index,false);

	pgTextureProps->ClearSelection();

	if (texture == NULL) {
		pgTextureProps->SetPropertyValue(wxT("Name"),"");
		pgTextureProps->SetPropertyValue(wxT("Dimensions(WxHxD)"),"");
		return;
	}

		
	wxString s = wxString(texture->getLabel().c_str());
	pgTextureProps->SetPropertyValue(wxT("Name"),s);
	wxBitmap *bm = texture->getBitmap();
	//unsigned char *bm = texture->getBitmap();
	pgTextureProps->SetPropertyImage(wxT("Name"), *bm);

	int v = texture->getPrope(Texture::INTERNAL_FORMAT);
	pgTextureProps->SetPropertyValue(wxT("Internal Format"),v);

	pgTextureProps->SetPropertyValue(wxT("Type"),texture->getPrope(Texture::DIMENSION));

	v = texture->m_IntProps[Texture::DEPTH];
	//if (texture->getHeight() == 1 || texture->getWidth() == 1) 
	//	pgTextureProps->SetPropertyValue(wxT("Type"),Texture::TEXTURE_1D);
	//else if (texture->getDepth() == 0)
	//	pgTextureProps->SetPropertyValue(wxT("Type"),Texture::TEXTURE_2D);
	//else
	//	pgTextureProps->SetPropertyValue(wxT("Type"),Texture::TEXTURE_3D);

	wxString texDim;
	texDim.Printf(wxT("%d x %d x %d"),texture->m_IntProps[Texture::WIDTH],
								 texture->m_IntProps[Texture::HEIGHT],
								 v);

	pgTextureProps->SetPropertyValue(wxT("Dimensions(WxHxD)"),texDim);

}



void DlgTextureLib::OnProcessTexturePropsChange( wxPropertyGridEvent& e) {


	const wxString& name = e.GetPropertyName();
	const int index = m_activeTexture;
	const int value = pgTextureProps->GetPropertyValueAsLong(name);

	nau::render::Texture *texture = RESOURCEMANAGER->getTexture(index);

//	m_parent->Refresh();
//	DlgMaterials::Instance()->updateTexture(texture);
}


void DlgTextureLib::OnSaveRaw( wxCommandEvent& event) 
{
	nau::render::Texture *texture = RESOURCEMANAGER->getTexture(m_activeTexture);
	TexImage *ti = RESOURCEMANAGER->createTexImage(texture);
	
	void *data = ti->getData();

	int w = ti->getWidth();
	int h = ti->getHeight();
	int n = ti->getNumComponents();
	std::string type = ti->getType();

	// WHEN ADDING MORE TYPES MAKE SURE THEY EXIST  IN GLTEXIMAGE.CPP
	float *fData;
	unsigned int *uiData;
	unsigned short *usData;
	unsigned char *ubData;
	short *sData;
	char *cData;
	int *iData;
	if (type == "FLOAT")
		fData = (float *)data;
	else if (type == "UNSIGNED BYTE")
		ubData = (unsigned char *)data;
	else if (type == "UNSIGNED_SHORT")
		usData = (unsigned short *)data;
	else if (type == "UNSIGNED_INT")
		uiData = (unsigned int *)data;
	else if (type == "SHORT")
		sData = (short *)data;
	else if (type == "BYTE")
		cData = (char *)data;
	else if (type == "INT")
		iData = (int *)data;
 
	FILE *fp;
	char name[256];
	sprintf(name,"%s.raw", texture->getLabel().c_str());
	for (int i = 0; name[i] != '\0'; i++)
		if (name[i] == ':')
			name[i] = '_';

	fp = fopen(name, "wt+");

	for (int i = 0; i < w; i++) {
	
		for  (int j = 0; j < h; j++) {
		
			for (int k = 0; k < n; k++) {

				if (type == "FLOAT")
					fprintf(fp,"%f ",fData[(i*h + j)*n + k]);
				else if (type == "UNSIGNED_SHORT")
					fprintf(fp,"%u ",usData[(i*h + j)*n + k]);
				else if (type == "UNSIGNED_INT")
					fprintf(fp,"%u ",uiData[(i*h + j)*n + k]);
				else if (type == "SHORT")
					fprintf(fp,"%i ",sData[(i*h + j)*n + k]);

			}
			fprintf(fp,"; ");
		}
		fprintf(fp,"\n");
	}
	fclose(fp);

/*	ILuint ilFormat, ilType;
	if (n == 3 || n == 4) {
		if (n == 3)
			ilFormat = IL_RGB;
		else
			ilFormat = IL_RGBA;
		if (type == "FLOAT")
			ilType = IL_FLOAT;
		else if (type == "UNSIGNED_BYTE")
			ilType = IL_UNSIGNED_BYTE;
		ILuint image;
		ilInit();
		ilOriginFunc(IL_ORIGIN_LOWER_LEFT); 
		ilGenImages(1,&image);
		ilBindImage(image);
		ilTexImage(w,h,1,n,ilFormat,ilType,data);
		ilEnable(IL_FILE_OVERWRITE);

		ilSave(IL_JPG, "bla.jpg");
	}
*/
}

void DlgTextureLib::OnSavePNG( wxCommandEvent& event) 
{
	nau::render::Texture *texture = RESOURCEMANAGER->getTexture(m_activeTexture);
	TexImage *ti = RESOURCEMANAGER->createTexImage(texture);

	nau::loader::TextureLoader *loader = nau::loader::TextureLoader::create();
	loader->save(ti,"bla.png");
}


void DlgTextureLib::updateTexInfo(int pos) {

	if (pos == m_activeTexture) {
		setTextureProps(pos);
	}
}




/*
void DlgTextureLib::loadTextureDialog(wxGridEvent &e) {


	int index = m_textureManager->getNumTextures();

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
		int id = m_textureManager->addTexture((char *)dialog.GetPath().c_str());
		int c = gridTextures->GetNumberCols();
		if (c < m_textureManager->getNumTextures())  {
			gridTextures->SetReadOnly(0,c,true);
			imagesGrid.push_back(new ImageGridCellRenderer(new wxBitmap(96,96)));
			gridTextures->AppendCols();
			gridTextures->SetCellRenderer(0, c, imagesGrid[c]);
		}
		updateTextures(index);
		DlgMaterials::Instance()->updateTextureList();
	}
}
*/

void DlgTextureLib::OnprocessDClickGrid(wxGridEvent &e) {
/*
	switch(e.GetId()) {
		case DLG_MI_TEXTURE_GRID:loadTextureDialog(e);
			break;
	}
*/}

void DlgTextureLib::OnprocessClickGrid(wxGridEvent &e) {

	wxString s;

	int col = e.GetCol();

	setTextureProps(col);
	
}

void DlgTextureLib::OnAddTex(wxCommandEvent& event) {

	wxString filter;

	filter = wxT("Texture Files (*.gif, *.jpg, *.png, *.tif, *.tga) | *.gif;*.jpg;*.png;*.tif;*.tga");

    wxFileDialog dialog
                 (
                    this,
                    _T("Open Texture"),
                    _T(""),
                    _T(""),
                    filter
                 );

 //	dialog.SetDirectory(wxGetHomeDir());

    if (dialog.ShowModal() == wxID_OK)
    {
		RESOURCEMANAGER->addTexture(std::string(dialog.GetPath().mb_str()));
		updateTextures(0);
		gridTextures->Refresh();
		notifyUpdate(NEW_TEXTURE,std::string(dialog.GetPath().mb_str()),"");
//		DlgMaterials::Instance()->updateTextureList();
	}

}
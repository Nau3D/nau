#include "dialogs/dlgtexturelib.h"

#include "dialogs/propertyManager.h"

#include <nau/event/eventFactory.h>
#include <nau/material/iTexImage.h>
#include <nau/loader/iTextureLoader.h>
#include <nau/system/file.h>

#include <GL/glew.h>

using namespace nau::loader;
using namespace nau::render;
using namespace nau::material;

BEGIN_EVENT_TABLE(DlgTextureLib, wxDialog)

	EVT_PG_CHANGED( DLG_MI_PGTEXTPROPS, DlgTextureLib::OnProcessTexturePropsChange )
	EVT_BUTTON(DLG_MI_BUTTON_ADD_TEX, DlgTextureLib::OnAddTex)
	EVT_BUTTON(DLG_MI_BUTTON_SAVE_RAW, DlgTextureLib::OnSaveRaw)
	EVT_BUTTON(DLG_MI_BUTTON_SAVE_PNG, DlgTextureLib::OnSavePNG)
	EVT_BUTTON(DLG_MI_BUTTON_SAVE_HDR, DlgTextureLib::OnSaveHDR)
	EVT_BUTTON(DLG_MI_BUTTON_UPDATE_ICONS, DlgTextureLib::OnUpdateIcons)

	EVT_GRID_CELL_LEFT_DCLICK(DlgTextureLib::OnprocessDClickGrid)
	EVT_GRID_CMD_CELL_LEFT_CLICK(DLG_MI_TEXTURE_GRID,DlgTextureLib::OnprocessClickGrid)
	
END_EVENT_TABLE()

wxWindow *DlgTextureLib::Parent = NULL;
DlgTextureLib *DlgTextureLib::Inst = NULL;


void DlgTextureLib::SetParent(wxWindow *p) {

	Parent = p;
}


DlgTextureLib* DlgTextureLib::Instance () {

	if (Inst == NULL)
		Inst = new DlgTextureLib();

	return Inst;
}



DlgTextureLib::DlgTextureLib()
	: wxDialog(DlgTextureLib::Parent, -1, wxString(wxT("Nau - Materials")),
	wxDefaultPosition,wxDefaultSize,wxRESIZE_BORDER|wxDEFAULT_DIALOG_STYLE),
	m_Name("DlgTextureLib")
                
{
	m_EmptyBitmap = new wxBitmap(96, 96);

	m_ActiveTexture = 0;

	wxBoxSizer *sizer = new wxBoxSizer(wxVERTICAL);

	setupTexturesPanel(sizer,this);

	SetAutoLayout(TRUE);
    SetSizer(sizer);

    sizer->SetSizeHints(this);
    sizer->Fit(this);

	this->SetTitle(wxT("Nau - Textures"));
}


DlgTextureLib::~DlgTextureLib() {

	for (auto b : m_Bitmaps) {
		delete b.second;
	}
	m_Bitmaps.clear();

	delete m_EmptyBitmap;
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
	else if (aNot == TEXTURE_ICON_UPDATE) {
		nau::event_::IEventData *e= nau::event_::EventFactory::create("String");
		e->setData("");
		EVENTMANAGER->notifyEvent("NEW_TEXTURE", "","", e);
		delete e;
	}

}


void 
DlgTextureLib::updateDlg() {

	for (auto b : m_Bitmaps) {
		delete b.second;
	}
	m_Bitmaps.clear();

	updateTextures(0);
	m_GridTextures->Refresh(true);
	Parent->Refresh();	
}


void 
DlgTextureLib::setupTexturesPanel(wxSizer *siz, wxWindow *parent) {

	int numTex = RESOURCEMANAGER->getNumTextures(); 

	m_GridTextures = new wxGrid(parent,DLG_MI_TEXTURE_GRID,wxDefaultPosition,wxSize( 420, 125 ));

	m_GridTextures->CreateGrid(1,0);
	m_GridTextures->SetColLabelSize(0);
	m_GridTextures->SetRowLabelSize(0);
	//gridTextures->SetColSize(0,100);
	//gridTextures->SetColSize(1,100);
	//gridTextures->SetColSize(2,100);
	//gridTextures->SetColSize(3,100);
	m_GridTextures->SetRowMinimalAcceptableHeight(110);
	m_GridTextures->SetColMinimalAcceptableWidth(104);
	m_GridTextures->SetRowSize(0,110);

	//for (j = 0; j < 4; j++) {
	//	imagesGrid.push_back(new ImageGridCellRenderer(new wxBitmap(96,96)));
	//	gridTextures->SetColSize(j,104);
	//	gridTextures->SetReadOnly(0,j,true);
	//	gridTextures->SetCellRenderer(0, j, imagesGrid[j]);
	//}


	//wxStaticBox *texProp = new wxStaticBox(parent,-1,"ITexture Properties");
	//wxSizer *sizerTP = new wxStaticBoxSizer(texProp,wxVERTICAL);

	m_PGTextureProps = new wxPropertyGridManager(parent, DLG_MI_PGTEXTPROPS,
				wxDefaultPosition, wxDefaultSize,
				// These and other similar styles are automatically
				// passed to the embedded wxPropertyGrid.
				wxPG_BOLD_MODIFIED|wxPG_SPLITTER_AUTO_CENTER|
				// Plus defaults.
				wxPGMAN_DEFAULT_STYLE
           );

	m_PGTextureProps->AddPage(wxT("Texture"));
	m_PGTextureProps->Append(new wxStringProperty(wxT("Name"),wxPG_LABEL,wxT("")));
	m_PGTextureProps->DisableProperty(wxT("Name"));	
	
	std::vector<std::string> order = { "ID", "FORMAT", "TYPE", "INTERNAL_FORMAT", "WIDTH",
		"HEIGHT", "DEPTH", "LAYERS", "SAMPLES", "LEVELS" };

	PropertyManager::createOrderedGrid(m_PGTextureProps, ITexture::Attribs, order);
	PropertyManager::setAllReadOnly(m_PGTextureProps, ITexture::Attribs);
	m_PGTextureProps->SetSplitterLeft(true, true);

	siz->Add(m_GridTextures,0,wxALL|wxGROW|wxHORIZONTAL|wxALIGN_CENTER_HORIZONTAL,5);

	wxBoxSizer *sizH2 = new wxBoxSizer(wxHORIZONTAL);
	sizH2->Add(m_PGTextureProps,1,wxALIGN_CENTER_HORIZONTAL | wxGROW | wxALL  ,5);
	siz->Add(sizH2,1,wxGROW|wxALL|wxEXPAND,5);

	wxBoxSizer *sizH = new wxBoxSizer(wxHORIZONTAL);

	m_BAddTex = new wxButton(this,DLG_MI_BUTTON_ADD_TEX,wxT("Add Texture"));
	m_BSaveRaw = new wxButton(this, DLG_MI_BUTTON_SAVE_RAW,wxT("Save Raw"));
	m_BSavePNG = new wxButton(this, DLG_MI_BUTTON_SAVE_PNG,wxT("Save PNG"));
	m_BSaveHDR = new wxButton(this, DLG_MI_BUTTON_SAVE_HDR,wxT("Save HDR"));
	m_BSaveHDR = new wxButton(this, DLG_MI_BUTTON_SAVE_HDR,wxT("Save HDR"));
	m_BUpdateIcons = new wxButton(this, DLG_MI_BUTTON_UPDATE_ICONS,wxT("Update Icons"));

	sizH-> Add(m_BAddTex,0,wxALL |wxGROW |wxHORIZONTAL|wxALIGN_CENTER_HORIZONTAL,5);
	sizH-> Add(m_BSaveRaw,0,wxALL |wxGROW |wxHORIZONTAL|wxALIGN_CENTER_HORIZONTAL,5);
	sizH-> Add(m_BSavePNG,0,wxALL |wxGROW |wxHORIZONTAL|wxALIGN_CENTER_HORIZONTAL,5);
	sizH-> Add(m_BSaveHDR,0,wxALL |wxGROW |wxHORIZONTAL|wxALIGN_CENTER_HORIZONTAL,5);
	sizH-> Add(m_BUpdateIcons,0,wxALL |wxGROW |wxHORIZONTAL|wxALIGN_CENTER_HORIZONTAL,5);


	siz->Add(sizH,0,wxALL |wxGROW |wxHORIZONTAL,5);
}


void DlgTextureLib::updateTextures(int index) {

	int j,numTex = RESOURCEMANAGER->getNumTextures(); 
	int gridSize = m_ImagesGrid.size();

	if (numTex > gridSize) {
		for (j = gridSize; j < numTex; j++) {
			m_ImagesGrid.push_back(new ImageGridCellRenderer());
			m_GridTextures->AppendCols();
			m_GridTextures->SetColSize(j,104);
			m_GridTextures->SetCellRenderer(0, j, m_ImagesGrid[j]);
			m_GridTextures->SetReadOnly(0,j,true);
		}
	} else if (numTex < gridSize){
		m_GridTextures->DeleteCols(numTex,gridSize-numTex);
		for (j = gridSize; j > numTex; j--)
			m_ImagesGrid.pop_back();
	}

	nau::material::ITexture *texture;
		
	ITexImage *ti;
	wxBitmap *bm;
	for(j = 0 ; j < numTex ; j++) {
		texture = RESOURCEMANAGER->getTexture(j);
		ti = ITexImage::create(texture);
		unsigned char *data = ti->getRGBData();

		if (data == NULL)
			m_ImagesGrid[j]->setBitmap(m_EmptyBitmap);
		else {
			wxImage *ima = new wxImage(ti->getWidth(), ti->getHeight(), data, true);
			ima->Rescale(96, 96);
			bm = new wxBitmap(ima->Mirror(false));
			m_Bitmaps[texture->getPropi(ITexture::ID)] = bm;
			m_ImagesGrid[j]->setBitmap(bm);
			free (data);
			delete ima;
		}
		// have to free memory
		delete ti;
	}

	//for(j = 0 ; j < numTex ; j++) {
	//	texture = RESOURCEMANAGER->getTexture(j);
	//	wxBitmap *bm = texture->getBitmap();
	//	if (bm == NULL)
	//		m_ImagesGrid[j]->setBitmap(new wxBitmap(96,96));
	//	else
	//		m_ImagesGrid[j]->setBitmap(bm);
	//	//gridTextures->SetCellRenderer(0, j, imagesGrid[j]);
	//}

//	for (j = numTex; j < 4; j++) {
//		imagesGrid[j]->setBitmap(new wxBitmap(96,96));
	//	gridTextures->SetCellRenderer(0, j, imagesGrid[j]);
//	}

	setTextureProps(index);

}



void DlgTextureLib::setTextureProps(int index) {

	nau::material::ITexture *texture = RESOURCEMANAGER->getTexture(index);
	m_ActiveTexture = index;

	m_GridTextures->SelectBlock(0,index,0,index,false);

	m_PGTextureProps->ClearSelection();

	if (texture == NULL) {
		m_PGTextureProps->SetPropertyValue(wxT("Name"),"");
		//m_PGTextureProps->SetPropertyValue(wxT("Dimensions(WxHxD)"),"");
		return;
	}

	PropertyManager::updateGrid(m_PGTextureProps, ITexture::Attribs, (AttributeValues *)texture);
	
	wxString s = wxString(texture->getLabel().c_str());
	m_PGTextureProps->SetPropertyValue(wxT("Name"),s);
	wxBitmap *bm = m_Bitmaps[ texture->getPropi(ITexture::ID)];
	//wxBitmap *bm = texture->getBitmap();
	//unsigned char *bm = texture->getBitmap();
	m_PGTextureProps->SetPropertyImage(wxT("Name"), *bm);
}



void DlgTextureLib::OnProcessTexturePropsChange( wxPropertyGridEvent& e) {

}


void DlgTextureLib::OnUpdateIcons(wxCommandEvent& event) {

	for (auto b : m_Bitmaps) {
		delete b.second;
	}
	m_Bitmaps.clear();

	updateTextures(0);

	m_GridTextures->Refresh(true);
	Parent->Refresh();
	notifyUpdate(TEXTURE_ICON_UPDATE,"","");
}



void DlgTextureLib::OnSaveRaw(wxCommandEvent& event) {

	nau::material::ITexture *texture = RESOURCEMANAGER->getTexture(m_ActiveTexture);
	char name[256];
	sprintf(name, "%s.raw", texture->getLabel().c_str());
	std::string sname = nau::system::File::Validate(name);
	ITextureLoader::SaveRaw(texture, sname);
}


void DlgTextureLib::OnSavePNG( wxCommandEvent& event) {

	nau::material::ITexture *texture = RESOURCEMANAGER->getTexture(m_ActiveTexture);
	std::string s = texture->getLabel() + ".png";
	std::string sname = nau::system::File::Validate(s);
	ITextureLoader::Save(texture, ITextureLoader::PNG);
}


void DlgTextureLib::OnSaveHDR( wxCommandEvent& event) {

	nau::material::ITexture *texture = RESOURCEMANAGER->getTexture(m_ActiveTexture);
	std::string s = texture->getLabel() + ".hdr";
	std::string sname = nau::system::File::Validate(s);
	ITextureLoader::Save(texture, ITextureLoader::HDR);
}


void DlgTextureLib::updateTexInfo(int pos) {

	if (pos == m_ActiveTexture) {
		setTextureProps(pos);
	}
}




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
		m_GridTextures->Refresh();
		notifyUpdate(NEW_TEXTURE,std::string(dialog.GetPath().mb_str()),"");
//		DlgMaterials::Instance()->updateTextureList();
	}

}
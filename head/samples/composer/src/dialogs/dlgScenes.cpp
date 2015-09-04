#include "dialogs/dlgScenes.h"

#include "main.h"
#include "dialogs/propertyManager.h"

#include <nau/event/eventFactory.h>
#include <nau/geometry/iBoundingVolume.h>
#include <nau/math/utils.h>
#include <nau/scene/iScenePartitioned.h>

#ifdef NAU_PLATFORM_WIN32
#include <dirent.h>
#else
#include <dirent.h>
#include <sys/types.h>
#endif

#include <wx/dir.h>

using namespace nau::geometry;

BEGIN_EVENT_TABLE(DlgScenes, wxDialog)

	EVT_COMBOBOX(DLG_COMBO, DlgScenes::OnListSelect)
	EVT_PG_CHANGED(DLG_PROPS, DlgScenes::OnPropsChange)

	EVT_MENU(NEW_SCENE,DlgScenes::OnNewScene)
	EVT_MENU(SAVE_SCENE, DlgScenes::OnSaveScene)
	EVT_MENU(ADD_FILE,DlgScenes::OnAddFile)
	EVT_MENU(ADD_DIR,DlgScenes::OnAddDir)
	EVT_MENU(COMPILE, DlgScenes::OnCompile)
	EVT_MENU(BUILD, DlgScenes::OnBuild)
	
END_EVENT_TABLE()

wxWindow *DlgScenes::Parent = NULL;
DlgScenes *DlgScenes::Inst = NULL;


void 
DlgScenes::SetParent(wxWindow *p) 
{
	Parent = p;
}

DlgScenes* 
DlgScenes::Instance () 
{
	if (Inst == NULL)
		Inst = new DlgScenes();

	return Inst;
}



DlgScenes::DlgScenes()
	: wxDialog(DlgScenes::Parent, -1, wxT("Nau - Scenes"),wxDefaultPosition,wxDefaultSize,wxRESIZE_BORDER|wxDEFAULT_DIALOG_STYLE)                
{
	m_Parent = DlgScenes::Parent;
	wxBoxSizer *sizer = new wxBoxSizer(wxVERTICAL);

	/* ----------------------------------------------------------------
	                             Toolbar
	-----------------------------------------------------------------*/

    // load the image wanted on the toolbar
    wxBitmap newImage(wxT("bitmaps/new.bmp"), wxBITMAP_TYPE_BMP);
    wxBitmap cutImage(wxT("bitmaps/cut.bmp"), wxBITMAP_TYPE_BMP);

//	wxBitmap aboutImage = wxBITMAP(help) ;

    // create the toolbar and add our 1 tool to it
    m_toolbar = new wxToolBar(this,TOOLBAR_ID);
	long tstyle = m_toolbar->GetWindowStyle();
	tstyle |= wxTB_TEXT;
	m_toolbar->SetWindowStyle(tstyle);

    m_toolbar->AddTool(NEW_SCENE, _("New"), newImage, _("New Scene"));
    m_toolbar->AddTool(SAVE_SCENE, _("Save"), newImage, _("Save Scene as CBO"));

 	m_toolbar->AddSeparator();
    m_toolbar->AddTool(ADD_FILE, _("Model"), cutImage, _("Add Model"));
    m_toolbar->AddTool(ADD_DIR, _("Folder"), cutImage, _("Add Folder with Models"));

 	m_toolbar->AddSeparator();
    m_toolbar->AddTool(BUILD, _("Build"), cutImage, _("Build Octree"));
    m_toolbar->AddTool(COMPILE, _("Compile"), newImage, _("Compile Scene"));

	m_toolbar->Realize();

	sizer->Add(m_toolbar,0,wxGROW |wxALL, 1);

	/* ----------------------------------------------------------------
	                             End Toolbar
	-----------------------------------------------------------------*/

	setupPanel(sizer,Parent);

	SetAutoLayout(TRUE);
    sizer->SetSizeHints(this);
    sizer->Fit(this);
    SetSizer(sizer);

	this->SetTitle(wxT("Nau - Scenes"));
	this->SetSize(350,400);
}


void DlgScenes::updateDlg() {
	
	setupGrid();
	updateList();
	m_List->SetSelection(0);
	update();
	m_parent->Refresh();	
}


void 
DlgScenes::setupPanel(wxSizer *siz, wxWindow *Parent) {


	/* TOP: COMBO with scene names */
	wxBoxSizer *sizH1 = new wxBoxSizer(wxHORIZONTAL);

	wxStaticText *stg1 =  new wxStaticText(this,-1,wxT("Scene: "));
	m_List = new wxComboBox(this,DLG_COMBO,wxT(""),wxDefaultPosition,wxDefaultSize,0,NULL,wxCB_READONLY );

	updateList();
	m_List->SetSelection(0);

	sizH1->Add(stg1, 2, wxGROW|wxALL|wxHORIZONTAL,5);
	sizH1->Add(m_List, 5, wxGROW|wxALL|wxHORIZONTAL,5);

	wxStaticBox *sb = new wxStaticBox(this,-1,wxT(" Properties "));
	wxSizer *sizer = new wxStaticBoxSizer(sb,wxHORIZONTAL);

		wxFlexGridSizer *gs = new wxFlexGridSizer(3,2,0,0);

		wxStaticText *st1 = new wxStaticText(this,-1,wxT("Type:"));
		m_TType = new wxStaticText(this,-1,wxT("Octree"));

		gs ->Add(st1,0,wxALL,3);
		gs ->Add(m_TType,0,wxALL,3);
		//gs ->Add(st3,0,wxALL,3);
		//gs ->Add(tBoundingBoxMin,0,wxALL,3);
		//gs ->Add(st5,0,wxALL,3);
		//gs ->Add(tBoundingBoxMax,0,wxALL,3);

	sizer->Add(gs,0,wxGROW|wxALL,5);

	m_PG = new wxPropertyGridManager(this, DLG_PROPS,
		wxDefaultPosition, wxDefaultSize,
		// These and other similar styles are automatically
		// passed to the embedded wxPropertyGrid.
		wxPG_BOLD_MODIFIED | 
		// Plus defaults.
		wxPGMAN_DEFAULT_STYLE
		);

	m_PG->AddPage(wxT("properties"));
	m_PG->Append(new wxBoolProperty(wxT("dummy"), wxPG_LABEL));

	wxStaticBox *sb3 = new wxStaticBox(this,-1,wxT(" Scene m_Objects "));
	wxSizer *sizer3 = new wxStaticBoxSizer(sb3,wxVERTICAL);

	m_Objects = new wxListBox(this,-1);//,wxDefaultPosition,wxDefaultSize,li);

	sizer3->Add(m_Objects,1,wxGROW|wxEXPAND|wxALL,5);

	siz->Add(sizH1, 0, wxGROW|wxALL | wxHORIZONTAL, 5);
	siz->Add(sizer,	0,wxALL |wxGROW |wxHORIZONTAL,5);
	siz->Add(m_PG, 1, wxEXPAND | wxALL, 5);
	siz->Add(sizer3,1,wxALL |wxGROW |wxHORIZONTAL,5);
}


void
DlgScenes::setupGrid() {

	m_PG->Clear();
	m_PG->AddPage(wxT("properties"));

	std::vector<std::string> s = { "TRANSFORM_ORDER", "TRANSLATE", "ROTATE", "SCALE" };
	PropertyManager::createOrderedGrid(m_PG, IScene::Attribs, s);
	m_PG->SetSplitterLeft(true, true);
}


void 
DlgScenes::updateList() {

	std::vector<std::string> *names = RENDERMANAGER->getSceneNames();
	int num = names->size();

	if (num == 0) {
		m_Active = "";
		return;
	}

	m_List->Clear();

	for(int i = 0; i < num; i++)  {
		wxString s;
		s << i;
		m_List->Append(wxString(names->at(i).c_str()));
	}
	m_Active = names->at(0);
	delete names;
}


void DlgScenes::update() 
{
	if (m_Active == "")
		return;

	IScene *scene = RENDERMANAGER->getScene(m_Active);

	/* Is scene a partitioned scene? */
	if (scene->getType() == "Octree" || scene->getType() == "OctreeByMatScene") {
	
		IScenePartitioned *part = (IScenePartitioned *)scene;

		if (part->isBuilt() || part->isCompiled())
			m_toolbar->EnableTool(BUILD, FALSE);
		else
			m_toolbar->EnableTool(BUILD, TRUE);
	}
	else
		m_toolbar->EnableTool(BUILD, FALSE);

	if (scene->isCompiled() || (scene->getType() == "ScenePoses") || (scene->getType() == "SceneSkeleton"))
		m_toolbar->EnableTool(COMPILE, FALSE);
	else
		m_toolbar->EnableTool(COMPILE, TRUE);

	m_TType->SetLabel(wxString(scene->getType().c_str()));
	IBoundingVolume &bv = scene->getBoundingVolume();

	m_PG->ClearModifiedStatus();

	PropertyManager::updateGrid(m_PG, IScene::Attribs, (AttributeValues *)scene);
	//wxString s;
	//s.Printf("(%f,%f,%f)",bv.getMax().x, bv.getMax().y, bv.getMax().z);
	//tBoundingBoxMax->SetLabel(s);
	//s.Printf("(%f,%f,%f)",bv.getMin().x, bv.getMin().y, bv.getMin().z);
	//tBoundingBoxMin->SetLabel(s);

	m_Objects->Clear();
	std::vector<SceneObject*> objList = scene->getAllObjects();
	std::vector<SceneObject*>::iterator iter = objList.begin();

	for ( ; iter != objList.end(); ++iter )
		m_Objects->AppendString(wxString((*iter)->getName().c_str()));

}


void DlgScenes::OnPropsChange( wxPropertyGridEvent& e) {

	IScene *scene = RENDERMANAGER->getScene(m_Active);
	const wxString& name = e.GetPropertyName();
	unsigned int dotLocation = name.find_first_of(wxT("."), 0);
	std::string topProp = std::string(name.substr(0, dotLocation).mb_str());
	std::string prop = std::string(name.substr(dotLocation + 1, name.size() - dotLocation - 1).mb_str());

	PropertyManager::updateProp(m_PG, name.ToStdString(), IScene::Attribs, (AttributeValues *)scene);

}

void 
DlgScenes::OnSaveScene( wxCommandEvent& event)
{
	wxFileDialog *saveOctDlg = 
		new wxFileDialog (this, _("Save As CBO"), _(""), _(""), _(""), wxFD_SAVE, wxDefaultPosition);

	if (wxID_OK == saveOctDlg->ShowModal ()) {
		nau::NAU->writeAssets("CBO", (const char *)(saveOctDlg->GetPath()).c_str(), m_Active);
	}		

}


void 
DlgScenes::OnCompile( wxCommandEvent& event)
{
	IScene *scene = RENDERMANAGER->getScene(m_Active);
	((FrmMainFrame *)Parent)->compile(scene);
	//scene->compile();

	m_toolbar->EnableTool(COMPILE, FALSE);
	m_toolbar->EnableTool(BUILD, FALSE);
}

	
void 
DlgScenes::OnBuild( wxCommandEvent& event)
{
	IScenePartitioned *part = (IScenePartitioned *) RENDERMANAGER->getScene(m_Active);
	part->build();

	m_toolbar->EnableTool(BUILD, FALSE);
	update();
}


void DlgScenes::updateInfo(std::string name) 
{
	if (name == m_Active) 
		update();
}


void DlgScenes::OnNewScene(wxCommandEvent& event) 
{
	int result;
	bool nameUnique,exit = false;
	std::string name;

	do {
		wxTextEntryDialog dialog(this,
								 _T("(the name must be unique)\n")
								 _T("Please Input a Scene Name"),
								 _T("Scene Name\n"),
								 _T(""),
								 wxOK | wxCANCEL);

		result = dialog.ShowModal();
		name = std::string(dialog.GetValue().mb_str());
		nameUnique =  !RENDERMANAGER->hasScene(name); 

		if (!nameUnique && (result == wxID_OK)){
			wxMessageBox(_T("Scene name must be unique") , _T("Scene Name Error"), wxOK | wxICON_INFORMATION, this);
		}
		if (name == "" && (result == wxID_OK)){
			wxMessageBox(_T("Scene name must not be void"), _T("Scene Name Error"), wxOK | wxICON_INFORMATION, this);
		}

		if (result == wxID_CANCEL) {
			exit = true;
		}
		else if (nameUnique && name != "") {
			exit = true;
		}

	} while (!exit);

	if (result == wxID_OK) {
		RENDERMANAGER->createScene(name);
		updateList();
		m_List->Select(m_List->FindString(wxString(name.c_str())));
		m_Active = name;
		update();
		EVENTMANAGER->notifyEvent("NEW_SCENE", m_Active,"", NULL);
	}
}



void DlgScenes::OnAddFile(wxCommandEvent& event) 
{
	static const wxChar *fileTypes = _T( "3D Files (*.cbo, *.3ds, *.dae, *.obj, *.xml)|*.cbo;*.3ds;*.dae;*.obj; *.xml|CBO files (*.cbo)|*.cbo|COLLADA files (*.dae)|*.dae|3DS files (*.3ds)|*.3ds|OBJ files (*.obj)|*.obj|Ogre XML Meshes (*.xml)|*.xml");
	wxFileDialog *openFileDlg = new wxFileDialog (this, _("Open File"), _(""), _(""), fileTypes, wxFD_OPEN, wxDefaultPosition);

	if (wxID_OK == openFileDlg->ShowModal ()) {
		wxStopWatch aTimer;
		aTimer.Start();
		wxString path = openFileDlg->GetPath ();

		try {
			NAU->loadAsset (std::string(path.mb_str()), m_Active);
			EVENTMANAGER->notifyEvent("SCENE_CHANGED", m_Active,"", NULL);

			char buffer [1024];
			sprintf (buffer, "Elapsed time: %f", aTimer.Time()/1000.0);
			wxMessageBox (wxString(buffer));
		}
		catch (std::string &s) {
			wxMessageBox(wxString(s.c_str()));
		}
	}
	update();
	//IScene *scene = RENDERMANAGER->getScene(m_Active);
	//IBoundingVolume &bv = scene->getBoundingVolume();
	//wxString s;
	//s.Printf("(%f,%f,%f)",bv.getMax().x, bv.getMax().y, bv.getMax().z);
	//tBoundingBoxMax->SetLabel(s);
	//s.Printf("(%f,%f,%f)",bv.getMin().x, bv.getMin().y, bv.getMin().z);
	//tBoundingBoxMin->SetLabel(s);

}


void DlgScenes::OnAddDir(wxCommandEvent& event) 
{
	DIR *dir;
	struct dirent *ent;
	bool result = true;
	char fileName [1024];

	wxDirDialog *openDirDlg = new wxDirDialog (this);

	wxStopWatch aTimer;

	if (wxID_OK == openDirDlg->ShowModal()) {
		aTimer.Start();
		wxDir directory (openDirDlg->GetPath());

		try {
			dir = opendir ((const char *)directory.GetName().c_str());

			if (0 == dir) {
				wxMessageBox(wxT("Can't open folder"));
			}
			while (0 != (ent = readdir (dir))) {

		#ifdef NAU_PLATFORM_WIN32
				sprintf (fileName, "%s\\%s", directory.GetName().c_str(), ent->d_name);
		#else
				sprintf (fileName, "%s/%s", dirName, ent->d_name);						
		#endif
				try {
					NAU->loadAsset (fileName, m_Active);
				}
				catch(std::string &s) {
					closedir(dir);
					wxMessageBox(wxString(s.c_str()));
				}
			}
			EVENTMANAGER->notifyEvent("SCENE_CHANGED", m_Active,"", NULL);
			closedir (dir);

#ifndef FINAL
			char buffer [1024];
			sprintf (buffer, "Elapsed time: %f", aTimer.Time()/1000.0);
				
			wxMessageBox (wxString(buffer));
#endif		
		}
		catch(std::string &s) {
			wxMessageBox(wxString(s.c_str()));
		}
	}
	update();
}



void DlgScenes::OnListSelect(wxCommandEvent& event)
{
	int sel;

	sel = event.GetSelection();
	m_Active = std::string(m_List->GetString(sel).mb_str());
	update();
}

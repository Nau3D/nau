#include "dlgPass.h"

#include "dialogs/propertyManager.h"

#include <nau.h>
#include <nau/event/eventFactory.h>
#include <nau/render/passFactory.h>
#include <nau/event/eventString.h>


BEGIN_EVENT_TABLE(DlgPass, wxDialog)

	EVT_CLOSE(DlgPass::OnClose)

	EVT_COMBOBOX(DLG_MI_COMBO_PASS, DlgPass::OnSelectPass)
	EVT_COMBOBOX(DLG_MI_COMBO_PIPELINE, DlgPass::OnSelectPipeline)

	EVT_PG_CHANGED(DLG_MI_PG, DlgPass::OnProcessPGChange)
	EVT_BUTTON(DLG_BUTTON_ACTIVATE, DlgPass::OnActivate)

	EVT_MENU(PIPELINE_NEW, DlgPass::toolbarPipelineNew)
    EVT_MENU(PIPELINE_REMOVE, DlgPass::toolbarPipelineRemove)
    EVT_MENU(PASS_NEW, DlgPass::toolbarPassNew)
    EVT_MENU(PASS_REMOVE, DlgPass::toolbarPassRemove)

END_EVENT_TABLE()



// ----------------------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------------------


wxWindow *DlgPass::Parent = NULL;
DlgPass *DlgPass::Inst = NULL;


void 
DlgPass::SetParent(wxWindow *p) {

	Parent = p;
}


DlgPass* 
DlgPass::Instance () {

	if (Inst == NULL)
		Inst = new DlgPass();

	return Inst;
}


DlgPass::DlgPass()
                : wxDialog(DlgPass::Parent, -1, wxT("Nau - Pass"),wxDefaultPosition,wxDefaultSize,wxRESIZE_BORDER|wxDEFAULT_DIALOG_STYLE),
				m_Name("Pass Dialog")
{

	m_Parent = DlgPass::Parent;
	wxBoxSizer *sizer = new wxBoxSizer(wxVERTICAL);

	/* ----------------------------------------------------------------
	                             Toolbar
	-----------------------------------------------------------------*/

    // load the image wanted on the toolbar
    //wxBitmap newImage(wxT("bitmaps/new.bmp"), wxBITMAP_TYPE_BMP);
    //wxBitmap cutImage(wxT("bitmaps/cut.bmp"), wxBITMAP_TYPE_BMP);

//	wxBitmap aboutImage = wxBITMAP(help) ;

    // create the toolbar and add our 1 tool to it
 /*   m_toolbar = new wxToolBar(this,TOOLBAR_ID);
	long tstyle = m_toolbar->GetWindowStyle();
	tstyle |= wxTB_TEXT;
	m_toolbar->SetWindowStyle(tstyle);
    m_toolbar->AddTool(PIPELINE_NEW, _("New"), newImage, _("New Pipeline"));
    m_toolbar->AddTool(PIPELINE_REMOVE, _("Remove"), cutImage, _("Remove Pipeline"));

 	m_toolbar->AddSeparator();
    m_toolbar->AddTool(PASS_NEW, _("New"), newImage, _("New Pass"));
    m_toolbar->AddTool(PASS_REMOVE, _("Remove"), cutImage, _("Remove Pass"));

	m_toolbar->EnableTool(PIPELINE_REMOVE, FALSE);

	m_toolbar->Realize();


	sizer->Add(m_toolbar,0,wxGROW |wxALL, 1);
*/
	/* ----------------------------------------------------------------
	                             Pipelines and Passes
	-----------------------------------------------------------------*/

	m_PipelineList = new wxComboBox(this,DLG_MI_COMBO_PIPELINE,wxT(""),wxDefaultPosition,wxDefaultSize,0,NULL,wxCB_READONLY );
	m_PassList = new wxComboBox(this,DLG_MI_COMBO_PASS,wxT(""),wxDefaultPosition,wxDefaultSize,0,NULL,wxCB_READONLY );

	wxBoxSizer *sizer1 = new wxBoxSizer(wxHORIZONTAL);
	wxStaticText *stg1 =  new wxStaticText(this,-1,wxT("Pipeline: "));

	sizer1->Add(stg1,0,wxGROW |wxALL, 5);
	sizer1->Add(m_PipelineList,0,wxGROW |wxALL,5);
	wxStaticText *stg2 =  new wxStaticText(this,-1,wxT("Pass: "));
	sizer1->Add(stg2,0,wxGROW |wxALL, 5);
	sizer1->Add(m_PassList,0,wxGROW |wxALL,5);
	sizer1->Add(5,5,0, wxGROW |wxALL,5);

	sizer->Add(sizer1,0,wxGROW | wxALL,5);

	/* ----------------------------------------------------------------
	                             Properties
	-----------------------------------------------------------------*/

	wxStaticBox *sb;
	sb = new wxStaticBox(this,-1,wxT("Pass Properties"));
	wxSizer *sizerS = new wxStaticBoxSizer(sb,wxVERTICAL);

		m_PG = new wxPropertyGridManager(this, DLG_MI_PG,
				wxDefaultPosition, wxDefaultSize,
				// These and other similar styles are automatically
				// passed to the embedded wxPropertyGrid.
				wxPG_BOLD_MODIFIED|wxPG_SPLITTER_AUTO_CENTER|
				// Plus defaults.
				wxPGMAN_DEFAULT_STYLE
           );

		// dummy prop
		m_PG->AddPage(wxT("Properties"));
		m_PG->Append(new wxPGProperty(wxT("Parameters"), wxPG_LABEL));
		m_PG->Append(new wxFloatProperty(wxT("alpha"),wxPG_LABEL, 0.0f));

	sizerS->Add(m_PG, 1, wxALIGN_CENTER_HORIZONTAL | wxGROW | wxALL, 5);


	/* ----------------------------------------------------------------
								Activate button
	-----------------------------------------------------------------*/

	wxBoxSizer *sizH3 = new wxBoxSizer(wxHORIZONTAL);
	m_BActivate = new wxButton(this, DLG_BUTTON_ACTIVATE, wxT("Activate Pipeline"));
	m_ActivePipText = new wxStaticText(this, wxID_ANY, wxT("..."));
	sizH3->Add(m_BActivate, 0, wxALL | wxGROW | wxHORIZONTAL | wxALIGN_CENTER_HORIZONTAL, 5);
	sizH3->Add(m_ActivePipText, 0, wxALL | wxGROW | wxHORIZONTAL | wxALIGN_CENTER_HORIZONTAL, 5);


	/* ----------------------------------------------------------------
	                             End
	-----------------------------------------------------------------*/


	sizer->Add(sizerS,1,wxGROW | wxALL,5);
	sizer->Add(sizH3, 0, wxALL | wxGROW | wxHORIZONTAL, 15);
	//sizer->Add(5,5,0, wxGROW | wxALL,5);

    SetAutoLayout(TRUE);
    SetSizer(sizer);

    sizer->SetSizeHints(this);
    sizer->Fit(this);

}



void 
DlgPass::OnClose(wxCloseEvent& event) {
	
	this->Hide();
}


void 
DlgPass::updateDlg() {

	EVENTMANAGER->addListener("NEW_LIGHT",this);
	EVENTMANAGER->addListener("NEW_MATERIAL",this);
	EVENTMANAGER->addListener("NEW_CAMERA",this);
	EVENTMANAGER->addListener("NEW_RENDER_TARGET",this);
	EVENTMANAGER->addListener("NEW_SCENE",this);
	EVENTMANAGER->addListener("NEW_VIEWPORT", this);

	updatePipelines();
}


Pass *
DlgPass::getPass() {

	std::string pip, pass;
	Pass *p;

	pip = std::string(m_PipelineList->GetValue().mb_str());
	pass = std::string(m_PassList->GetValue().mb_str());

	p = RENDERMANAGER->getPass(pip,pass);

	return p;
}


// Fill Lists and Property Grid //

void 
DlgPass::updatePipelines() {

	std::vector<std::string> *pips = RENDERMANAGER->getPipelineNames();
	std::vector<std::string>::iterator iter;

	//wxString sel = m_PipelineList->GetStringSelection();
	wxString sel = wxString(RENDERMANAGER->getActivePipelineName());
	m_PipelineList->Clear();
	for (iter = pips->begin(); iter != pips->end(); ++iter)
		m_PipelineList->Append(wxString(iter->c_str()));

	delete pips;

	if (! m_PipelineList->SetStringSelection(sel)) 
		m_PipelineList->SetSelection(0);

	sel = m_PipelineList->GetStringSelection();
	m_ActivePipText->SetLabelText(sel);
	std::string pipName = std::string(sel.mb_str());

	Pipeline *pip = RENDERMANAGER->getPipeline(pipName);

	std::vector<std::string> passes;
	pip->getPassNames(&passes);

	sel = m_PassList->GetStringSelection();
	m_PassList->Clear();

	for (auto &name:passes)
		m_PassList->Append(wxString(name.c_str()));

	if (! m_PassList->SetStringSelection(sel))
		m_PassList->SetSelection(0);
		
	Pass *p = getPass();


	m_PG->ClearPage(0);

		m_PG->Append(new wxStringProperty(wxT("Class"), wxPG_LABEL, wxT("")));
		m_PG->DisableProperty(wxT("Class"));

		m_pgCamList.Add(wxT("dummy"));
		m_pgPropCam = new wxEnumProperty(wxT("Camera"),wxPG_LABEL,m_pgCamList);
		m_PG->Append(m_pgPropCam);
		
		m_pgViewportList.Add(wxT("From Camera"));
		m_pgPropViewport = new wxEnumProperty(wxT("Viewport"),wxPG_LABEL,m_pgViewportList);
		m_PG->Append(m_pgPropViewport);

		m_PG->Append(new wxBoolProperty(wxT("Use Render Target"), wxPG_LABEL, true));
		m_PG->SetPropertyAttribute( wxT("Use Render Target"),
                              wxPG_BOOL_USE_CHECKBOX,
                              true );

		m_pgRenderTargetList.Add(wxT("None"));
		m_pgPropRenderTarget = new wxEnumProperty(wxT("Render Target"),wxPG_LABEL,m_pgRenderTargetList);
		m_PG->Append(m_pgPropRenderTarget);

		//m_PG->Append(new wxBoolProperty(wxT("Clear Color"), wxPG_LABEL, true));
		//m_PG->SetPropertyAttribute( wxT("Clear Color"),
  //                            wxPG_BOOL_USE_CHECKBOX,
  //                            true );
		//m_PG->Append(new wxBoolProperty(wxT("Clear Depth"), wxPG_LABEL, true));
		//m_PG->SetPropertyAttribute( wxT("Clear Depth"),
  //                            wxPG_BOOL_USE_CHECKBOX,
  //                            true );

	updateLists(p);
	setupGrid();
	updateProperties(p) ;
}


void
DlgPass::setupGrid() {

	std::vector<std::string> order;

	PropertyManager::createOrderedGrid(m_PG, Pass::Attribs, order);
	m_PG->SetSplitterLeft(true, true);
}

/* ----------------------------------------------------------------

				TOOLBAR STUFF

-----------------------------------------------------------------*/

void 
DlgPass::toolbarPipelineNew(wxCommandEvent& WXUNUSED(event) ) {


	//std::string path,libFile,fullName,relativeName,libName;

	//wxFileDialog dialog
 //                (
 //                   this,
 //                   _T("New Material Lib"),
 //                   _T(""),
 //                   _T(""),
 //                   _T("SP Library (*.spl)|*.spl"),
	//				wxSAVE | wxOVERWRITE_PROMPT
 //                );

	//dialog.SetDirectory(CProject::Instance()->m_path.c_str());

	//if (dialog.ShowModal() != wxID_OK)
	//	return;

	//path = (char *)dialog.GetDirectory().c_str();
	//libFile = (char *)dialog.GetFilename().c_str();
	//fullName = (char *)dialog.GetPath().c_str();
	//relativeName = CFilename::GetRelativeFileName(CProject::Instance()->m_path,fullName);
	//libName = CFilename::RemoveExt(relativeName);
	//
	//CMaterialLib *ml;
	//CMaterial *mat;

	//mat = new CMaterial();
	//mat->setName("Default");

	//ml = new CMaterialLib();
	//ml->m_filename = libName;
	//ml->add(mat);

	//m_libManager->addLib(ml);

	//updateDlg();
	//m_toolbar->EnableTool(LIBMAT_SAVEALL, TRUE);
	//DlgModelInfo::Instance()->updateDlg();

}

void DlgPass::toolbarPipelineRemove(wxCommandEvent& WXUNUSED(event) ) {


	//std::string path,libFile,fullName,relativeName,libName;

	//wxFileDialog dialog
 //                (
 //                   this,
 //                   _T("Open Material Lib"),
 //                   _T(""),
 //                   _T(""),
 //                   _T("SP Library (*.spl)|*.spl"),
	//				wxOPEN | wxFILE_MUST_EXIST 
 //                );

	//dialog.SetDirectory(CProject::Instance()->m_path.c_str());

	//if (dialog.ShowModal() != wxID_OK)
	//	return;

	//path = (char *)dialog.GetDirectory().c_str();
	//libFile = (char *)dialog.GetFilename().c_str();
	//fullName = (char *)dialog.GetPath().c_str();
	//relativeName = CFilename::GetRelativeFileName(CProject::Instance()->m_path,fullName);

	//
	//m_libManager->load(relativeName, path);

	//updateDlg();
	//m_toolbar->EnableTool(LIBMAT_SAVEALL, TRUE);
	//DlgModelInfo::Instance()->updateDlg();

}


void DlgPass::toolbarPassNew(wxCommandEvent& WXUNUSED(event) ) {

	//std::string lib;

	//m_libManager->save(CProject::Instance()->m_path);

}


void DlgPass::toolbarPassRemove(wxCommandEvent& WXUNUSED(event) ) {

	//wxString name;
	//int exit = 0;
	//CMaterial *mat;
	//CMaterialLib *ml;
	//std::string lib;
	//int dialogRes;

	//lib = libList->GetValue().c_str();
	//ml = m_libManager->getLib(lib);

	//do {
	//	wxTextEntryDialog dialog(this,
	//							 _T("Enter the new material's name\n"),
	//							 _T("Material's Name"),
	//							 _T(name),
	//							 wxOK | wxCANCEL);
	//	dialogRes = dialog.ShowModal();
	//	if (dialogRes == wxID_OK)
	//	{
	//		name = dialog.GetValue();
	//		int status = m_libManager->validMatName(lib,name.c_str());
	//		if (status == CMaterialLibManager::OK) {

	//			mat = new CMaterial();
	//			mat->setName(name.c_str());
	//			ml->add(mat);
	//			exit = 1;
	//			updateDlg();
	//			DlgModelInfo::Instance()->updateDlg();
	//		}
	//		else if (status == CMaterialLibManager::INVALID_NAME) 
	//			wxMessageBox(dialog.GetValue(), _T("Invalid Name (can't begin with a space)"), wxOK | wxICON_INFORMATION, this);
	//		else if (status == CMaterialLibManager::NAME_EXISTS) 
	//			wxMessageBox(dialog.GetValue(), _T("Name Already Exists"), wxOK | wxICON_INFORMATION, this);
	//	}
	//}
	//while (dialogRes != wxID_CANCEL && !exit);
}



/* ----------------------------------------------------------------

				EVENTS FROM OTHER DIALOGS AND NAU

-----------------------------------------------------------------*/


void 
DlgPass::eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt) {

	Pass *p = getPass();

	updatePipelines();

/*	if (eventType == "NEW_RENDERTARGET") {
		updateRenderTargetList(p);
	}
	else if (eventType == "NEW_VIEWPORT") {
		updateViewportList(p);
	}
	else if (eventType == "NEW_CAMERA") {
		updateCameraList(p);
		updateScenes(p);
	}
	else if (eventType == "NEW_LIGHT") {
		updateLights(p);
	}
	else if (eventType == "NEW_MATERIAL") {
		updateMaterialList();
	}
	else if (eventType == "NEW_SCENE") {
		wxPGProperty *pid2;
		pid2 = m_PG->AppendIn(pidScenes,new wxBoolProperty( wxString(sender.c_str()), wxPG_LABEL, false ) );
		pid2->SetAttribute( wxPG_BOOL_USE_CHECKBOX, true );
	}
*/
}




/* ----------------------------------------------------------------------

	General Stuff

-----------------------------------------------------------------------*/





void DlgPass::updateProperties(Pass *p) {

	//
	m_PG->SetPropertyValue(wxT("Class"), wxString(p->getClassName().c_str()));

	// CAMERA
	if (p->getClassName() == "quad")
		m_PG->DisableProperty(wxT("Camera"));
	else
		m_PG->EnableProperty(wxT("Camera"));

	m_PG->SetPropertyValue(wxT("Camera"), wxString(p->getCameraName().c_str()));

	// VIEWPORT
	std::shared_ptr<Viewport> v = p->getViewport();

	if (p->hasRenderTarget() && p->isRenderTargetEnabled()) {
		m_PG->SetPropertyValue(wxT("Viewport"), wxT("From Render Target"));
		m_PG->DisableProperty(wxT("Viewport"));
	}
	else if (!v) {
		m_PG->EnableProperty(wxT("Viewport"));
		m_PG->SetPropertyValue(wxT("Viewport"), wxT("From Camera"));
	}
	else {
		m_PG->EnableProperty(wxT("Viewport"));
		m_PG->SetPropertyValue(wxT("Viewport"), wxString(v->getName().c_str()));
	}

	// RENDER TARGET
	if (RESOURCEMANAGER->getNumRenderTargets()) {
		m_PG->EnableProperty(wxT("Render Target"));
		m_PG->EnableProperty(wxT("Use Render Target"));
	}
	else {
		m_PG->DisableProperty(wxT("Render Target"));
		m_PG->DisableProperty(wxT("Use Render Target"));
	}

	if (p->hasRenderTarget())
		m_PG->SetPropertyValue(wxT("Render Target"), wxString(p->getRenderTarget()->getName().c_str()));
	else
		m_PG->SetPropertyValue(wxT("Render Target"), wxT("None"));
		
	m_PG->SetPropertyValue(wxT("Use Render Target"),p->isRenderTargetEnabled());

	// SCENES
	std::vector<std::string> *names = RENDERMANAGER->getAllSceneNames();
	std::vector<std::string>::iterator iter;
	bool b;

	for (iter = names->begin(); iter != names->end(); ++iter) {

		if (p->hasScene(*iter))
			b = true;
		else
			b = false;
		wxString str(wxT("Scenes."));
		str.Append(wxString(*iter).c_str());
		m_PG->SetPropertyValue(str, b);
	}
	delete names;

	// LIGHTS
	std::vector<std::string> lNames;
	RENDERMANAGER->getLightNames(&lNames);

	for (auto & name:lNames) {

		if (p->hasLight(name))
			b = true;
		else
			b = false;
		wxString str(wxT("Lights."));
		wxString aux(name.c_str());
		str.Append(wxString(name.c_str()));
		m_PG->SetPropertyValue(str, b);
	}

	if (m_PG->GetPropertyByName(wxT("Material Maps")))
		m_PG->DeleteProperty(wxT("Material Maps"));

	std::map<std::string, nau::material::MaterialID> mm = p->getMaterialMap();

	wxPGProperty *pgMM;
	pgMM = m_PG->Append(new wxPGProperty(wxT("Material Maps"), wxPG_LABEL));

	std::map<std::string, nau::material::MaterialID>::iterator iterMM = mm.begin();
	MaterialID mid;
	wxString mName,propName;

	if (mm.size() != 0 && iterMM->first == "*") {
		mid = iterMM->second;
		mName = wxString(mid.getLibName().c_str());
		mName.append(wxT("::"));
		mName.append(wxString(mid.getMaterialName().c_str()));
		m_PG->AppendIn(pgMM, new wxEnumProperty(wxT("*"), wxPG_LABEL,m_pgMaterialListPlus));
		propName = wxT("Material Maps.*");
		m_PG->SetPropertyValue(propName, mName);

		++iterMM;
	}
	else {
		m_PG->AppendIn(pgMM, new wxEnumProperty(wxT("*"), wxPG_LABEL,m_pgMaterialListPlus));
		propName = wxT("Material Maps.*");
		m_PG->SetPropertyValue(propName, wxT("None"));	
	}

	for ( ; iterMM != mm.end() ; ++iterMM) {

		mid = iterMM->second;
		mName = wxString(mid.getLibName().c_str());
		mName.append(wxT("::"));
		mName.append(wxString(mid.getMaterialName().c_str()));
		m_PG->AppendIn(pgMM, new wxEnumProperty(wxString(iterMM->first.c_str()), wxPG_LABEL,m_pgMaterialList));
		propName = wxT("Material Maps.");
		propName.append(wxString(iterMM->first.c_str()));
		m_PG->SetPropertyValue(propName, mName);
		//property name is the first value of the map, the prop value is the second value of the map
	}

	PropertyManager::updateGrid(m_PG, Pass::Attribs, (AttributeValues *)p);

	Refresh();

	//m_PG->RefreshGrid();//->Refresh();
}


void DlgPass::updateLists(Pass *p) 
{
	updateCameraList(p);
	updateViewportList(p);
	updateRenderTargetList(p);
	updateLights(p);
	updateScenes(p);
	updateMaterialList();
}



void DlgPass::updateCameraList(Pass *p) {

	m_PG->ClearSelection();
	m_pgCamList.RemoveAt(0,m_pgCamList.GetCount());

	int i = 0;
	std::vector<std::string> passes;
	RENDERMANAGER->getCameraNames(&passes);
	for (auto& name: passes) {
		m_pgCamList.Add(wxString(name.c_str()),i++);
	}
	m_pgPropCam->SetChoices(m_pgCamList);
}	
	
void DlgPass::updateViewportList(Pass *p) {

	int i = 0;

	m_PG->ClearSelection();
	m_pgViewportList.RemoveAt(0,m_pgViewportList.GetCount());
	m_pgViewportList.Add(wxT("From Camera"),++i);

	if (p->hasRenderTarget()) 
		m_pgViewportList.Add(wxT("From Render Target"),++i);
	
	std::vector<std::string> viewports;
	RENDERMANAGER->getViewportNames(&viewports);

	for (auto &v:viewports)
		m_pgViewportList.Add(wxString(v.c_str()), ++i);

	m_pgPropViewport->SetChoices(m_pgViewportList);
}


void DlgPass::updateRenderTargetList(Pass *p) {

	std::vector<std::string>::iterator iter;

	m_PG->ClearSelection();
	m_pgRenderTargetList.RemoveAt(0,m_pgRenderTargetList.GetCount());
	m_pgRenderTargetList.Add(wxT("None"),0);

	std::vector<std::string> *renderTargets = RESOURCEMANAGER->getRenderTargetNames();
	int i = 0;
	for (iter = renderTargets->begin(); iter != renderTargets->end(); ++iter)
		m_pgRenderTargetList.Add(wxString(iter->c_str()), ++i);

	delete renderTargets;

	m_pgPropRenderTarget->SetChoices(m_pgRenderTargetList);
}


void DlgPass::updateScenes(Pass *p)
{
	wxPGProperty *m_PidScenes;
	std::vector<std::string> *names = RENDERMANAGER->getAllSceneNames();
	std::vector<std::string>::iterator iter;
	
	m_PidScenes = m_PG->Append(new wxPGProperty(wxT("Scenes"), wxPG_LABEL));

	wxPGProperty *pid2;
	for (iter = names->begin(); iter != names->end(); ++iter) {
		pid2 = m_PG->AppendIn(m_PidScenes, new wxBoolProperty( wxString((*iter).c_str()), wxPG_LABEL, false ) );
		pid2->SetAttribute( wxPG_BOOL_USE_CHECKBOX, true );
	}
	delete names;
}


void DlgPass::updateLights(Pass *p)
{
	std::vector<std::string> names;
	RENDERMANAGER->getLightNames(&names);

	wxPGProperty *pid = m_PG->Append(new wxPGProperty(wxT("Lights"), wxPG_LABEL));

	wxPGProperty *pid2;
	for (auto &name:names) {
		pid2 = m_PG->AppendIn(pid, new wxBoolProperty( wxString(name.c_str()), wxPG_LABEL, false ) );
		pid2->SetAttribute( wxPG_BOOL_USE_CHECKBOX, true );
	}
}


void DlgPass::updateMaterialList() {

	std::string libmat;

	if (m_pgMaterialList.GetCount())
		m_pgMaterialList.RemoveAt(0,m_pgMaterialList.GetCount());

	if (m_pgMaterialListPlus.GetCount())
		m_pgMaterialListPlus.RemoveAt(0,m_pgMaterialListPlus.GetCount());

	m_pgMaterialListPlus.Add(wxT("None"));

	std::vector<std::string> libList; 
	MATERIALLIBMANAGER->getLibNames(&libList);

	wxString libname;
	for (auto& lib:libList) {

		libname = wxString(lib.c_str());
		libname.append(wxT("::*"));
		m_pgMaterialListPlus.Add(libname);
	}

	std::vector<std::string> matList;

	for (auto& lib : libList) {

		 MATERIALLIBMANAGER->getMaterialNames(lib, &matList);

		for(auto& mat: matList) {

			libmat = lib;
			libmat.append("::");
			libmat.append(mat);
			m_pgMaterialList.Add(wxString(libmat.c_str()));
			m_pgMaterialListPlus.Add(wxString(libmat.c_str()));
		}
	}
}


/* ----------------------------------------------------------------------

	Scenes Stuff

-----------------------------------------------------------------------*/

void DlgPass::OnActivate(wxCommandEvent& event) {

	wxString sel = m_PipelineList->GetValue().mb_str();
	m_ActivePipText->SetLabelText(sel);
	RENDERMANAGER->setActivePipeline(std::string(sel));
}



void DlgPass::OnSelectPass(wxCommandEvent& event) {

	Pass *p;
		
	p = getPass();

	updateMaterialList();
	updateProperties(p);
}


void DlgPass::OnSelectPipeline(wxCommandEvent& event) {

	wxString selName;
	selName = event.GetString();

	Pipeline *pip = RENDERMANAGER->getPipeline(std::string(selName.mb_str()));
	std::vector<std::string> passes;
	pip->getPassNames(&passes);

	m_PassList->Clear();
	
	for (auto &name:passes)
		m_PassList->Append(wxString(name.c_str()));

	m_PassList->SetSelection(0);
		
	Pass *p = getPass();
	updateMaterialList();
	updateProperties(p) ;
}



void 
DlgPass::OnProcessPGChange( wxPropertyGridEvent& e)
{
	Pass *p = getPass();

	wxString name,subname,value;
	std::string mat,lib;
	name = e.GetPropertyName();
	int index;
	index = e.GetPropertyValue().GetInteger();

	if (name == wxT("Camera")) {
		value = m_pgCamList.GetLabel(index);
		p->setCamera(std::string(value.mb_str()));	
	}
	else if (name == wxT("Viewport")) {
		if (index == 0)
			p->setViewport(RENDERMANAGER->getCamera(p->getCameraName())->getViewport());
		else {
			value = m_pgViewportList.GetLabel(index-1);
			p->setViewport(RENDERMANAGER->getViewport(std::string(value.mb_str())));
		}
	}
	else if (name == wxT("Use Render Target")) {
		p->enableRenderTarget(0 != e.GetPropertyValue().GetBool());
	}
	else if (name == wxT("Render Target")) {
		value = e.GetPropertyValue().GetString();
		if (value == wxT("None")) {
			p->setRenderTarget(NULL);
			m_PG->SetPropertyValue(wxT("Viewport"),wxT("From Camera"));
			m_PG->EnableProperty(wxT("Viewport"));
		}
		else {
			m_PG->SetPropertyValue(wxT("Viewport"), wxT("From Render Target"));
			m_PG->DisableProperty(wxT("Viewport"));
			p->setRenderTarget(RESOURCEMANAGER->getRenderTarget(std::string(value.mb_str())));
		}
	}
	//else if (name == wxT("Clear Color")) 
	//	p->setPropb(Pass::COLOR_CLEAR, (0 != e.GetPropertyValue().GetBool()));
	//else if (name == wxT("Clear Depth"))
	//	p->setPropb(Pass::DEPTH_CLEAR, (0 != e.GetPropertyValue().GetBool()));
	
	else if (name.substr(0,6) == wxT("Lights")) {

		subname = name.substr(7,std::string::npos);
		bool b = 0 != e.GetPropertyValue().GetBool();
		if (b)
			p->addLight(std::string(subname.mb_str()));
		else
			p->removeLight(std::string(subname.mb_str()));
	}
	else if (name.substr(0,6) == wxT("Scenes")) {

		subname = name.substr(7,std::string::npos);
		bool b = 0 != e.GetPropertyValue().GetBool();
		if (b)
			p->addScene(std::string(subname.mb_str()));
		else
			p->removeScene(std::string(subname.mb_str()));
	}
	else if (name.substr(0,13) == wxT("Material Maps")) { // Material Maps

		subname = name.substr(14, std::string::npos);
		if (subname == wxT("*"))
			value = m_pgMaterialListPlus.GetLabel(e.GetPropertyValue().GetInteger());
		else
			value = m_pgMaterialList.GetLabel(e.GetPropertyValue().GetInteger());
		mat = std::string(value.AfterLast(':').mb_str());
		lib = std::string(value.BeforeFirst(':').mb_str());

		if (subname == wxT("*")) {
			if (mat == "*") {
				p->remapAll(lib);
				updateMats(p);
			}
			else if (value != wxT("None")) {
				p->remapAll(lib,mat);
				updateMats(p);
			}
		}
		else
			p->remapMaterial(std::string(subname.c_str()),lib,mat);
	}
	else
		PropertyManager::updateProp(m_PG, name.ToStdString(), Pass::Attribs, (AttributeValues *)p);

	//else if (name.substr(0,10) == wxT("Parameters")) {

	//	subname = name.substr(11, std::string::npos);
	//	p->setParam(std::string(subname.mb_str()), (float)(e.GetPropertyValue().GetDouble()));
	//
	//} 
}

	
void 
DlgPass::updateMats(Pass *p)
{
	MaterialID mid;
	wxString mName,propName;

	std::map<std::string, nau::material::MaterialID> mm = p->getMaterialMap();

	std::map<std::string, nau::material::MaterialID>::iterator iterMM = mm.begin();

	for ( ; iterMM != mm.end() ; ++iterMM) {

		mid = iterMM->second;
		mName = wxString(mid.getLibName().c_str());
		mName.append(wxT("::"));
		mName.append(wxString(mid.getMaterialName().c_str()));
		propName = wxT("Material Maps.");
		propName.append(wxString(iterMM->first.c_str()));
		m_PG->SetPropertyValue(propName, mName);
	}
}



//
//



/* ----------------------------------------------------------------------

	Params Stuff

-----------------------------------------------------------------------*/

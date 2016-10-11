#include "dlgDbgPrograms.h"

//#include "../glInfo.h"

#include <nau.h>
#include <nau/util/tree.h>
//#include <nau/debug/profile.h>

BEGIN_EVENT_TABLE(DlgDbgPrograms, wxDialog)
	EVT_BUTTON(DLG_BTN_SAVELOG, OnSaveInfo)
	EVT_BUTTON(DLG_BTN_REFRESH, OnRefreshLog)
END_EVENT_TABLE()


wxWindow *DlgDbgPrograms::m_Parent = NULL;
DlgDbgPrograms *DlgDbgPrograms::m_Inst = NULL;


void 
DlgDbgPrograms::SetParent(wxWindow *p) {

	m_Parent = p;
}


DlgDbgPrograms* 
DlgDbgPrograms::Instance () {

	if (m_Inst == NULL)
		m_Inst = new DlgDbgPrograms();

	return m_Inst;
}
 

DlgDbgPrograms::DlgDbgPrograms(): wxDialog(DlgDbgPrograms::m_Parent, -1, wxT("Nau - Program Information"),wxDefaultPosition,
						   wxDefaultSize,wxRESIZE_BORDER|wxDEFAULT_DIALOG_STYLE)
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize);

	wxBoxSizer *bSizer1;
	bSizer1 = new wxBoxSizer( wxVERTICAL);

	wxStaticBoxSizer * sbSizer1;
	sbSizer1 = new wxStaticBoxSizer( new wxStaticBox( this, wxID_ANY, wxEmptyString ), wxVERTICAL );

	m_Log = new wxTreeCtrl(this,NULL,wxDefaultPosition,wxDefaultSize,wxLB_SINGLE | wxLB_HSCROLL | wxEXPAND);

	sbSizer1->Add(m_Log, 1, wxALL|wxEXPAND, 5);
	bSizer1->Add(sbSizer1, 1, wxEXPAND, 5);


	wxBoxSizer* bSizer2;
	bSizer2 = new wxBoxSizer( wxHORIZONTAL );

	m_bSave = new wxButton(this,DLG_BTN_SAVELOG,wxT("Save"));
	m_bRefresh = new wxButton(this, DLG_BTN_REFRESH, wxT("Refresh"));

	bSizer2->Add(m_bSave, 0, wxALL, 5);
	bSizer2->Add(m_bRefresh, 0, wxALL, 5);
	bSizer1->Add(bSizer2, 0, wxALIGN_CENTER|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxEXPAND|wxSHAPED, 5);

	this ->SetSizer(bSizer1);
	this->Layout();
	this->Centre(wxBOTH);

	m_IsLogClear = true;
}


void
DlgDbgPrograms::updateDlg() {

	EVENTMANAGER->addListener("SHADER_DEBUG_INFO_AVAILABLE", this);
}


void
DlgDbgPrograms::updateDlgTree() {

	clear();
	m_IsLogClear = false;
	std::string s = RENDERMANAGER->getActivePipelineName();
	wxTreeItemId root = m_Log->AddRoot("Pipeline: " + s + std::string(" >"));
	m_Log->Expand(root);
	nau::util::Tree *t = RENDERER->getShaderDebugTree();

	updateTree(t, root);
}


void 
DlgDbgPrograms::updateTree(nau::util::Tree *t, wxTreeItemId branch) {

	wxTreeItemId newBranch;

	size_t count = t->getElementCount();
	nau::util::Tree *t1;

	for (size_t i = 0; i < count; ++i) {

		t1 = t->getBranch(i);
		if (t1) {
			if (t->getValue(i) == "")
				newBranch = m_Log->AppendItem(branch, t->getKey(i) + std::string(" >"));
			else
				newBranch = m_Log->AppendItem(branch, t->getKey(i) + std::string(": ") + t->getValue(i) + std::string(" >"));

			m_Log->Expand(newBranch);
			updateTree(t1, newBranch);
		}
		else {
			m_Log->AppendItem(branch, t->getKey(i) + std::string(": ") + t->getValue(i));
		}
			
	}
	m_Log->Expand(branch);
}


std::string &
DlgDbgPrograms::getName () {

	m_Name = "DlgDbgPrograms";
	return(m_Name);
}


void
DlgDbgPrograms::eventReceived(const std::string &sender, const std::string &eventType, 
	const std::shared_ptr<nau::event_::IEventData> &evt) {

	if (eventType == "SHADER_DEBUG_INFO_AVAILABLE")
		updateDlgTree();
}


void DlgDbgPrograms::clear() {

	m_Log->DeleteAllItems();
	m_IsLogClear = true;
}


void DlgDbgPrograms::OnRefreshLog(wxCommandEvent& event) {

	RENDERER->setPropb(IRenderer::DEBUG_DRAW_CALL, true);
}


void DlgDbgPrograms::OnSaveInfo(wxCommandEvent& event) {

	wxFileDialog dialog(this,
		 wxT("Program info save file dialog"),
		wxEmptyString,
		wxT("programinfo.txt"),
		wxT("Text files (*.txt)|*.txt"),
		wxFD_SAVE|wxFD_OVERWRITE_PROMPT);

	wxTreeItemId m_Rootnode = m_Log->GetRootItem();

	if (wxID_OK == dialog.ShowModal ()) {
		
		wxString path = dialog.GetPath ();
	   
		fstream s;
		s.open(path.mb_str(), fstream::out);
		unsigned int lines = m_Log->GetCount();
		
		int nodelevel = 0;
		s <<  m_Log->GetItemText(m_Rootnode) << "\n"; 

		OnSaveInfoAux(s, m_Rootnode, nodelevel+1);

		s.close();
	}
}


void DlgDbgPrograms::OnSaveInfoAux(fstream &s, wxTreeItemId parent, int nodelevel) {
	wxTreeItemIdValue cookie;
	wxTreeItemId currentChild;

	currentChild = m_Log->GetFirstChild(parent, cookie);

	while (currentChild.IsOk()){
		for (int nl = 0; nl < nodelevel; nl++){
			s << "\t";
		}
		s <<  m_Log->GetItemText(currentChild) << "\n"; 
		OnSaveInfoAux(s, currentChild, nodelevel+1);
		currentChild = m_Log->GetNextChild(parent, cookie);
	}
}


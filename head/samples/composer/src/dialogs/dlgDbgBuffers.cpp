#include "dlgDbgBuffers.h"
#include "../glInfo.h"
#include <nau.h>
#include <nau/debug/profile.h>

BEGIN_EVENT_TABLE(DlgDbgBuffers, wxDialog)
	EVT_BUTTON(DLG_BTN_SAVELOG, OnSaveInfo)
END_EVENT_TABLE()


wxWindow *DlgDbgBuffers::m_Parent = NULL;
DlgDbgBuffers *DlgDbgBuffers::m_Inst = NULL;


void 
DlgDbgBuffers::SetParent(wxWindow *p) {

	m_Parent = p;
}


DlgDbgBuffers* 
DlgDbgBuffers::Instance () {

	if (m_Inst == NULL)
		m_Inst = new DlgDbgBuffers();

	return m_Inst;
}
 

DlgDbgBuffers::DlgDbgBuffers(): wxDialog(DlgDbgBuffers::m_Parent, -1, wxT("Nau - Buffers Information"),wxDefaultPosition,
						   wxDefaultSize,wxRESIZE_BORDER|wxDEFAULT_DIALOG_STYLE)
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize);

	wxBoxSizer *bSizer1;
	bSizer1 = new wxBoxSizer( wxVERTICAL);

	wxStaticBoxSizer * sbSizer1;
	sbSizer1 = new wxStaticBoxSizer( new wxStaticBox( this, wxID_ANY, wxEmptyString ), wxVERTICAL );

	m_log = new wxTreeCtrl(this,NULL,wxDefaultPosition,wxDefaultSize,wxLB_SINGLE | wxLB_HSCROLL | wxEXPAND);

	sbSizer1->Add(m_log, 1, wxALL|wxEXPAND, 5);

	bSizer1->Add(sbSizer1, 1, wxEXPAND, 5);


	wxBoxSizer* bSizer2;
	bSizer2 = new wxBoxSizer( wxHORIZONTAL );

	m_bSave = new wxButton(this,DLG_BTN_SAVELOG,wxT("Save"));

	bSizer2->Add(m_bSave, 0, wxALL, 5);

	bSizer1->Add(bSizer2, 0, wxALIGN_CENTER|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxEXPAND|wxSHAPED, 5);

	this ->SetSizer(bSizer1);
	this->Layout();
	this->Centre(wxBOTH);

	isLogClear=true;
}


void
DlgDbgBuffers::updateDlg() 
{

}


std::string &
DlgDbgBuffers::getName ()
{
	name = "DlgDbgBuffers";
	return(name);
}


void
DlgDbgBuffers::eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt)
{
}


void DlgDbgBuffers::append(std::string s) {

}


void DlgDbgBuffers::clear() {

	m_log->DeleteAllItems();
	isLogClear=true;
}

void DlgDbgBuffers::loadBufferInfo() {
	wxTreeItemId rootnode, buffernode;
    std::vector<string> bufferdata;
	int items = getCurrentBufferInfoData(bufferdata);
	
	if (isLogClear){
		rootnode = m_log->AddRoot("Buffers>");
		for (int i = 0; i < bufferdata.size(); i+=1+items){
			buffernode = m_log->AppendItem(rootnode,bufferdata[i]);
			for (int j = 0; j < items; j++){
				m_log->AppendItem(buffernode,bufferdata[i+j]);
			}
		}
		isLogClear=false;
	}
	m_log->Expand(rootnode);
}


void DlgDbgBuffers::OnSaveInfo(wxCommandEvent& event) {

	wxFileDialog dialog(this,
		 wxT("Buffer info save file dialog"),
		wxEmptyString,
		wxT("bufferinfo.txt"),
		wxT("Text files (*.txt)|*.txt"),
		wxFD_SAVE|wxFD_OVERWRITE_PROMPT);

	wxTreeItemId rootnode = m_log->GetRootItem();

	if (wxID_OK == dialog.ShowModal ()) {
		
		wxString path = dialog.GetPath ();
	   
		fstream s;
		s.open(path.mb_str(), fstream::out);
		unsigned int lines = m_log->GetCount();
		
		int nodelevel = 0;
		s <<  m_log->GetItemText(rootnode) << "\n"; 

		OnSaveInfoAux(s, rootnode, nodelevel+1);

		s.close();
	}
}


void DlgDbgBuffers::OnSaveInfoAux(fstream &s, wxTreeItemId parent, int nodelevel) {
	wxTreeItemIdValue cookie;
	wxTreeItemId currentChild;

	currentChild = m_log->GetFirstChild(parent, cookie);

	while (currentChild.IsOk()){
		for (int nl = 0; nl < nodelevel; nl++){
			s << "\t";
		}
		s <<  m_log->GetItemText(currentChild) << "\n"; 
		OnSaveInfoAux(s, currentChild, nodelevel+1);
		currentChild = m_log->GetNextChild(parent, cookie);
	}
}

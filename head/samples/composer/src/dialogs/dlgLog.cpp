#include "dlgLog.h"

#include <nau.h>
#include <nau/debug/profile.h>

#include <fstream>


BEGIN_EVENT_TABLE(DlgLog, wxDialog)
	EVT_BUTTON(DLG_BTN_CLEARLOG, OnClearLog)
	EVT_BUTTON(DLG_BTN_PROFILERLOG, OnProfilerLog)
	EVT_BUTTON(DLG_BTN_SAVELOG, OnSaveLog)

END_EVENT_TABLE()


wxWindow *DlgLog::m_Parent = NULL;
DlgLog *DlgLog::m_Inst = NULL;


void 
DlgLog::SetParent(wxWindow *p) {

	m_Parent = p;
}


DlgLog* 
DlgLog::Instance () {

	if (m_Inst == NULL)
		m_Inst = new DlgLog();

	return m_Inst;
}
 

DlgLog::DlgLog(): wxDialog(DlgLog::m_Parent, -1, wxT("Nau - LOG"),wxDefaultPosition,
						   wxDefaultSize,wxRESIZE_BORDER|wxDEFAULT_DIALOG_STYLE) {

	this->SetSizeHints( wxDefaultSize, wxDefaultSize);

	wxBoxSizer *bSizer1;
	bSizer1 = new wxBoxSizer( wxVERTICAL);

	wxStaticBoxSizer * sbSizer1;
	sbSizer1 = new wxStaticBoxSizer( new wxStaticBox( this, wxID_ANY, wxEmptyString ), wxVERTICAL );

	m_Log = new wxListBox(this,NULL,wxDefaultPosition,wxDefaultSize,0,NULL,wxLB_SINGLE | wxLB_HSCROLL | wxEXPAND);
	sbSizer1->Add(m_Log, 1, wxALL|wxEXPAND, 5);

	bSizer1->Add(sbSizer1, 1, wxEXPAND, 5);


	wxBoxSizer* bSizer2;
	bSizer2 = new wxBoxSizer( wxHORIZONTAL );

	m_bClear = new wxButton(this,DLG_BTN_CLEARLOG,wxT("Clear"));
	m_bProfiler = new wxButton(this,DLG_BTN_PROFILERLOG,wxT("Profiler"));
	m_bSave = new wxButton(this,DLG_BTN_SAVELOG,wxT("Save"));

	bSizer2->Add(m_bClear, 0, wxALL, 5);
	bSizer2->Add(m_bProfiler, 0, wxALL, 5);
	bSizer2->Add(m_bSave, 0, wxALL, 5);

	bSizer1->Add(bSizer2, 0, wxALIGN_CENTER|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxEXPAND|wxSHAPED, 5);

	this ->SetSizer(bSizer1);
	this->Layout();
	this->Centre(wxBOTH);

	EVENTMANAGER->addListener("LOG",this);
}


DlgLog::~DlgLog() {

}


void
DlgLog::updateDlg() {

	EVENTMANAGER->removeListener("LOG",this);
	EVENTMANAGER->addListener("LOG",this);
}


std::string &
DlgLog::getName () {

	name = "DlgLog";
	return(name);
}


void
DlgLog::eventReceived(const std::string &sender, const std::string &eventType, 
	const std::shared_ptr<IEventData> &evt) {

	if (eventType == "LOG") {
		append(*(std::string *)evt->getData());
	}
}


void DlgLog::append(std::string s) {

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


void DlgLog::clear() {

	m_Log->Clear();
}


void DlgLog::OnClearLog(wxCommandEvent& event) {

	clear();
}


void DlgLog::OnProfilerLog(wxCommandEvent& event) {

	wxFont f = wxSystemSettings::GetFont(wxSYS_OEM_FIXED_FONT);
	m_Log->SetFont(f);

	std::string s = Profile::DumpLevels();
	unsigned int i = 0, j;
	while  (i < s.size()) {
		j = i;
		for (; s[i] != '\n'; ++i);
		std::string aux = s.substr(j, i-j);
		m_Log->AppendAndEnsureVisible(wxString(s.substr(j, i-j).c_str()));
		i++;
	}
}


void DlgLog::OnSaveLog(wxCommandEvent& event) {

	wxFileDialog dialog(this,
		 wxT("Log save file dialog"),
		wxEmptyString,
		wxT("log.txt"),
		wxT("Text files (*.txt)|*.txt"),
		wxFD_SAVE|wxFD_OVERWRITE_PROMPT);

	if (wxID_OK == dialog.ShowModal ()) {
		
		wxString path = dialog.GetPath ();
	
		fstream s;
		s.open(path.mb_str(), fstream::out);
		unsigned int lines = m_Log->GetCount();
	
		for (unsigned int i = 0; i < lines; ++i) {
	
			s << m_Log->GetString(i) << "\n"; 
		}
		s.close();
	}
}

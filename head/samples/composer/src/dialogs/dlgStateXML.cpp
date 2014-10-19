#include "DlgStateXML.h"
#include "../glInfo.h"
#include <nau.h>
#include <nau/debug/profile.h>

BEGIN_EVENT_TABLE(DlgStateXML, wxDialog)
EVT_BUTTON(DLG_BTN_SAVELOG, OnSaveInfo)
EVT_BUTTON(DLG_BTN_LOADXML, OnLoadXML)
END_EVENT_TABLE()


wxWindow *DlgStateXML::m_Parent = NULL;
DlgStateXML *DlgStateXML::m_Inst = NULL;


void 
DlgStateXML::SetParent(wxWindow *p) {

	m_Parent = p;
}


DlgStateXML* 
DlgStateXML::Instance () {

	if (m_Inst == NULL)
		m_Inst = new DlgStateXML();

	return m_Inst;
}
 

DlgStateXML::DlgStateXML(): wxDialog(DlgStateXML::m_Parent, -1, wxT("Nau - Program Information"),wxDefaultPosition,
						   wxDefaultSize,wxRESIZE_BORDER|wxDEFAULT_DIALOG_STYLE)
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize);

	wxBoxSizer *bSizer1;
	bSizer1 = new wxBoxSizer( wxVERTICAL);

	wxStaticBoxSizer * sbSizer1;
	sbSizer1 = new wxStaticBoxSizer( new wxStaticBox( this, wxID_ANY, wxEmptyString ), wxVERTICAL );

	m_log = new wxPropertyGridManager(this, DLG_MI_LOG,
		wxDefaultPosition, wxDefaultSize,
		wxPG_BOLD_MODIFIED | wxPG_SPLITTER_AUTO_CENTER |
		wxPGMAN_DEFAULT_STYLE
		);
	m_log->AddPage(wxT("States"));

	sbSizer1->Add(m_log, 1, wxALL|wxEXPAND, 5);

	bSizer1->Add(sbSizer1, 1, wxEXPAND, 5);


	wxBoxSizer* bSizer2;
	bSizer2 = new wxBoxSizer( wxHORIZONTAL );

	m_bLoad = new wxButton(this, DLG_BTN_LOADXML, wxT("Load XML"));
	bSizer2->Add(m_bLoad, 0, wxALL, 5);

	m_bSave = new wxButton(this,DLG_BTN_SAVELOG,wxT("Save"));
	bSizer2->Add(m_bSave, 0, wxALL, 5);

	bSizer1->Add(bSizer2, 0, wxALIGN_CENTER|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxEXPAND|wxSHAPED, 5);

	this ->SetSizer(bSizer1);
	this->Layout();
	this->Centre(wxBOTH);
}


void
DlgStateXML::updateDlg() 
{
	std::vector<std::string> enumNames = NAU->getStateEnumNames();
	std::string value;
	wxPGProperty *enumProperty;
	wxPGProperty *root = m_log->GetCurrentPage()->GetRoot();
	for (std::string enumName : enumNames){
		value = NAU->getState(enumName);
		enumProperty = root->GetPropertyByName(enumName);
		if (!enumProperty){
			enumProperty = m_log->Append(new wxStringProperty(enumName, wxPG_LABEL, value));
			enumProperty->Enable(false);
		}
		else{
			enumProperty->SetValue(wxVariant(value));
		}
	}
}


std::string &
DlgStateXML::getName ()
{
	name = "DlgStateXML";
	return(name);
}


void
DlgStateXML::eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt)
{
}


void DlgStateXML::append(std::string s) {

}


void DlgStateXML::OnSaveInfo(wxCommandEvent& event) {
	wxFileDialog dialog(this,
		wxT("State info save file dialog"),
		wxEmptyString,
		wxT("stateinfo.txt"),
		wxT("Text files (*.txt)|*.txt"),
		wxFD_SAVE | wxFD_OVERWRITE_PROMPT);
	if (wxID_OK == dialog.ShowModal()) {

		wxString path = dialog.GetPath();

		fstream s;
		s.open(path.mb_str(), fstream::out);
		OnSavePropertyGridAux(s, m_log->GetCurrentPage());
		s.close();
	}
}


void DlgStateXML::OnSavePropertyGridAux(std::fstream &s, wxPropertyGridPage *page){
	wxPGProperty *bufferPG, *levelAux;
	wxPropertyGridIterator iterator = page->GetIterator();
	int level;
	while (!iterator.AtEnd()){
		bufferPG = iterator.GetProperty();
		level = 0;
		levelAux = bufferPG;
		while (levelAux->GetParent() != levelAux->GetMainParent()){
			levelAux = levelAux->GetParent();
			level++;
		}

		if (level == 0){
			s << bufferPG->GetBaseName().ToStdString() << ":\t" << bufferPG->GetValue().GetString().ToStdString() << "\n";
		}
		else{
			for (int i = 0; i < level; i++){
				s << "\t";
			}
			s << bufferPG->GetBaseName().ToStdString() << ":\t" << bufferPG->GetValue().GetString().ToStdString() << "\n";
		}
		iterator.Next();
	}
}


void DlgStateXML::OnLoadXML(wxCommandEvent& event) {
	static const wxChar *fileTypes = _T("XML files|*.xml|All files|*.*");

	wxFileDialog *openFileDlg = new wxFileDialog(this, _("Open File"), _(""), _(""), fileTypes, wxFD_OPEN, wxDefaultPosition);

	if (wxID_OK == openFileDlg->ShowModal()) {
		wxString path = openFileDlg->GetPath();

		try {
			NAU->loadStateXMLFile(path.ToStdString());
			updateDlg();
		}
		catch (nau::ProjectLoaderError &e) {
			wxMessageBox(wxString(e.getException()));
		}
		catch (std::string s) {
			wxMessageBox(wxString(s.c_str()));
		}

	}
}
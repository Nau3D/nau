#include "dlgAtomics.h"
#include <nau/render/iRenderer.h>
#include <nau.h>

///////////////////////////////////////////////////////////////////////////

wxWindow *DlgAtomics::parent = NULL;
DlgAtomics *DlgAtomics::inst = NULL;


void 
DlgAtomics::SetParent(wxWindow *p) {

	parent = p;
}


DlgAtomics* 
DlgAtomics::Instance () {

	if (inst == NULL)
		inst = new DlgAtomics();

	return inst;
}


DlgAtomics::DlgAtomics(): 
	wxDialog(DlgAtomics::parent, -1, wxT("Nau - Atomics"),wxDefaultPosition,wxDefaultSize,wxRESIZE_BORDER|wxDEFAULT_DIALOG_STYLE)
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	wxBoxSizer* bSizer1;
	bSizer1 = new wxBoxSizer( wxVERTICAL );
	
	m_propertyGrid1 = new wxPropertyGridManager(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxPGMAN_DEFAULT_STYLE);
	m_propertyGrid1->AddPage(wxT("Atomics"));
	m_propertyGrid1->Append(new wxFloatProperty( wxString("bla"), wxPG_LABEL ));

	bSizer1->Add( m_propertyGrid1, 1, wxEXPAND | wxALL, 5 );
	

	this->SetSizer( bSizer1 );
	this->Layout();
	
	this->Centre( wxBOTH );
}

DlgAtomics::~DlgAtomics()
{
}


void
DlgAtomics::updateDlg() {
#if NAU_OPENGL_VERSION >= 400

	IRenderer *renderer = RENDERER;

	m_propertyGrid1->Clear();
	m_propertyGrid1->AddPage(wxT("Atomics"));

	//std::map<std::pair<std::string, int>, std::string>::iterator iter;
	//iter = renderer->m_AtomicLabels.begin();
	
//	for (; iter != renderer->m_AtomicLabels.end(); ++iter) {
	for (auto iter:renderer->m_AtomicLabels) {
		m_propertyGrid1->Append(new wxFloatProperty( wxString(iter.second.c_str()), wxPG_LABEL ));

	}
#endif
}


void
DlgAtomics::update() {
#if NAU_OPENGL_VERSION >= 400
	IRenderer *renderer = RENDERER;
	std::vector<unsigned int> atValues = renderer->getAtomicCounterValues();
	std::map<std::pair<std::string, unsigned int>, std::string>::iterator iter;
	iter = renderer->m_AtomicLabels.begin();
	for (unsigned int i = 0; i < renderer->m_AtomicLabels.size(); ++i, ++iter) {
		if (m_propertyGrid1->GetProperty(wxString(iter->second.c_str())))
			m_propertyGrid1->SetPropertyValue(wxString(iter->second.c_str()),(int)(atValues[i]));
	}
#endif
}
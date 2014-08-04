#include "dlgAtomics.h"
#include <nau/render/irenderer.h>
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

	IRenderer *renderer = RENDERER;

	m_propertyGrid1->Clear();
	m_propertyGrid1->AddPage(wxT("Atomics"));

	std::map<int, std::string>::iterator iter;
	iter = renderer->m_AtomicLabels.begin();
	for (; iter != renderer->m_AtomicLabels.end(); ++iter) {
		m_propertyGrid1->Append(new wxFloatProperty( wxString(iter->second.c_str()), wxPG_LABEL ));

	}
}


void
DlgAtomics::update() {

	IRenderer *renderer = RENDERER;
	unsigned int *ac = renderer->getAtomicCounterValues();
	std::map<int, std::string>::iterator iter;
	iter = renderer->m_AtomicLabels.begin();
	for (; iter != renderer->m_AtomicLabels.end(); ++iter) {
	
		m_propertyGrid1->SetPropertyValue(wxString(iter->second.c_str()),(int)(ac[iter->first]));
	}
}
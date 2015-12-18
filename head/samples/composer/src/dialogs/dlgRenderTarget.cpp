#include "dialogs/dlgRenderTarget.h"

#include "dialogs/propertyManager.h"

#include "nau.h"
#include "nau/attribute.h"
#include "nau/event/eventFactory.h" 


BEGIN_EVENT_TABLE(DlgRenderTargets, wxDialog)

	EVT_COMBOBOX(DLG_COMBO, DlgRenderTargets::OnListSelect)
	EVT_PG_CHANGED( DLG_PROPS, DlgRenderTargets::OnPropsChange )
	
END_EVENT_TABLE()

wxWindow *DlgRenderTargets::Parent = NULL;
DlgRenderTargets *DlgRenderTargets::Inst = NULL;


void DlgRenderTargets::SetParent(wxWindow *p) {

	Parent = p;
}

DlgRenderTargets* DlgRenderTargets::Instance () {

	if (Inst == NULL)
		Inst = new DlgRenderTargets();

	return Inst;
}


DlgRenderTargets::DlgRenderTargets()
	: wxDialog(DlgRenderTargets::Parent, -1, wxT("Nau - Render Targets"),wxDefaultPosition,wxDefaultSize,wxRESIZE_BORDER|wxDEFAULT_DIALOG_STYLE)
                
{
	wxBoxSizer *sizer = new wxBoxSizer(wxVERTICAL);

	setupPanel(sizer,Parent);

	SetAutoLayout(TRUE);
	sizer->SetSizeHints(this);
	sizer->Fit(this);
    SetSizer(sizer);

	this->SetSize(350,200);
}


void
DlgRenderTargets::notifyUpdate(Notification aNot, std::string itemName, std::string value) {

	// sends events on behalf of the light
	std::shared_ptr<nau::event_::IEventData> e= nau::event_::EventFactory::Create("String");
	e->setData(&value);
	EVENTMANAGER->notifyEvent("RENDER_TARGET_CHANGED", itemName, "", e);
}


void 
DlgRenderTargets::updateDlg() {

	setupGrid();
	if (RESOURCEMANAGER->getNumRenderTargets()) {
		updateList();
		m_List->SetSelection(0);
		update();
		m_parent->Refresh();	
	}
	else {
		m_PG->Disable();
		m_List->Disable();
	}
}


void 
DlgRenderTargets::setupPanel(wxSizer *siz, wxWindow *parent) {

	int count = RESOURCEMANAGER->getNumRenderTargets();

	/* TOP: COMBO with item names */
	wxBoxSizer *sizH1 = new wxBoxSizer(wxHORIZONTAL);

	wxStaticText *stg1 =  new wxStaticText(this,-1,wxT("Render Target: "));
	m_List = new wxComboBox(this,DLG_COMBO,wxT(""),wxDefaultPosition,wxDefaultSize,0,NULL,wxCB_READONLY );

	if (count) {
		updateList();
		m_List->SetSelection(0);
	}
	sizH1->Add(stg1, 0, wxGROW|wxHORIZONTAL,5);
	sizH1->Add(m_List, 1, wxGROW|wxHORIZONTAL,5);
	siz->Add(sizH1, 0, wxGROW|wxALL, 15);

	/* MIDDLE: Property grid */

	m_PG = new wxPropertyGridManager(this, DLG_PROPS,
				wxDefaultPosition, wxDefaultSize,
				// These and other similar styles are automatically
				// passed to the embedded wxPropertyGrid.
				wxPG_BOLD_MODIFIED|wxPG_SPLITTER_AUTO_CENTER|
				// Plus defaults.
				wxPGMAN_DEFAULT_STYLE
           );

	m_PG->AddPage(wxT("Items"));

	m_PG->SetSplitterLeft(true);
	siz->Add(m_PG,1, wxEXPAND|wxALL,5);

	if (!count) {
		m_PG->Disable();
		m_List->Disable();
	}
}


void
DlgRenderTargets::setupGrid() {

	m_PG->Clear();
	m_PG->AddPage(wxT("properties"));

	PropertyManager::createGrid(m_PG, IRenderTarget::Attribs);

	m_PG->SetSplitterLeft(true, true);
}


void 
DlgRenderTargets::updateList() {

	std::vector<std::string> *names = RESOURCEMANAGER->getRenderTargetNames();
	int num = names->size();

	m_List->Clear();

	for(int i = 0; i < num; i++)  {
		wxString s;
		s << i;
		m_List->Append(wxString(names->at(i).c_str()));
	}
	m_active = names->at(0);

	delete names;
}


void 
DlgRenderTargets::update() {

	nau::render::IRenderTarget *elem;
	elem = RESOURCEMANAGER->getRenderTarget(m_active);

	setupGrid();

	PropertyManager::updateGrid(m_PG, IRenderTarget::Attribs, (AttributeValues *)elem);

	elem = RESOURCEMANAGER->getRenderTarget(m_active);

	ITexture *t;
	wxPGProperty *pid;
	t = elem->getDepthTexture();
	if (t) {
		pid = m_PG->Append(new wxStringProperty(wxT("Depth Texture"), wxPG_LABEL));
		m_PG->SetPropertyValue(pid, wxString(t->getLabel().c_str()));
		m_PG->DisableProperty(pid);
	}
	t = elem->getStencilTexture();
	if (t) {
		pid = m_PG->Append(new wxStringProperty(wxT("Stencil Texture"), wxPG_LABEL));
		m_PG->SetPropertyValue(pid, wxString(t->getLabel().c_str()));
		m_PG->DisableProperty(pid);
	}

	unsigned int k = elem->getNumberOfColorTargets();
	std::string s = "Color Texture ";
	std::string res;
	for (unsigned int i = 0; i < k; ++i) {

		t = elem->getTexture(i);
		if (t) {
			res = s + std::to_string(i);
			pid = m_PG->Append(new wxStringProperty(res.c_str(), wxPG_LABEL));
			m_PG->SetPropertyValue(pid, wxString(t->getLabel().c_str()));
			m_PG->DisableProperty(pid);
		}
	}

}


void DlgRenderTargets::OnPropsChange( wxPropertyGridEvent& e) {

	nau::render::IRenderTarget *elem;
	elem = RESOURCEMANAGER->getRenderTarget(m_active);

	const wxString& name = e.GetPropertyName();
	unsigned int dotLocation = name.find_first_of(wxT("."),0);
	std::string topProp = std::string(name.substr(0,dotLocation).mb_str());
	std::string prop = std::string(name.substr(dotLocation+1,name.size()-dotLocation-1).mb_str());
	wxVariant variant;
	wxColour col;

	PropertyManager::updateProp(m_PG, topProp, IRenderTarget::Attribs, (AttributeValues *)elem);
	notifyUpdate(PROPS_CHANGED,m_active,topProp);
}


void DlgRenderTargets::updateInfo(std::string name) {

	if (name == m_active) {
		update();
	}
}


void DlgRenderTargets::OnListSelect(wxCommandEvent& event) {

	int sel;

	sel = event.GetSelection();
	m_active = std::string(m_List->GetString(sel).mb_str());
	update();
}

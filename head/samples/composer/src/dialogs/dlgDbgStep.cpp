#include "DlgDbgStep.h"
#include <nau.h>
#include <nau/debug/profile.h>


#ifdef GLINTERCEPTDEBUG
#include "..\..\GLIntercept\Src\MainLib\ConfigDataExport.h"
#endif

BEGIN_EVENT_TABLE(DlgDbgStep, wxDialog)
	EVT_BUTTON(DLG_BTN_NEXTPASS, OnNextPass)
	EVT_BUTTON(DLG_BTN_NEXTFRAME, OnNextFrame)
	EVT_BUTTON(DLG_BTN_TOPASS, OnToPass)
END_EVENT_TABLE()


wxWindow *DlgDbgStep::m_Parent = NULL;
GLCanvas *DlgDbgStep::m_Canvas = NULL;
DlgDbgStep *DlgDbgStep::m_Inst = NULL;


void 
DlgDbgStep::SetParent(wxWindow *p) {

	m_Parent = p;
}


void
DlgDbgStep::SetCanvas(GLCanvas *c) {

	m_Canvas = c;
}


DlgDbgStep* 
DlgDbgStep::Instance () {

	if (m_Inst == NULL)
		m_Inst = new DlgDbgStep();

	return m_Inst;
}
 

DlgDbgStep::DlgDbgStep(): wxDialog(DlgDbgStep::m_Parent, -1, wxT("Nau - Frame pass Information"),wxDefaultPosition,
						   wxDefaultSize,wxRESIZE_BORDER|wxDEFAULT_DIALOG_STYLE)
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize);



	wxBoxSizer *sizer = new wxBoxSizer(wxVERTICAL);

	/* ----------------------------------------------------------------
	Pipelines and Passes
	-----------------------------------------------------------------*/

	wxBoxSizer *bSizer1;
	bSizer1 = new wxBoxSizer(wxVERTICAL);

	wxBoxSizer *sizerh = new wxBoxSizer(wxHORIZONTAL);
	wxStaticText *stg1 = new wxStaticText(this, -1, wxT("Pipeline: "));

	sizerh->Add(stg1, 0, wxGROW | wxALL, 5);

	bSizer1->Add(sizerh, 0, wxGROW | wxALL, 5);

	wxStaticBoxSizer * sbSizer1;
	sbSizer1 = new wxStaticBoxSizer(new wxStaticBox(this, wxID_ANY, wxEmptyString), wxVERTICAL);

	m_list = new wxListBox(this, NULL, wxDefaultPosition, wxDefaultSize, 0, NULL, wxLB_SINGLE | wxLB_HSCROLL | wxEXPAND);

	sbSizer1->Add(m_list, 1, wxALL|wxEXPAND, 5);

	bSizer1->Add(sbSizer1, 1, wxEXPAND, 5);


	wxBoxSizer* bSizer2;
	bSizer2 = new wxBoxSizer(wxHORIZONTAL);

	m_bNextPass = new wxButton(this, DLG_BTN_NEXTPASS, wxT("Next Pass"));

	bSizer2->Add(m_bNextPass, 0, wxALL, 5);

	m_bNextFrame = new wxButton(this, DLG_BTN_NEXTFRAME, wxT("Next Frame"));

	bSizer2->Add(m_bNextFrame, 0, wxALL, 5);

	m_bNextToPass = new wxButton(this, DLG_BTN_TOPASS, wxT("Execute Pass"));

	bSizer2->Add(m_bNextToPass, 0, wxALL, 5);

	bSizer1->Add(bSizer2, 0, wxALIGN_CENTER|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxEXPAND|wxSHAPED, 5);


	this->SetSizer(bSizer1);
	this->Layout();
	this->Centre(wxBOTH);

	pipenameString = "";
}


void
DlgDbgStep::updateDlg() {

	std::string newPipeName = RENDERMANAGER->getActivePipelineName();
	if (pipenameString.compare(newPipeName) != 0){
		pipenameString = newPipeName;
		getPasses(pipenameString);
	}
	int nextIndex = 0;
	m_list->Clear();
	
	if (pipenameString.size() > 0 && passes.size()){

		Pass *currentPass = RENDERMANAGER->getCurrentPass();
		std::string passCurrentName;
		if (currentPass){
			passCurrentName = currentPass->getName();
		}

		for (auto &passName :passes){
			if (passCurrentName.compare(passName) == 0){
				m_list->AppendAndEnsureVisible(wxString("Next:> " + passName));
				m_list->SetSelection(nextIndex);
				currentPassIndex = nextIndex;
			}
			else{
				m_list->AppendAndEnsureVisible(wxString(passName));
			}
			nextIndex++;
		}

	}
	m_list->Refresh();
}


void
DlgDbgStep::getPasses(std::string pipenameString) {
	Pipeline *pip = RENDERMANAGER->getPipeline(pipenameString);

	passes.clear();
	pip->getPassNames(&passes);
}




std::string &
DlgDbgStep::getName () {

	name = "DlgDbgStep";
	return(name);
}


void
DlgDbgStep::eventReceived(const std::string &sender, const std::string &eventType, 
	const std::shared_ptr<nau::event_::IEventData> &evt)
{
}


void DlgDbgStep::append(std::string s) {

}


void DlgDbgStep::OnNextPass(wxCommandEvent& event){

	if (m_Canvas->IsPaused()){
#ifdef GLINTERCEPTDEBUG 
		gliSetIsGLIActive(true);
#endif
		m_Canvas->StepPass();
#ifdef GLINTERCEPTDEBUG 
		gliSetIsGLIActive(false);
#endif
		updateDlg();
	}
}


void DlgDbgStep::OnNextFrame(wxCommandEvent& event){

	if (m_Canvas->IsPaused()){
#ifdef GLINTERCEPTDEBUG 
		gliSetIsGLIActive(true);
#endif
		m_Canvas->StepToEndOfFrame();
#ifdef GLINTERCEPTDEBUG 
		gliSetIsGLIActive(false);
#endif
		updateDlg();
	}
}


void DlgDbgStep::OnToPass(wxCommandEvent& event){


	if (m_Canvas->IsPaused()){
#ifdef GLINTERCEPTDEBUG 
		gliSetIsGLIActive(true);
#endif
		m_Canvas->StepUntilSamePassNextFrame();
#ifdef GLINTERCEPTDEBUG 
		gliSetIsGLIActive(false);
#endif
		updateDlg();
	}
}
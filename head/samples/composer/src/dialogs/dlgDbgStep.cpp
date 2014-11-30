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
GlCanvas *DlgDbgStep::m_Canvas = NULL;
DlgDbgStep *DlgDbgStep::m_Inst = NULL;


void 
DlgDbgStep::SetParent(wxWindow *p) {

	m_Parent = p;
}


void
DlgDbgStep::SetCanvas(GlCanvas *c) {

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
	//wxTextCtrl  *pipename = new wxTextCtrl(this, DLG_TXT_PIPELINE, wxT(""));

	sizerh->Add(stg1, 0, wxGROW | wxALL, 5);
	//sizerh->Add(pipename, 0, wxGROW | wxALL, 5);


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

	passes = 0;
	pipenameString = "";
}


void
DlgDbgStep::updateDlg()
{
	//if (pipename){
	//	std::string pipenameString = RENDERMANAGER->getActivePipelineName();
	//	wxString str = wxString(pipenameString.c_str());
	//	pipename->SetValue(str);
	//}
	std::vector<std::string>::iterator iter;
	std::string newPipeName = RENDERMANAGER->getActivePipelineName();
	if (pipenameString.compare(newPipeName) != 0){
		pipenameString = newPipeName;
		getPasses(pipenameString);
	}
	int nextIndex = 0;
	m_list->Clear();
	
	if (pipenameString.size() > 0 && passes){

		Pass *currentPass = RENDERMANAGER->getCurrentPass();
		std::string passCurrentName;
		if (currentPass){
			passCurrentName = currentPass->getName();
		}



		
		for (iter = passes->begin(); iter != passes->end(); ++iter){
			std::string passName = *iter;
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
}


void
DlgDbgStep::getPasses(std::string pipenameString){
	Pipeline *pip = RENDERMANAGER->getPipeline(pipenameString);

	if (passes){
		delete passes;

	}
	passes = pip->getPassNames();

}




std::string &
DlgDbgStep::getName ()
{
	name = "DlgDbgStep";
	return(name);
}


void
DlgDbgStep::eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt)
{
}


void DlgDbgStep::append(std::string s) {

}




void DlgDbgStep::OnNextPass(wxCommandEvent& event){

#ifdef GLINTERCEPTDEBUG 
	if (m_Canvas->IsPaused()){
		gliSetIsGLIActive(true);
		m_Canvas->MultiStep();
	}
#endif
}


void DlgDbgStep::OnNextFrame(wxCommandEvent& event){

#ifdef GLINTERCEPTDEBUG 
	if (m_Canvas->IsPaused() && passes){
		int end = passes->size();
		int stepsize = end - currentPassIndex;
		gliSetIsGLIActive(true);
		m_Canvas->MultiStep(stepsize);
	}
#endif
}


void DlgDbgStep::OnToPass(wxCommandEvent& event){
#ifdef GLINTERCEPTDEBUG 
	if (m_Canvas->IsPaused() && passes){
		int end = passes->size();
		int item = m_list->GetSelection();
		int stepsize;
		if (item < currentPassIndex){
			stepsize = (end - currentPassIndex) + item + 1;
		}
		else{
			stepsize = item - currentPassIndex + 1;
		}
		if (item != wxNOT_FOUND){
			gliSetIsGLIActive(true);
			m_Canvas->MultiStep(stepsize);
		}
	}
#endif
}
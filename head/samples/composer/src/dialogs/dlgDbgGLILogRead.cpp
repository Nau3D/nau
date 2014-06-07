#include "dlgDbgGLILogRead.h"
#include <nau.h>
#include <nau/debug/profile.h>
#include "..\..\GLIntercept\Src\MainLib\ConfigDataExport.h"

BEGIN_EVENT_TABLE(DlgDbgGLILogRead, wxDialog)

END_EVENT_TABLE()


wxWindow *DlgDbgGLILogRead::m_Parent = NULL;
DlgDbgGLILogRead *DlgDbgGLILogRead::m_Inst = NULL;


void 
DlgDbgGLILogRead::SetParent(wxWindow *p) {

	m_Parent = p;
}


DlgDbgGLILogRead* 
DlgDbgGLILogRead::Instance () {

	if (m_Inst == NULL)
		m_Inst = new DlgDbgGLILogRead();

	return m_Inst;
}
 

DlgDbgGLILogRead::DlgDbgGLILogRead(): wxDialog(DlgDbgGLILogRead::m_Parent, -1, wxT("Nau - GLINTERCEPT LOG"),wxDefaultPosition,
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

	this ->SetSizer(bSizer1);
	this->Layout();
	this->Centre(wxBOTH);

	isLogClear=true;
}


void
DlgDbgGLILogRead::updateDlg() 
{

}


std::string &
DlgDbgGLILogRead::getName ()
{
	name = "DlgDbgGLILogRead";
	return(name);
}


void
DlgDbgGLILogRead::eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt)
{
}


void DlgDbgGLILogRead::append(std::string s) {

}


void DlgDbgGLILogRead::clear() {

	m_log->DeleteAllItems();
	isLogClear=true;
}


void DlgDbgGLILogRead::loadLog() {
	wxTreeItemId rootnode;
	wxTreeItemId frame;
	string logname = gliGetLogName();
	string logfile = gliGetLogPath()+logname+".txt";
	ifstream filestream(logfile);
	string line;
	int frameNumber = 0;
	bool isNewFrame = true;
	rootnode = m_log->AddRoot(logname);
	
		//DEBUG
		//std::vector<std::string> enums;
		//int count = gliGetEnumsCount();
		//
		//for (int i=0;i<count;i++){
		//		m_log->AppendItem(rootnode,gliGetEnumsName(i));

		//}

	if (isLogClear){
		if (filestream){
	
			getline(filestream,line);
			getline(filestream,line);
			getline(filestream,line);
			getline(filestream,line);
			getline(filestream,line);
		
			while (std::getline(filestream, line))
			{
				if (isNewFrame){
					frame = m_log->AppendItem(rootnode, "frame "+to_string(frameNumber));
					frameNumber++;
					isNewFrame=false;
				}
				m_log->AppendItem(frame,line);
				if(strcmp(line.substr(0, strlen("wglSwapBuffers")).c_str(), "wglSwapBuffers") == 0){
					isNewFrame=true;
				}
			}

		}
		else{
			frame = m_log->AppendItem(rootnode, "logfile not found");
		}
		filestream.close();

		m_log->Expand(rootnode);
		isLogClear=false;
	}

}

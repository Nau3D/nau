#include "dlgDbgPrograms.h"
#include "../glInfo.h"
#include <nau.h>
#include <nau/debug/profile.h>

BEGIN_EVENT_TABLE(DlgDbgPrograms, wxDialog)
	EVT_BUTTON(DLG_BTN_SAVELOG, OnSaveInfo)
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

	isLogClear = true;
	//isRecording = true;
	//frameNumber = 0;
}


void
DlgDbgPrograms::updateDlg() 
{

}


std::string &
DlgDbgPrograms::getName ()
{
	name = "DlgDbgPrograms";
	return(name);
}


void
DlgDbgPrograms::eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt)
{
}


void DlgDbgPrograms::append(std::string s) {

}


void DlgDbgPrograms::clear() {

	m_log->DeleteAllItems();
	isLogClear=true;
}

void DlgDbgPrograms::loadShaderInfo() {
	//if (isRecording){
		//wxTreeItemId framenode;
		std::vector<unsigned int> programs = getProgramNames();

		if (isLogClear){
			rootnode = m_log->AddRoot("Shader Uniforms>");//infostream.str()
			isLogClear = false;
			m_log->Expand(rootnode);
		}

		//framenode = m_log->AppendItem(rootnode, "Frame " + to_string(frameNumber) + ">");
		//frameNumber++;
		for (int i = 0; i < programs.size(); i++){
			loadProgramInfo(rootnode, programs[i]);
		}
	//}
}

void DlgDbgPrograms::loadProgramInfo(wxTreeItemId basenode, unsigned int program){
    wxTreeItemId programnode;
	programnode = m_log->AppendItem(basenode,"Program " + to_string(program) + ">");
	
	//loadStandardProgramInfo(programnode, program);
	loadProgramAttributesInfo(programnode, program);
	loadProgramUniformsInfo(programnode, program);

}


void DlgDbgPrograms::loadProgramUniformsInfo(wxTreeItemId basenode, unsigned int program){
	wxTreeItemId uniformsnode, blocknode, uniformnode;
    std::vector<string> uniformNames, blockNames;
	uniformsnode = m_log->AppendItem(basenode,"Uniforms>");
	blocknode = m_log->AppendItem(uniformsnode,"Default block>");

	getUniformNames(program,uniformNames);

	for (int un = 0; un < uniformNames.size(); un++){
		uniformnode = m_log->AppendItem(blocknode,uniformNames[un] + ">");
		loadUniformInfo(uniformnode, program, "", uniformNames[un]);
	}

	getBlockNames(program, blockNames);

	for (int b = 0; b < blockNames.size(); b++){
		blocknode = m_log->AppendItem(uniformsnode,blockNames[b]);
		loadBlockInfo(blocknode, program, blockNames[b]);
	}
}

                    
void DlgDbgPrograms::loadUniformInfo(wxTreeItemId basenode, unsigned int program, std::string blockName, std::string uniformName){
	wxTreeItemId valuesnode;
	string uniformType;
	int uniformSize, uniformArrayStride;
	vector<string> values;

	if (blockName.length()>0){
		getUniformData(program, blockName, uniformName, uniformType, uniformSize, uniformArrayStride, values);
	}
	else{
		getUniformData(program, uniformName, uniformType, uniformSize, uniformArrayStride, values);
	}
	m_log->AppendItem(basenode,"Type: "+uniformType);
	m_log->AppendItem(basenode,"Size: "+to_string(uniformSize));
	if (uniformArrayStride>0)
		m_log->AppendItem(basenode,"Array stride: "+to_string(uniformArrayStride));
	if (values.size()>0){
		if (values.size()==1){
			m_log->AppendItem(basenode,"Values: "+values[0]);
		}
		else{
			valuesnode = m_log->AppendItem(basenode,"Values>");
			for (int i = 0; i < values.size(); i++){
				m_log->AppendItem(valuesnode,values[i]);
			}
			m_log->Expand(valuesnode);
		}
	}
}

                    
void DlgDbgPrograms::loadBlockInfo(wxTreeItemId basenode, unsigned int program, std::string blockName){
	wxTreeItemId infonode, uniformsnode, uniformnode;
	int datasize, blockbindingpoint, bufferbindingpoint;
    std::vector<string> uniformNames;
	
	infonode = m_log->AppendItem(basenode, "Info>");
	uniformsnode = m_log->AppendItem(basenode, "Uniforms>");

	getBlockData(program, blockName, datasize, blockbindingpoint, bufferbindingpoint, uniformNames);
	
	m_log->AppendItem(infonode, "Size: " + to_string(datasize));
	m_log->AppendItem(infonode, "Block binding point: " + to_string(blockbindingpoint));
	m_log->AppendItem(infonode, "Buffer binding point: " + to_string(bufferbindingpoint));
	
	for (int un = 0; un < uniformNames.size(); un++){
		uniformnode = m_log->AppendItem(uniformsnode,uniformNames[un]);
		loadUniformInfo(uniformnode, program, blockName, uniformNames[un]);
	}
}

void DlgDbgPrograms::loadStandardProgramInfo(wxTreeItemId basenode, unsigned int program){
	bool tess = false;
	wxTreeItemId infonode, shadersnode, geomnode, tessnode;
	vector<pair<string,char>> shadersInfo;
	vector<string> stdInfo, geomInfo, tessInfo;
	getProgramInfoData(program, shadersInfo, stdInfo,  geomInfo,  tessInfo);

	infonode = m_log->AppendItem(basenode, "Program info>");
	for (int i = 0; i < stdInfo.size(); i++){
		m_log->AppendItem(infonode, stdInfo[i]);
	}

	shadersnode = m_log->AppendItem(basenode, "Shaders>");
	for (int i = 0; i < shadersInfo.size(); i++){
		switch (shadersInfo[i].second ){
		case 'g':
			geomnode = m_log->AppendItem(shadersnode, shadersInfo[i].first + " >");
			for (int j = 0; j < geomInfo.size(); j++){
				m_log->AppendItem(geomnode, geomInfo[j]);
			}
			break;
		case 't':
			if (!tess){
				tessnode = m_log->AppendItem(shadersnode, "GL_TESSELATION_SHADER>");
				tess = true;
			}
			m_log->AppendItem(tessnode, shadersInfo[i].first);
			break;
		default:
			m_log->AppendItem(shadersnode, shadersInfo[i].first);
			break;
		}
	}

	if (tess){
		for (int i = 0; i < tessInfo.size(); i++){
			m_log->AppendItem(tessnode, tessInfo[i]);
		}
	}
}

void DlgDbgPrograms::loadProgramAttributesInfo(wxTreeItemId basenode, unsigned int program){
	wxTreeItemId attributesnode, attributenode;
	std::vector<std::pair<std::string, std::pair<int,std::string>>> attributeList;

	attributesnode = m_log->AppendItem(basenode, "Attributes>");
	getAttributesData(program, attributeList);


	for (int i = 0; i < attributeList.size(); i++){
		attributenode = m_log->AppendItem(attributesnode, attributeList[i].first + ">");
		m_log->AppendItem(attributenode, "Location: " + to_string(attributeList[i].second.first));
		m_log->AppendItem(attributenode, "Type: " + attributeList[i].second.second);
	}

}


void DlgDbgPrograms::OnSaveInfo(wxCommandEvent& event) {

	wxFileDialog dialog(this,
		 wxT("Program info save file dialog"),
		wxEmptyString,
		wxT("programinfo.txt"),
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


void DlgDbgPrograms::OnSaveInfoAux(fstream &s, wxTreeItemId parent, int nodelevel) {
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

//void DlgDbgPrograms::startRecording(){
//	isRecording = true;
//	frameNumber = 0;
//}
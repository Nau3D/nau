#include "dlgTrace.h"
#include <nau.h>
#include <nau/debug/profile.h>
#include <nau/slogger.h>
#include "..\..\GLIntercept\Src\MainLib\ConfigDataExport.h"
#include <dirent.h>
#include <algorithm>

BEGIN_EVENT_TABLE(DlgDbgGLILogRead, wxDialog)

END_EVENT_TABLE()


wxWindow *DlgDbgGLILogRead::m_Parent = NULL;
DlgDbgGLILogRead *DlgDbgGLILogRead::m_Inst = NULL;

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		elems.push_back(item);
	}
	return elems;
}


std::vector<std::string> split(const std::string &s, char delim) {
	std::vector<std::string> elems;
	split(s, delim, elems);
	return elems;
}


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
 

DlgDbgGLILogRead::DlgDbgGLILogRead(): wxDialog(DlgDbgGLILogRead::m_Parent, -1, wxT("Nau - Trace Log"),wxDefaultPosition,
						   wxDefaultSize,wxRESIZE_BORDER|wxDEFAULT_DIALOG_STYLE)
{

	this->SetSizeHints( wxDefaultSize, wxDefaultSize);

	wxBoxSizer *bSizer1;
	bSizer1 = new wxBoxSizer( wxVERTICAL);

	wxStaticBoxSizer * sbSizer1;
	sbSizer1 = new wxStaticBoxSizer( new wxStaticBox( this, wxID_ANY, wxEmptyString ), wxVERTICAL );

	m_Log = new wxTreeCtrl(this,NULL,wxDefaultPosition,wxDefaultSize,wxLB_SINGLE | wxLB_HSCROLL | wxEXPAND);

	sbSizer1->Add(m_Log, 1, wxALL|wxEXPAND, 5);

	bSizer1->Add(sbSizer1, 1, wxEXPAND, 5);

	this ->SetSizer(bSizer1);
	this->Layout();
	this->Centre(wxBOTH);

	isLogClear=true;
	nextFunctionIndex = 0;
	numGLFunctionCalls = 0;
	statsnode = NULL;
}


void
DlgDbgGLILogRead::updateDlg() {

}


std::string &
DlgDbgGLILogRead::getName () {

	name = "DlgTraceRead";
	return(name);
}


void
DlgDbgGLILogRead::eventReceived(const std::string &sender, const std::string &eventType, 
	nau::event_::IEventData *evt) {

}


void 
DlgDbgGLILogRead::append(std::string s) {

}


void 
DlgDbgGLILogRead::clear() {

	m_Log->DeleteAllItems();
	isLogClear = true;
	statsnode = NULL;
	nextFunctionIndex = 0;
	numGLFunctionCalls = 0;
	functionDataArray.clear();
	functionIndexList.clear();
}


void 
DlgDbgGLILogRead::loadNewLogFile(string logfile, int fNumber, bool tellg, bool appendCount) {

	string line;
	isNewFrame = false;
	filestream.clear();
	filestream.open(logfile);
	
	if (fNumber >= 0){
		frameNumber=fNumber;
	}
	if (filestream){

		if (tellg){
			filestream.seekg(streamlnnum);
			getline(filestream, line);
			if (strcmp(line.substr(0, 4).c_str(), "#NAU") == 0){
				if (strcmp(line.substr(5, 5).c_str(), "FRAME") == 0){
					if (strcmp(line.substr(11, 5).c_str(), "START") == 0){
						PrintFunctionCount();
						ZeroStatsHeaders();
						frameStatNumber = frameNumber;
						frame = m_Log->AppendItem(lognode, "Frame " + to_string(frameNumber) + " >");
						statsnode = m_Log->AppendItem(frame, "Statistics Log >");
						frame = m_Log->AppendItem(frame, "Call Log >");
						pass = frame;
						frameNumber++;
						isNewFrame = false;
					}
				}
			}
		}
		else{
			//getline(filestream, line);
			//getline(filestream, line);
			//getline(filestream, line);
			//getline(filestream, line);
			//getline(filestream, line);
			isNewFrame = true;
		}

		while (getline(filestream, line))
		{
			if (strcmp(line.substr(0, 4).c_str(), "#NAU") == 0){
				if (strcmp(line.substr(5, 5).c_str(), "FRAME") == 0){
					if (strcmp(line.substr(11, 5).c_str(), "START") == 0){
						if (frame){
							m_Log->Expand(frame);
						}
						PrintFunctionCount();
						ZeroStatsHeaders();
						frameStatNumber = frameNumber;
						frame = m_Log->AppendItem(lognode, "Frame " + to_string(frameNumber) + " >");
						statsnode = m_Log->AppendItem(frame, "Statistics Log >");
						frame = m_Log->AppendItem(frame, "Call Log >");
						pass = frame;
						frameNumber++;
						isNewFrame = false;
					}
				}
				else if (strcmp(line.substr(5, 4).c_str(), "PASS") == 0){
					if (strcmp(line.substr(10, 5).c_str(), "START") == 0){
						pass = m_Log->AppendItem(frame, "NAUPASS(" + line.substr(16, line.length() - 17) + ") >");
					}
					else if (strcmp(line.substr(10, 3).c_str(), "END") == 0){
						pass = frame;
					}
				}
			}
			else{
				if (isNewFrame){
					if (frame){
						m_Log->Expand(frame);
					}
					PrintFunctionCount();
					ZeroStatsHeaders();
					frameStatNumber = frameNumber;
					frame = m_Log->AppendItem(lognode, "Frame " + to_string(frameNumber) + " >");
					statsnode = m_Log->AppendItem(frame, "Statistics Log >");
					frame = m_Log->AppendItem(frame, "Call Log >");
					pass = frame;
					frameNumber++;
					isNewFrame = false;
				}
				m_Log->AppendItem(pass, line);
				if (appendCount){
					CountFunction(split(line, '(')[0]);
				}
			}
			if (filestream.tellg() >= 0){
				streamlnnum = filestream.tellg();
			}
		}

	}
	//if (gliIsLogPerFrame()){
		filestream.close();
	//}
}

void 
DlgDbgGLILogRead::finishReadLogFile() {
	filestream.close();
}


void 
DlgDbgGLILogRead::loadLog() {

	string logname = gliGetLogName();
	string logfile;
	
	if (isLogClear) {

		//if (!gliIsLogPerFrame()){
		//	// Corresponding logfile
		//	logfile = gliGetLogPath()+logname+".txt";
		//	rootnode = m_Log->AddRoot(logfile);
		//	lognode = m_Log->AppendItem(rootnode, "Frame Log>");

		//	//Reads logfile
		//	loadNewLogFile(logfile, 0, false, true);

		//	//If no logfile was found then leave a message
		//	if (m_Log->GetChildrenCount(rootnode, false) == 0){
		//		m_Log->AppendItem(rootnode, "logfile not found");
		//	}
		//	
		//}
		//else
		{
			rootnode = m_Log->AddRoot(string("./__nau3Dtrace/") + string("Frame_*.txt"));
			lognode = m_Log->AppendItem(rootnode, "Frame Log >");
			//Directory searching algorithm source:
			//dirent.h
			DIR *dir;
			struct dirent *ent;
			if ((dir = opendir ("./__nau3Dtrace")) != NULL) {
				// Read all files and directories in the directory
				while ((ent = readdir (dir)) != NULL) {
					// Filters directories starting with Frame_* only
					if (ent->d_type == S_IFREG && strstr(ent->d_name, "Frame_")){
						// Corresponding logfile
						logfile = string("./__nau3Dtrace/")+string(ent->d_name);

						//Reads logfile
						loadNewLogFile(logfile, atoi(ent->d_name + 6), false, true);
					}
				}
				closedir (dir);
			}

			//If no logfile was found then leave a message
			if (m_Log->GetChildrenCount(lognode, false) == 0){
				m_Log->AppendItem(rootnode, "no related logfiles were found");
			}
		}
		m_Log->Expand(rootnode);
		m_Log->Expand(lognode);
		isLogClear=false;
	}
	else {
/*		if (!gliIsLogPerFrame()){
			logfile = gliGetLogPath() + logname + ".txt";
			loadNewLogFile(logfile, frameNumber, true, true);
		}
		else*/{
			clear();
			rootnode = m_Log->AddRoot(string("./__nau3Dtrace") + string("Frame_*\\") + logname + ".txt");
			lognode = rootnode;
			//Directory searching algorithm source:
			//dirent.h
			DIR *dir;
			struct dirent *ent;
			if ((dir = opendir(gliGetLogPath())) != NULL) {
				// Read all files and directories in the directory
				while ((ent = readdir(dir)) != NULL) {
					// Filters directories starting with Frame_* only


					if (ent->d_type == S_IFREG && strstr(ent->d_name, "Frame_")){
//					if (ent->d_type == S_IFDIR && strstr(ent->d_name, "Frame_")){
						// Corresponding logfile
						logfile = gliGetLogPath() + string(ent->d_name) + "/" + logname + ".txt";

						//Reads logfile
						loadNewLogFile(logfile, atoi(ent->d_name + 6), false, true);
					}
				}
				closedir(dir);
			}

			//If no logfile was found then leave a message
			if (m_Log->GetChildrenCount(lognode, false) == 0){
				m_Log->AppendItem(rootnode, "no related logfiles were found");
			}
			m_Log->Expand(rootnode);
			m_Log->Expand(lognode);
			isLogClear = false;
		
		}
	}
	PrintFunctionCount();
}


DlgDbgGLILogRead::FunctionCallData::FunctionCallData() :
	funcCallCount(0) {

}

bool
DlgDbgGLILogRead::FunctionCallData::SortByName(const FunctionCallData &a, const FunctionCallData &b) {

	return a.functionName < b.functionName;
}


bool 
DlgDbgGLILogRead::FunctionCallData::SortByCount(const FunctionCallData &a, const FunctionCallData &b) {

	return a.funcCallCount > b.funcCallCount;
}


void 
DlgDbgGLILogRead::CleanStatsHeaders(){
	m_Log->DeleteChildren(statsnode);
	statsnamenode = m_Log->AppendItem(statsnode, "Ordered by Name >");
	statscountnode = m_Log->AppendItem(statsnode, "Ordered by Count >");
}


void 
DlgDbgGLILogRead::ZeroStatsHeaders(){
	functionDataArray.clear();
	numGLFunctionCalls = 0;
}


void 
DlgDbgGLILogRead::CountFunction(std::string funcName)
{
	unsigned int funcIndex;
	if (functionIndexList.find(funcName) == functionIndexList.end()) {
		funcIndex = nextFunctionIndex;
		functionIndexList[funcName] = nextFunctionIndex;
		nextFunctionIndex++;
	}
	else {
		funcIndex = functionIndexList[funcName];
	}

	//Loop and add array entries to ensure the index value is valid
	while (funcIndex >= functionDataArray.size())
	{
		functionDataArray.push_back(FunctionCallData());
	}

	//If this function has not been called before
	if (functionDataArray[funcIndex].funcCallCount == 0)
	{
		//Assign the function name
		functionDataArray[funcIndex].functionName = funcName;
	}

	//Increment the call count for the function
	functionDataArray[funcIndex].funcCallCount++;

	//Increment the total call count
	numGLFunctionCalls++;
}


void 
DlgDbgGLILogRead::PrintFunctionCount()
{ 
	if (statsnode){
		CleanStatsHeaders();
		std::vector<FunctionCallData> functionDataArrayClone = functionDataArray;
		//Dump the total call count and average per frame (excluding first frame of xxx calls)
		m_Log->AppendItem(statsnode, "Total GL Calls: " + std::to_string(numGLFunctionCalls));
		//m_Log->AppendItem(statsnode, "Frame Number: " + std::to_string(frameStatNumber));

		//Sort the array based on function call count
		sort(functionDataArrayClone.begin(), functionDataArrayClone.end(), FunctionCallData::SortByCount);


		//Loop and dump the function data 
		for (unsigned int i = 0; i < functionDataArrayClone.size(); i++)
		{
			//Only dump functions that have been called
			if (functionDataArrayClone[i].funcCallCount > 0)
			{
				m_Log->AppendItem(statscountnode, functionDataArrayClone[i].functionName + ": " + std::to_string(functionDataArrayClone[i].funcCallCount));
			}
		}

		//Sort the array based on function name
		sort(functionDataArrayClone.begin(), functionDataArrayClone.end(), FunctionCallData::SortByName);

		//Loop and dump the function data 
		for (unsigned int i = 0; i < functionDataArrayClone.size(); i++)
		{
			//Only dump functions that have been called
			if (functionDataArrayClone[i].funcCallCount > 0)
			{
				m_Log->AppendItem(statsnamenode, functionDataArrayClone[i].functionName + ": " + std::to_string(functionDataArrayClone[i].funcCallCount));
			}
		}
	}
}


#include <nau.h>

#include "dlgTrace.h"


#include <nau/debug/profile.h>
#include <nau/slogger.h>

#include <dirent.h>
#include <algorithm>
#include <sys/types.h>
#include <sys/stat.h>

BEGIN_EVENT_TABLE(DlgTrace, wxDialog)

END_EVENT_TABLE()


wxWindow *DlgTrace::m_Parent = NULL;
DlgTrace *DlgTrace::m_Inst = NULL;

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
DlgTrace::SetParent(wxWindow *p) {

	m_Parent = p;
}


DlgTrace* 
DlgTrace::Instance () {

	if (m_Inst == NULL)
		m_Inst = new DlgTrace();

	return m_Inst;
}
 

DlgTrace::DlgTrace(): wxDialog(DlgTrace::m_Parent, -1, wxT("Nau - Trace Log"),wxDefaultPosition,
						   wxDefaultSize,wxRESIZE_BORDER|wxDEFAULT_DIALOG_STYLE)
{

	this->SetSizeHints( wxDefaultSize, wxDefaultSize);

	wxBoxSizer *bSizer1;
	bSizer1 = new wxBoxSizer( wxVERTICAL);

	wxStaticBoxSizer * sbSizer1;
	sbSizer1 = new wxStaticBoxSizer( new wxStaticBox( this, wxID_ANY, wxEmptyString ), wxVERTICAL );

	m_Log = new wxTreeCtrl(this,0,wxDefaultPosition,wxDefaultSize,wxLB_SINGLE | wxLB_HSCROLL | wxEXPAND);

	sbSizer1->Add(m_Log, 1, wxALL|wxEXPAND, 5);

	bSizer1->Add(sbSizer1, 1, wxEXPAND, 5);

	this ->SetSizer(bSizer1);
	this->Layout();
	this->Centre(wxBOTH);

	isLogClear=true;
	nextFunctionIndex = 0;
	numGLFunctionCalls = 0;
	m_Statsnode = NULL;
	time(&m_ProjectStartTime);
}


/* ----------------------------------------------------------------

EVENTS FROM OTHER DIALOGS AND NAU

-----------------------------------------------------------------*/

void
DlgTrace::eventReceived(const std::string &sender, const std::string &eventType, 
	const std::shared_ptr<IEventData> &evt) {

	if (eventType == "TRACE_FILE_READY") {
		loadLog();
	}
}


void
DlgTrace::updateDlg() {

	//FILETIME lt;
	//GetSystemTimeAsFileTime(&lt);
	//ULARGE_INTEGER uLargeIntegerTime1;
	//uLargeIntegerTime1.LowPart = lt.dwLowDateTime;
	//uLargeIntegerTime1.HighPart = lt.dwHighDateTime;
	//m_LastTime = uLargeIntegerTime1.QuadPart;

	m_Filenames.clear();
	time(&m_ProjectStartTime);
	m_Log->DeleteAllItems();
	EVENTMANAGER->addListener("TRACE_FILE_READY", this);

}


std::string &
DlgTrace::getName () {

	name = "DlgTraceRead";
	return(name);
}


void 
DlgTrace::append(std::string s) {

}


void 
DlgTrace::clear() {

	m_Log->DeleteAllItems();
	isLogClear = true;
	m_Statsnode = NULL;
	nextFunctionIndex = 0;
	numGLFunctionCalls = 0;
	functionDataArray.clear();
	functionIndexList.clear();
}


void 
DlgTrace::loadNewLogFile(string logfile, int fNumber, bool tellg, bool appendCount) {

	string line;
	isNewFrame = false;
	filestream.clear();
	filestream.open(logfile);
	
	//if (fNumber >= 0){
		frameNumber=fNumber;
	//}
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
						m_Frame = m_Log->AppendItem(m_Lognode, "Frame " + to_string(frameNumber) + " >");
						m_Statsnode = m_Log->AppendItem(m_Frame, "Statistics Log >");
						m_Frame = m_Log->AppendItem(m_Frame, "Call Log >");
						m_Pass = m_Frame;
						frameNumber++;
						isNewFrame = false;
					}
				}
			}
		}
		else{
			isNewFrame = true;
		}

		while (getline(filestream, line))
		{
			if (strcmp(line.substr(0, 4).c_str(), "#NAU") == 0) {
				if (strcmp(line.substr(5, 5).c_str(), "FRAME") == 0) {
					if (strcmp(line.substr(11, 5).c_str(), "START") == 0){
						if (m_Frame){
							m_Log->Expand(m_Frame);
						}
						PrintFunctionCount();
						ZeroStatsHeaders();
						frameStatNumber = frameNumber;
						m_Frame = m_Log->AppendItem(m_Lognode, "Frame " + to_string(frameNumber) + " >");
						m_Statsnode = m_Log->AppendItem(m_Frame, "Statistics Log >");
						m_Frame = m_Log->AppendItem(m_Frame, "Call Log >");
						m_Pass = m_Frame;
						frameNumber++;
						isNewFrame = false;
					}
				}
				else if (strcmp(line.substr(5, 4).c_str(), "PASS") == 0){
					if (strcmp(line.substr(10, 5).c_str(), "START") == 0){
						m_Pass = m_Log->AppendItem(m_Frame, "NAUPASS(" + line.substr(16, line.length() - 17) + ") >");
					}
					else if (strcmp(line.substr(10, 3).c_str(), "END") == 0){
						m_Pass = m_Frame;
					}
				}
			}
			else{
				if (isNewFrame){
					if (m_Frame){
						m_Log->Expand(m_Frame);
					}
					PrintFunctionCount();
					ZeroStatsHeaders();
					frameStatNumber = frameNumber;
					m_Frame = m_Log->AppendItem(m_Lognode, "Frame " + to_string(frameNumber) + " >");
					m_Statsnode = m_Log->AppendItem(m_Frame, "Statistics Log >");
					m_Frame = m_Log->AppendItem(m_Frame, "Call Log >");
					m_Pass = m_Frame;
					frameNumber++;
					isNewFrame = false;
				}
				m_Log->AppendItem(m_Pass, line);
				if (appendCount){
					CountFunction(split(line, '(')[0]);
				}
			}
			if (filestream.tellg() >= 0){
				streamlnnum = filestream.tellg();
			}
		}

	}
	filestream.close();
}


void 
DlgTrace::finishReadLogFile() {
	filestream.close();
}


void 
DlgTrace::loadLog() {

	string logfile, filename;
	struct stat fst;
	DIR *dir;
	struct dirent *ent;
//	time_t tempLastTime;
	std::map<int, std::string> filesToProcess;
	
	if (isLogClear) {
		m_Rootnode = m_Log->AddRoot(string("./__nau3Dtrace/") + string("Frame_*.txt"));
		m_Lognode = m_Log->AppendItem(m_Rootnode, "Frame Log >");
	}
	if ((dir = opendir("./__nau3Dtrace")) != NULL) {
		//m_FileTimes.clear();
		//tempLastTime = m_LastTime;
		//unsigned long long tempLastTime = m_LastTime;
		// Read all files and directories in the directory
		while ((ent = readdir(dir)) != NULL) {

			// Filters files starting with Frame_* only
			if (ent->d_type == S_IFREG && strstr(ent->d_name, "Frame_")) {
				// Corresponding logfile with path
				filename = std::string(ent->d_name);
				logfile = std::string("./__nau3Dtrace/") + filename;

				// if file is at least as recent as project loading
				if (stat(logfile.c_str(), &fst) == 0 && fst.st_mtime >= m_ProjectStartTime) {

					int frame = -1;
					if (filename.size() > 10)
						frame = stoi(filename.substr(6, 20), NULL, 10);
					
					// if file has not been processed yet
					if (0 == m_Filenames.count(frame)) {
						filesToProcess[frame] = logfile;
						m_Filenames[frame] = logfile;
					}
				}
			}
		}
		closedir(dir);

		for (auto f : filesToProcess)
			loadNewLogFile(f.second, f.first, false, true);

		m_Log->Expand(m_Rootnode);
		m_Log->Expand(m_Lognode);
		isLogClear = false;
	}

	PrintFunctionCount();
}


DlgTrace::FunctionCallData::FunctionCallData() :
	funcCallCount(0) {

}


bool
DlgTrace::FunctionCallData::SortByName(const FunctionCallData &a, const FunctionCallData &b) {

	return a.functionName < b.functionName;
}


bool 
DlgTrace::FunctionCallData::SortByCount(const FunctionCallData &a, const FunctionCallData &b) {

	return a.funcCallCount > b.funcCallCount;
}


void 
DlgTrace::CleanStatsHeaders(){
	m_Log->DeleteChildren(m_Statsnode);
	m_Statsnamenode = m_Log->AppendItem(m_Statsnode, "Ordered by Name >");
	m_Statscountnode = m_Log->AppendItem(m_Statsnode, "Ordered by Count >");
}


void 
DlgTrace::ZeroStatsHeaders(){
	functionDataArray.clear();
	numGLFunctionCalls = 0;
}


void 
DlgTrace::CountFunction(std::string funcName)
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
DlgTrace::PrintFunctionCount()
{ 
	if (m_Statsnode){
		CleanStatsHeaders();
		std::vector<FunctionCallData> functionDataArrayClone = functionDataArray;
		//Dump the total call count and average per frame (excluding first frame of xxx calls)
		m_Log->AppendItem(m_Statsnode, "Total GL Calls: " + std::to_string(numGLFunctionCalls));

		//Sort the array based on function call count
		sort(functionDataArrayClone.begin(), functionDataArrayClone.end(), FunctionCallData::SortByCount);


		//Loop and dump the function data 
		for (unsigned int i = 0; i < functionDataArrayClone.size(); i++)
		{
			//Only dump functions that have been called
			if (functionDataArrayClone[i].funcCallCount > 0)
			{
				m_Log->AppendItem(m_Statscountnode, functionDataArrayClone[i].functionName + ": " + std::to_string(functionDataArrayClone[i].funcCallCount));
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
				m_Log->AppendItem(m_Statsnamenode, functionDataArrayClone[i].functionName + ": " + std::to_string(functionDataArrayClone[i].funcCallCount));
			}
		}
	}
}

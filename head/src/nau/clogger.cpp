#include "nau/clogger.h"

#include <sys/timeb.h>
#include <ctime>

static CLogger *instance = 0;


CLogger::CLogger(void) {

}


CLogger::~CLogger(void) {

}


CLogger& 
CLogger::GetInstance() {

	if (0 == instance){
		instance = new CLogger;
	}
	return *instance;
}


void 
CLogger::addLog (LogLevel level, std::string file) {

	if (level != LEVEL_NONE)
		m_Logs[level] = CLogHandler (file);
}


bool
CLogger::hasLog(LogLevel ll) {

	return 0 != m_Logs.count(ll);
}


void 
CLogger::log (LogLevel logLevel, std::string sourceFile, int line, std::string message) {
	
	// log exists?
	if (m_Logs.count(logLevel) == 0)
		return;

	timeb time;
	std::string result;
	char tempBuffer[256];

	result += "[";
	result += logLevelNames[logLevel];
	result += "]";

	ftime(&time);
	struct tm*	tempTm = localtime(&time.time);
	strftime(tempBuffer, 255, "(%d/%m/%Y %H:%M:%S.", tempTm);
	
	result += tempBuffer;
	sprintf (tempBuffer,"%d", time.millitm);
	result += tempBuffer;
	result += ")";

	result += "[";
	result += sourceFile;
	result += "]";

	result += "(";
	sprintf (tempBuffer, "%d", line);
	result += tempBuffer;
	result += ")";

	result += ":";

	result += message;

	result += "\n";

	m_Logs[logLevel].log(result);
}


void 
CLogger::logSimple (LogLevel logLevel, std::string message) {
	
	// log exists?
	if (m_Logs.count(logLevel) == 0)
		return;

	std::string result = message;

	result += "\n";

	m_Logs[logLevel].log(result);
}


void 
CLogger::logSimpleNR (LogLevel logLevel, std::string message) {
	
	// log exists?
	if (m_Logs.count(logLevel) == 0)
		return;

	m_Logs[logLevel].log(message);
}


void 
CLogger::reset(LogLevel ll) {

	if (m_Logs.count(ll) == 0)
		return;

	m_Logs[ll].reset();
}


// ----------------------------------------------------------
//			CLogHandler
// ----------------------------------------------------------


CLogHandler::CLogHandler() {

	m_FileName = "";
}

CLogHandler::CLogHandler (std::string file) {

	m_FileName = file;
}


void 
CLogHandler::log(std::string& message) {

	FILE* fileHandler;

	fileHandler = m_FileName == "" ? stdout : 0;

	if (0 == fileHandler){
		
		fileHandler = fopen (m_FileName.c_str (), "a");
		if (0 == fileHandler){
			return;
		}
	} 

	fwrite (message.c_str(), message.size (), 1, fileHandler);
	if (fileHandler != stdout){
		fclose (fileHandler);
	}
}


void 
CLogHandler::reset() {

	FILE* fileHandler;

	fileHandler = m_FileName == "" ? stdout : 0;

	if (0 == fileHandler){
		
		fileHandler = fopen (m_FileName.c_str (), "w");
		if (!fileHandler)
			m_FileName = "";
		else
			fclose (fileHandler);
	} 
}

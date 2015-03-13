#include "nau/clogger.h"

// INFO: The commented out headers came from clogger.h and are
// probably useless I've just kept them here for safekeeping before we
// do a major sweep of the headers

//#include <stdlib.h>

#include <sys/timeb.h>
#include <ctime>

//#include <fcntl.h>
//#include <sys/stat.h>
//#include <sys/types.h>

//Hooray to windows and the standards!
//#ifdef NAU_PLATFORM_WIN32
//#include <io.h>
//#else
//#include <unistd.h>
//#endif


static CLogger *instance = 0;

CLogger::CLogger(void) :
	m_LogLevel (LEVEL_CRITICAL)
{
}

CLogger::~CLogger(void)
{
	for (unsigned int i = 0; i < m_Logs.size (); i++){
		delete m_Logs.at (i);
	}
}

CLogger& 
CLogger::getInstance()
{
	if (0 == instance){
		instance = new CLogger;
	}
	return *instance;
}

void 
CLogger::addLog (LogLevel level, std::string file)
{
	CLogHandler *aLog = new CLogHandler (level, file);

	m_Logs.push_back (aLog);
}

void 
CLogger::log (LogLevel logLevel, std::string sourceFile, int line, std::string message)
{
	
	if (m_LogLevel == LEVEL_NONE || m_LogLevel < logLevel){
		return;
	}
	
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

	for (unsigned int i = 0; i < m_Logs.size (); i++){
		if (logLevel <= m_Logs.at(i)->getLogLevel()){
			m_Logs.at(i)->log (result);
		}
	}
}

void 
CLogger::setLogLevel (LogLevel level)
{
	this->m_LogLevel = level;
}

LogLevel 
CLogger::getLogLevel ()
{
	return m_LogLevel;
}

CLogHandler::CLogHandler (LogLevel level, std::string file)
{
	m_LogLevel = level;
	m_FileName = file;
}

void 
CLogHandler::log(std::string& message)
{
	//int fileHandler;

	//fileHandler = m_FileName == "" ? 2 : -1;

	FILE* fileHandler;

	fileHandler = m_FileName == "" ? stdout : 0;

	if (0 == fileHandler){
		
/*		fileHandler = open(m_FileName.c_str (),
				O_APPEND );
		if (fileHandler < 0){
			fileHandler = open(m_FileName.c_str (), O_CREAT | O_RDWR);
			if (fileHandler < 0){
				return;
			}
		}
*/
		fileHandler = fopen (m_FileName.c_str (), "a");
		if (0 == fileHandler){
			return;
		}
	} else {
		//PHONY
	}

	fwrite (message.c_str(), message.size (), 1, fileHandler);
	//write (fileHandler, message.c_str (), message.size ());
	if (fileHandler != stdout){
		fclose (fileHandler);
	}
}

LogLevel 
CLogHandler::getLogLevel ()
{
	return m_LogLevel;
}

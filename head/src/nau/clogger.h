#ifndef CLOGGER_H
#define CLOGGER_H

#include "nau/config.h"

#include <stdio.h>
#include <string>
#include <vector>

typedef enum {
	LEVEL_NONE,
	LEVEL_CONFIG,
	LEVEL_INFO,
	LEVEL_WARN,
	LEVEL_ERROR,
	LEVEL_CRITICAL,
	LEVEL_ALL
} LogLevel;

static const char logLevelNames[][9] = {
 	"NONE", "CONFIG", "INFO", "WARN", "ERROR", "CRITICAL", "ALL"
};

class CLogHandler
{
	private:
		std::string m_FileName;
		LogLevel m_LogLevel;

	public:
		CLogHandler (LogLevel level, std::string file);
		void log(std::string& message);
		LogLevel getLogLevel ();
};

class CLogger
{
	private:
		std::vector <CLogHandler*> m_Logs;
		LogLevel m_LogLevel;

	public:
		static CLogger& getInstance();
		void addLog (LogLevel level, std::string file = "");
		void log (LogLevel logLevel, std::string sourceFile, int line, std::string message);
		void setLogLevel (LogLevel level);
		LogLevel getLogLevel ();

	private:
		CLogger(void);
	public:
		~CLogger(void);
};


#if NAU_DEBUG == 1
#define DEBUG_INFO(message, ...) \
{\
  char m[256];\
  sprintf(m, message, ## __VA_ARGS__);\
  (CLogger::getInstance()).log(LEVEL_INFO,__FILE__,__LINE__,m);\
};
#else
#define DEBUG_INFO
#endif

#define LOG_CONFIG(message, ...) \
{\
	char m[256];\
	sprintf(m, message, ## __VA_ARGS__);\
	(CLogger::getInstance()).log(LEVEL_CONFIG,__FILE__,__LINE__,m);\
};

#define LOG_INFO(message, ...) \
{\
  char m[256];\
  sprintf(m, message, ## __VA_ARGS__);\
  (CLogger::getInstance()).log(LEVEL_INFO,__FILE__,__LINE__,m);\
};

#define LOG_WARN(message, ...) \
{\
  char m[256];\
  sprintf(m, message, ## __VA_ARGS__);\
  (CLogger::getInstance()).log(LEVEL_WARN,__FILE__,__LINE__,m);\
};

#define LOG_ERROR(message, ...) \
{\
  char m[256];\
  sprintf(m, message, ## __VA_ARGS__);\
  (CLogger::getInstance()).log(LEVEL_ERROR,__FILE__,__LINE__,m);\
};

#define LOG_CRITICAL(message, ...) \
{\
  char m[256];\
  sprintf(m, message, ## __VA_ARGS__);\
  (CLogger::getInstance()).log(LEVEL_CRITICAL,__FILE__,__LINE__,m);\
};

#endif // CLOGGER_H

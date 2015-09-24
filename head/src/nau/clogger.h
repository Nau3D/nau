#ifndef CLOGGER_H
#define CLOGGER_H

#include "nau/config.h"

#include <stdio.h>
#include <string>
#include <map>


static const char logLevelNames[][9] = {
	"CONFIG", "INFO", "WARN", "ERROR", "CRITICAL", "ALL", "TRACE"
};

class CLogHandler
{
	private:
		std::string m_FileName;

	public:
		CLogHandler();
		CLogHandler (std::string file);
		void log(std::string& message);
		void reset();
};

class CLogger
{

	public:
		typedef enum {
			LEVEL_NONE,
			LEVEL_CONFIG,
			LEVEL_INFO,
			LEVEL_WARN,
			LEVEL_ERROR,
			LEVEL_CRITICAL,
			LEVEL_TRACE
		} LogLevel;
		static CLogger& GetInstance();
		void addLog (LogLevel level, std::string file = "");
		void log (LogLevel logLevel, std::string sourceFile, int line, std::string message);
		void logSimple (LogLevel logLevel, std::string message);
		void logSimpleNR (LogLevel logLevel, std::string message);
		bool hasLog(LogLevel logLevel);
		void reset(LogLevel logLevel);

		~CLogger(void);

	private:
		CLogger(void);
		std::map <LogLevel, CLogHandler> m_Logs;
};


#if NAU_DEBUG == 1
#define LOG_DEBUG(message, ...) \
{\
  char m[256];\
  sprintf(m, message, ## __VA_ARGS__);\
  (CLogger::GetInstance()).log(LEVEL_INFO,__FILE__,__LINE__,m);\
};
#else
#define LOG_DEBUG
#endif


#define LOG_TRACE(message, ...) \
{\
	char m[256];\
	sprintf(m, message, ## __VA_ARGS__);\
	(CLogger::GetInstance()).log(CLogger::LEVEL_TRACE,__FILE__,__LINE__,m);\
};


#define LOG_trace(message, ...) \
{\
	char m[256];\
	sprintf(m, message, ## __VA_ARGS__);\
	(CLogger::GetInstance()).logSimple(CLogger::LEVEL_TRACE,m);\
};


#define LOG_trace_nr(message, ...) \
{\
	char m[256];\
	sprintf(m, message, ## __VA_ARGS__);\
	(CLogger::GetInstance()).logSimpleNR(CLogger::LEVEL_TRACE,m);\
};


#define LOG_CONFIG(message, ...) \
{\
	char m[256];\
	sprintf(m, message, ## __VA_ARGS__);\
	(CLogger::GetInstance()).log(CLogger::LEVEL_CONFIG,__FILE__,__LINE__,m);\
};


#define LOG_INFO(message, ...) \
{\
  char m[256];\
  sprintf(m, message, ## __VA_ARGS__);\
  (CLogger::GetInstance()).log(CLogger::LEVEL_INFO,__FILE__,__LINE__,m);\
};


#define LOG_WARN(message, ...) \
{\
  char m[256];\
  sprintf(m, message, ## __VA_ARGS__);\
  (CLogger::GetInstance()).log(CLogger::LEVEL_WARN,__FILE__,__LINE__,m);\
};


#define LOG_ERROR(message, ...) \
{\
  char m[256];\
  sprintf(m, message, ## __VA_ARGS__);\
  (CLogger::GetInstance()).log(CLogger::LEVEL_ERROR,__FILE__,__LINE__,m);\
};


#define LOG_CRITICAL(message, ...) \
{\
  char m[256];\
  sprintf(m, message, ## __VA_ARGS__);\
  (CLogger::GetInstance()).log(CLogger::LEVEL_CRITICAL,__FILE__,__LINE__,m);\
};


#endif // CLOGGER_H

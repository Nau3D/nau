#ifndef CLOGGER_H
#define CLOGGER_H

#include "nau/config.h"

#include <stdio.h>
#include <string>
#include <vector>
#include <map>




class CLogHandler
{
	private:
		std::string m_Filename;
		FILE *m_FileHandler;

	public:
		CLogHandler();
		~CLogHandler();
		CLogHandler (std::string file);
		void log(std::string& message);
		void setFile(std::string &filename);
		void close();
		//void reset();
};

class CLogger
{
	public:
		typedef enum {
			LEVEL_CONFIG,
			LEVEL_INFO,
			LEVEL_WARN,
			LEVEL_ERROR,
			LEVEL_CRITICAL,
			LEVEL_TRACE
		} LogLevel;

		static void AddLog (LogLevel level, std::string file = "");
		static void Log (LogLevel logLevel, std::string sourceFile, int line, std::string message);
		static void LogSimple (LogLevel logLevel, std::string message);
		static void LogSimpleNR (LogLevel logLevel, std::string message);
		static bool HasLog(LogLevel logLevel);
		static void CloseLog(LogLevel level);
		//static void Reset(LogLevel logLevel);

		~CLogger(void);

	private:
		CLogger(void);
		static std::map <LogLevel, CLogHandler> Logs;
		static const std::vector<std::string> LogNames;
};


#if NAU_DEBUG == 1
#define LOG_DEBUG(message, ...) \
{\
  char m[256];\
  snprintf(m, 256, message, ## __VA_ARGS__);\
  CLogger::Log(LEVEL_INFO,__FILE__,__LINE__,m);\
};
#else
#define LOG_DEBUG
#endif


#define LOG_TRACE(message, ...) \
{\
	char m[256];\
	snprintf(m, 256, message, ## __VA_ARGS__);\
	CLogger::Log(CLogger::LEVEL_TRACE,__FILE__,__LINE__,m);\
};


#define LOG_trace(message, ...) \
{\
	char m[256];\
	snprintf(m, 256, message, ## __VA_ARGS__);\
	CLogger::LogSimple(CLogger::LEVEL_TRACE,m);\
};

// log without a return at the end
#define LOG_trace_nr(message, ...) \
{\
	char m[256];\
	snprintf(m, 256, message, ## __VA_ARGS__);\
	CLogger::LogSimpleNR(CLogger::LEVEL_TRACE,m);\
};


#define LOG_CONFIG(message, ...) \
{\
	char m[256];\
	snprintf(m, 256, message, ## __VA_ARGS__);\
	CLogger::Log(CLogger::LEVEL_CONFIG,__FILE__,__LINE__,m);\
};


#define LOG_INFO(message, ...) \
{\
  char m[256];\
  snprintf(m, 256, message, ## __VA_ARGS__);\
  CLogger::Log(CLogger::LEVEL_INFO,__FILE__,__LINE__,m);\
};


#define LOG_WARN(message, ...) \
{\
  char m[256];\
  snprintf(m, 256,message, ## __VA_ARGS__);\
  CLogger::Log(CLogger::LEVEL_WARN,__FILE__,__LINE__,m);\
};


#define LOG_ERROR(message, ...) \
{\
  char m[256];\
  snprintf(m, 256, message, ## __VA_ARGS__);\
  CLogger::Log(CLogger::LEVEL_ERROR,__FILE__,__LINE__,m);\
};


#define LOG_CRITICAL(message, ...) \
{\
  char m[256];\
  snprintf(m, 256, message, ## __VA_ARGS__);\
  CLogger::Log(CLogger::LEVEL_CRITICAL,__FILE__,__LINE__,m);\
};


#endif // CLOGGER_H

#ifndef SLOGGER_H
#define SLOGGER_H

#include <stdio.h>
#include "nau/event/eventString.h"
#include <string>

#ifdef _WINDLL
#ifdef nau_EXPORTS
#define nau_API __declspec(dllexport)   
#else  
#define nau_API __declspec(dllimport)   
#endif 
#else
#define nau_API
#endif

class SLogger
{
private:
	nau::event_::EventString *m_Evt;
	static SLogger *Instance;

public:
	static nau_API SLogger* GetInstance();
	static void DeleteInstance();
	nau_API void log (std::string message);

private:
	SLogger(void);
public:
	~SLogger(void);
};



#define SLOG(message, ...) \
{\
  char m[256];\
  snprintf(m, 256, message, ## __VA_ARGS__);\
  (SLogger::GetInstance())->log(m);\
};

#define SLOG_INFO(message, ...) \
{\
  char m[256];\
  snprintf(m, 256, message, ## __VA_ARGS__);\
  (SLogger::GetInstance())->log(m);\
};

#endif // SLOGGER_H

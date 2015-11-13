#ifndef SLOGGER_H
#define SLOGGER_H

#include <stdio.h>
#include "nau/event/eventString.h"
#include <string>



class SLogger
{
private:
	nau::event_::EventString *m_Evt;
	static SLogger *Instance;

public:
	static SLogger* GetInstance();
	static void DeleteInstance();
	void log (std::string message);

private:
	SLogger(void);
public:
	~SLogger(void);
};



#define SLOG(message, ...) \
{\
  char m[32768];\
  sprintf(m, message, ## __VA_ARGS__);\
  (SLogger::GetInstance())->log(m);\
};

#define SLOG_INFO(message, ...) \
{\
  char m[32768];\
  sprintf(m, message, ## __VA_ARGS__);\
  (SLogger::GetInstance())->log(m);\
};

#endif // SLOGGER_H

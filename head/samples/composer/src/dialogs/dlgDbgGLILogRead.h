#ifndef __DBGGLILOGREAD_DIALOG__
#define __DBGGLILOGREAD_DIALOG__


#ifdef __GNUG__
#pragma implementation
#pragma interface
#endif


// For compilers that support precompilation, includes "wx/wx.h".
#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif

// For compilers that support precompilation, includes "wx/wx.h".
#include "wx/wxprec.h"
#include <wx/string.h>
#include <wx/treectrl.h>
#include <string>
#include <nau/event/ilistener.h>
#include <fstream> 

class DlgDbgGLILogRead : public wxDialog, nau::event_::IListener
{


protected:
	DlgDbgGLILogRead();
	DlgDbgGLILogRead(const DlgDbgGLILogRead&);
	DlgDbgGLILogRead& operator= (const DlgDbgGLILogRead&);
	static DlgDbgGLILogRead *m_Inst;

	void loadNewLogFile(std::string logfile, int fNumber);
	void continueReadLogFile();
	void finishReadLogFile();
	
	wxTreeCtrl *m_log;
	wxButton *m_bClear, *m_bProfiler, *m_bSave;
	wxTreeItemId rootnode;
	std::string name;
	bool isLogClear;
	bool isNewFrame;
	int frameNumber;
	std::ifstream filestream;



public:
	static wxWindow *m_Parent; 

	static DlgDbgGLILogRead* Instance () ;
	static void SetParent(wxWindow *parent);
	virtual std::string &getName ();
	void eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt);
	void updateDlg();
	void append(std::string s);
	void clear();
	void loadLog();


    DECLARE_EVENT_TABLE();

};

#endif


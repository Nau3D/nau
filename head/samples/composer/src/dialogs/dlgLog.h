#ifndef __LOG_DIALOG__
#define __LOG_DIALOG__


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
#include <string>
#include <nau/event/ilistener.h>


class DlgLog : public wxDialog, nau::event_::IListener
{


protected:
	DlgLog();
	DlgLog(const DlgLog&);
	DlgLog& operator= (const DlgLog&);
	static DlgLog *m_Inst;

	wxListBox *m_log;
	wxButton *m_bClear, *m_bProfiler, *m_bSave;
	std::string name;



public:

	~DlgLog();
	static wxWindow *m_Parent; 

	static DlgLog* Instance () ;
	static void SetParent(wxWindow *parent);
	virtual std::string &getName ();
	void eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt);
	void updateDlg();
	void append(std::string s);
	void clear();
	void OnClearLog(wxCommandEvent& event);
	void OnSaveLog(wxCommandEvent& event);
	void OnProfilerLog(wxCommandEvent& event);
	enum {DLG_BTN_CLEARLOG, DLG_BTN_SAVELOG, DLG_BTN_PROFILERLOG};


    DECLARE_EVENT_TABLE()

};

#endif


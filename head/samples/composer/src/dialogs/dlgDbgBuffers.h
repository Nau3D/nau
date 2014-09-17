#ifndef __DBGBUFFERS_DIALOG__
#define __DBGBUFFERS_DIALOG__


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
#include <iostream>
#include <sstream>
#include <nau/event/ilistener.h>


class DlgDbgBuffers : public wxDialog, nau::event_::IListener
{


protected:
	DlgDbgBuffers();
	DlgDbgBuffers(const DlgDbgBuffers&);
	DlgDbgBuffers& operator= (const DlgDbgBuffers&);
	static DlgDbgBuffers *m_Inst;

	
	wxTreeCtrl *m_log;
	wxButton *m_bClear, *m_bProfiler, *m_bSave;
	std::string name;
	bool isLogClear;

public:
	static wxWindow *m_Parent; 

	static DlgDbgBuffers* Instance () ;
	static void SetParent(wxWindow *parent);
	virtual std::string &getName ();
	void eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt);
	void updateDlg();
	void append(std::string s);
	void clear();
	void loadBufferInfo();
	void OnSaveInfo(wxCommandEvent& event);
	void OnSaveInfoAux(std::fstream &s, wxTreeItemId parent, int nodelevel);
	enum { DLG_BTN_SAVELOG};


    DECLARE_EVENT_TABLE();

};

#endif


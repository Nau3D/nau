#ifndef __DBGSTEP_DIALOG__
#define __DBGSTEP_DIALOG__


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
#include <wx/combobox.h>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <nau/event/ilistener.h>
#include "../glcanvas.h"


class DlgDbgStep : public wxDialog, nau::event_::IListener
{


protected:
	DlgDbgStep();
	DlgDbgStep(const DlgDbgStep&);
	DlgDbgStep& operator= (const DlgDbgStep&);
	static DlgDbgStep *m_Inst;

	//wxTextCtrl  *pipename;
	wxListBox *m_list;
	wxButton *m_bNextPass;
	wxButton *m_bNextFrame;
	wxButton *m_bNextToPass;
	std::string name;
	std::vector<std::string> *passes;
	std::string pipenameString;
	int currentPassIndex;

	void getPasses(std::string pipenameString);

public:
	static wxWindow *m_Parent;
	static GLCanvas *m_Canvas;

	static DlgDbgStep* Instance();
	static void SetParent(wxWindow *parent);
	static void SetCanvas(GLCanvas *c);
	virtual std::string &getName ();
	void eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt);
	void updateDlg();
	void append(std::string s);
	void clear();

	enum { DLG_TXT_PIPELINE, DLG_BTN_NEXTPASS, DLG_BTN_NEXTFRAME, DLG_BTN_TOPASS };


	void OnNextPass(wxCommandEvent& event);
	void OnNextFrame(wxCommandEvent& event);
	void OnToPass(wxCommandEvent& event);

    DECLARE_EVENT_TABLE();

};

#endif


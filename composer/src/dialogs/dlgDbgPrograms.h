#ifndef __DBGPROGRAMS_DIALOG__
#define __DBGPROGRAMS_DIALOG__


#ifdef __GNUG__
#pragma implementation
#pragma interface
#endif


#ifdef __BORLANDC__
#pragma hdrstop
#endif

#include <nau.h>
#include <nau/event/iListener.h>
#include <nau/util/tree.h>

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


class DlgDbgPrograms : public wxDialog, nau::event_::IListener
{


protected:
	DlgDbgPrograms();
	DlgDbgPrograms(const DlgDbgPrograms&);
	DlgDbgPrograms& operator= (const DlgDbgPrograms&);
	static DlgDbgPrograms *m_Inst;

	
	wxTreeCtrl *m_Log;
	wxButton *m_bSave, *m_bRefresh;
	std::string m_Name;
	bool m_IsLogClear;
	wxTreeItemId m_Rootnode;

	void updateDlgTree();
	void updateTree(nau::util::Tree *t, wxTreeItemId);

public:
	static wxWindow *m_Parent; 

	static DlgDbgPrograms* Instance () ;
	static void SetParent(wxWindow *parent);
	virtual std::string &getName ();
	void eventReceived(const std::string &sender, const std::string &eventType, 
		const std::shared_ptr<nau::event_::IEventData> &evt);
	void updateDlg();
	void clear();
	void loadShaderInfo();

	void OnRefreshLog(wxCommandEvent& event);
	void OnSaveInfo(wxCommandEvent& event);
	void OnSaveInfoAux(std::fstream &s, wxTreeItemId parent, int nodelevel);

	enum { DLG_BTN_SAVELOG, DLG_BTN_REFRESH};


    DECLARE_EVENT_TABLE();

};

#endif


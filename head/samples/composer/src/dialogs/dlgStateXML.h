#ifndef __STATEXML_DIALOG__
#define __STATEXML_DIALOG__


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
#include <wx/propgrid/propgrid.h>
#include <wx/propgrid/advprops.h>
#include <wx/propgrid/manager.h>
#include <string>
#include <iostream>
#include <sstream>
#include <nau/event/ilistener.h>


class DlgStateXML : public wxDialog, nau::event_::IListener
{


protected:
	DlgStateXML();
	DlgStateXML(const DlgStateXML&);
	DlgStateXML& operator= (const DlgStateXML&);
	static DlgStateXML *m_Inst;

	
	wxPropertyGridManager *m_log;
	wxButton *m_bSave, *m_bLoad;
	std::string name;

public:
	static wxWindow *m_Parent; 

	static DlgStateXML* Instance () ;
	static void SetParent(wxWindow *parent);
	virtual std::string &getName ();
	void eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt);
	void updateDlg();
	void append(std::string s);
	void loadShaderInfo();
	void OnSaveInfo(wxCommandEvent& event);
	void OnSavePropertyGridAux(std::fstream &s, wxPropertyGridPage *page);
	void OnLoadXML(wxCommandEvent& event);
	//void startRecording();
	enum { DLG_BTN_SAVELOG, DLG_BTN_LOADXML, DLG_MI_LOG };


    DECLARE_EVENT_TABLE();

};

#endif


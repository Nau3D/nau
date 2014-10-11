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
//#include <wx/treectrl.h>
#include <wx/notebook.h>
#include <wx/combobox.h>
#include <wx/grid.h>
#include <wx/propgrid/propgrid.h>
#include <wx/propgrid/advprops.h>
#include <wx/propgrid/manager.h>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <nau/event/ilistener.h>


class DlgDbgBuffers : public wxDialog, nau::event_::IListener
{
private:

	enum DataTypes {
		DLG_BYTE, DLG_UNSIGNED_BYTE, DLG_INT, DLG_UNSIGNED_INT, DLG_SHORT, DLG_UNSIGNED_SHORT, DLG_FLOAT, DLG_DOUBLE
	};

	void loadBufferInfoGrid();

	void OnBufferSettingsChange(wxPropertyGridEvent& e);
	void updateBufferData(wxPGProperty *bufferProperty, int currentTypeCount = -1);
	DataTypes getBufferDataType(wxPGProperty *typeProperty, int index);
	int getBufferDataTypeSize(DataTypes type);
	void updateBufferDataValues(wxPGProperty *valuesProperty, std::vector<void*> &pointers, std::vector<DataTypes> types);
	std::string getStringFromPointer(DataTypes type, void* ptr);
protected:
	DlgDbgBuffers();
	DlgDbgBuffers(const DlgDbgBuffers&);
	DlgDbgBuffers& operator= (const DlgDbgBuffers&);
	static DlgDbgBuffers *m_Inst;

	wxPropertyGridManager *pgBuffers;
	wxPropertyGridManager *pgVAOs;
	
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
	enum { 
		DLG_MI_PGBUFFERS,
		DLG_MI_PGVAOS
	};


    DECLARE_EVENT_TABLE();

};

#endif


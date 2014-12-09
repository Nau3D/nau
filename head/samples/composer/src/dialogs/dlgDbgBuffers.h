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
#include <wx/spinctrl.h>
#include <wx/combobox.h>
#include <wx/splitter.h>
#include <wx/grid.h>
#include <wx/propgrid/propgrid.h>
#include <wx/propgrid/advprops.h>
#include <wx/propgrid/manager.h>
#include <string>
#include <vector>
#include <map>
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


	void OnBufferSelection(wxPropertyGridEvent& e);
	void OnBufferChanged(wxPropertyGridEvent& e);
	void loadBufferSettings();
	void loadBufferSettingsPGUpdate(wxPGProperty *settings, int buffer);


	void OnBufferValuesLengthChange(wxSpinEvent& e);
	void OnBufferValuesLinesChange(wxSpinEvent& e);
	void OnBufferValuesPageChange(wxSpinEvent& e);

	void OnBufferSettingsChange();
	void updateBufferData();
	DataTypes getBufferDataType(wxPGProperty *typeProperty, int index);
	DataTypes getDLGDataType(int type);
	DataTypes getDLGDataType(std::string type);
	int getBufferDataTypeSize(DataTypes type);
	void updateBufferDataValues(std::vector<void*> &pointers);
	std::string getStringFromPointer(DataTypes type, void* ptr);

	void OnGridCellChange(wxGridEvent& event);

	struct BufferSettings
	{
		BufferSettings();

		unsigned int length; //Number of elements per line
		unsigned int lines;  //Number of lines per page
		unsigned int currentPage;
		std::string bufferName;
		std::vector<DataTypes> types; // Only for non VAOs
	};


	static std::map<int, BufferSettings> bufferSettingsList;
	int currentBufferIndex;
protected:
	DlgDbgBuffers();
	DlgDbgBuffers(const DlgDbgBuffers&);
	DlgDbgBuffers& operator= (const DlgDbgBuffers&);
	static DlgDbgBuffers *m_Inst;

	wxPropertyGridManager *pgBuffers;
	wxPropertyGridManager *pgVAOs;
	wxGrid *gridBufferValues;
	std::vector<wxGridCellChoiceEditor *> gridBufferValuesHeaders;

	wxSpinCtrl *spinBufferLength, *spinBufferLines;
	wxSpinCtrl *spinBufferPage;

	wxButton *m_bSavevaos;
	wxButton *m_bSavebuffers;
	wxButton *m_bSavepage;
	
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
	void clear(bool fullclear = false);
	void loadBufferInfo();
	enum { 
		DLG_MI_PGBUFFERS,
		DLG_MI_PGVAOS,
		DLG_MI_GRIDBUFFERINFO,
		DLG_MI_GRIDBUFFERINFOLENGTH,
		DLG_MI_GRIDBUFFERINFOLINES,
		DLG_MI_GRIDBUFFERINFOPAGE,
		DLG_MI_SAVEBUFFER,
		DLG_MI_SAVEVALUEPAGE,
		DLG_MI_SAVEVAO
	};

	void OnSaveBufferInfo(wxCommandEvent& event);
	void OnSavePageInfo(wxCommandEvent& event);
	void OnSavePropertyGridAux(std::fstream &s, wxPropertyGridPage *page);
	void OnSaveVaoInfo(wxCommandEvent& event);

    DECLARE_EVENT_TABLE();

};

#endif


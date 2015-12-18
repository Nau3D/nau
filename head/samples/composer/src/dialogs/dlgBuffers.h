#ifndef __DBGBUFFERS_DIALOG__
#define __DBGBUFFERS_DIALOG__


#ifdef __GNUG__
#pragma implementation
#pragma interface
#endif

#include <nau/event/iListener.h>
#include <nau/material/iBuffer.h>
#include <nau/enums.h>

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
#include <wx/string.h>
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



class DlgDbgBuffers : public wxDialog, nau::event_::IListener
{
protected:
	const int MAX_COLUMNS = 32;
	const int MAX_ROWS = 1024;
	const std::string NO_BUFFER = "__No Buffer";

	enum {
		DLG_MI_PGBUFFERS,
		DLG_MI_PGVAOS,
		DLG_MI_GRIDBUFFERINFO,
		DLG_MI_GRIDBUFFERINFOLENGTH,
		DLG_MI_GRIDBUFFERINFOLINES,
		DLG_MI_GRIDBUFFERINFOPAGE,
		DLG_MI_REFRESH,
		DLG_MI_REFRESH_BUFFER_DATA,
		DLG_MI_SAVEVALUEPAGE,
		DLG_MI_SAVEVAO,
		DLG_MI_UPDATE_BUFFER
	};

	typedef 	// ((VAO index, (Element index, Element name)) , vector(Array index, Array Name))
		std::pair<std::pair<int, std::pair<int, std::string>>, std::vector<std::pair<int, std::string>>> VAOInfo;
	typedef std::vector<VAOInfo> VAOInfoList;

	bool m_UseShortNames = true;

	void setVAOList();
	void setBufferList();
	void setBufferData();
	void setSpinners(int lines, int columns, int page, int lineSize, int size);
	void setBufferProperties();

	void OnGridCellChange(wxGridEvent& e);
	void OnBufferValuesLengthChange(wxSpinEvent& e);
	void OnBufferValuesLinesChange(wxSpinEvent& e);
	void OnBufferValuesPageChange(wxSpinEvent& e);
	void OnBufferSelection(wxPropertyGridEvent& e);

	void OnRefreshBufferInfo(wxCommandEvent& event);

	void OnRefreshBufferData(wxCommandEvent& event);
	
	void OnUpdateBuffer(wxCommandEvent& event);
	//void OnRefresh(std::fstream &s, wxPropertyGridPage *page);
	//void OnSaveVaoInfo(wxCommandEvent& event);

	std::string getStringFromPointer(Enums::DataType type, void* ptr);
	void insertIntoBuffer(std::string elem, Enums::DataType type, void *ptr);

	void clear();

	struct BufferSettings
	{
		BufferSettings();

		unsigned int lines;  //Number of lines per page
		unsigned int currentPage;
		int ID;
		int size;
		std::vector<Enums::DataType> types;
		int lineSize;
		nau::material::IBuffer *bufferPtr;
		std::string shortName;
		std::string fullName;
	};


	std::map<std::string, BufferSettings> bufferSettingsList;
	std::string currentBuffer;


	static std::map<Enums::DataType, std::string> DataType;

	DlgDbgBuffers();
	DlgDbgBuffers(const DlgDbgBuffers&);
	DlgDbgBuffers& operator= (const DlgDbgBuffers&);
	static DlgDbgBuffers *Inst;

	wxPropertyGridManager *pgBuffers;
	wxPropertyGridManager *pgVAOs;
	wxGrid *gridBufferValues;
	std::vector<wxGridCellChoiceEditor *> gridBufferValuesHeaders;

	wxSpinCtrl *spinBufferLength, *spinBufferLines;
	wxSpinCtrl *spinBufferPage;

	wxButton *m_bRefresh;
	wxButton *m_bRefreshBufferData;
	wxButton *m_bUpdateBuffer;
	wxButton *m_bSavevaos;
	wxButton *m_bSavebuffers;
	
	std::string name;
	bool isLogClear;



public:
	static wxWindow *Parent; 

	static DlgDbgBuffers* Instance () ;
	static void SetParent(wxWindow *parent);
	virtual std::string &getName ();
	void eventReceived(const std::string &sender, const std::string &eventType, 
		const std::shared_ptr<nau::event_::IEventData> &evt);
	void updateDlg(bool shortNames = true);

    DECLARE_EVENT_TABLE();

};

#endif


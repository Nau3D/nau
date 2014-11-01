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
#include <vector>
#include <map>
#include <nau/event/ilistener.h>
#include <fstream> 

class DlgDbgGLILogRead : public wxDialog, nau::event_::IListener
{

private:
	//Copied from GLIPlugins funcstats plugin
	struct FunctionCallData
	{
		FunctionCallData();

		//@
		//  Summary:
		//    Sorting functions to sort and array of elements
		//
		static inline bool SortByName(const FunctionCallData &a, const FunctionCallData &b);
		static inline bool SortByCount(const FunctionCallData &a, const FunctionCallData &b);

		std::string functionName;                          // The name of the function
		unsigned int   funcCallCount;                         // The call count of the function
	};

	std::vector<FunctionCallData>  functionDataArray;    // Array of function data
	std::map<std::string, unsigned int> functionIndexList;

	unsigned int nextFunctionIndex;
	unsigned int frameStatNumber;
	unsigned int numGLFunctionCalls;

	void CleanStatsHeaders();
	void ZeroStatsHeaders();
	void CountFunction(std::string funcName);
	void PrintFunctionCount();

protected:
	DlgDbgGLILogRead();
	DlgDbgGLILogRead(const DlgDbgGLILogRead&);
	DlgDbgGLILogRead& operator= (const DlgDbgGLILogRead&);
	static DlgDbgGLILogRead *m_Inst;

	void loadNewLogFile(std::string logfile, int fNumber, bool tellg = false, bool appendCount = false);
	void finishReadLogFile();
	
	wxTreeCtrl *m_log;
	wxButton *m_bClear, *m_bProfiler, *m_bSave;
	wxTreeItemId rootnode, lognode, statsnode, statsnamenode, statscountnode;
	wxTreeItemId frame, pass;
	std::string name;
	bool isLogClear;
	bool isNewFrame;
	int frameNumber;
	std::ifstream filestream;
	int streamlnnum;



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


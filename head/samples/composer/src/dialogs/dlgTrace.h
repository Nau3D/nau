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
#include <nau/event/iListener.h>
#include <fstream> 

class DlgTrace : public wxDialog, nau::event_::IListener
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

	unsigned long long m_LastTime;
	std::map<unsigned long long, std::pair<std::string, int>> m_FileTimes;


	DlgTrace();
	DlgTrace(const DlgTrace&);
	DlgTrace& operator= (const DlgTrace&);
	static DlgTrace *m_Inst;

	void loadNewLogFile(std::string logfile, int fNumber, bool tellg = false, bool appendCount = false);
	void finishReadLogFile();
	
	wxTreeCtrl *m_Log;
	wxButton *m_bClear, *m_bProfiler, *m_bSave;
	wxTreeItemId m_Rootnode, m_Lognode, m_Statsnode, m_Statsnamenode, m_Statscountnode;
	wxTreeItemId m_Frame, m_Pass;
	std::string name;
	bool isLogClear;
	bool isNewFrame;
	int frameNumber;
	std::ifstream filestream;
	int streamlnnum;



public:
	static wxWindow *m_Parent; 

	static DlgTrace* Instance () ;
	static void SetParent(wxWindow *parent);
	virtual std::string &getName ();
	void updateDlg();
	void append(std::string s);
	void clear();
	void loadLog();

	void eventReceived(const std::string &sender, const std::string &eventType, 
		const std::shared_ptr<nau::event_::IEventData> &evt);

    DECLARE_EVENT_TABLE();

};

#endif


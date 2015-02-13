#ifndef __SLANGER_DIALOGS_VIEWPORTS__
#define __SLANGER_DIALOGS_VIEWPORTS__


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

#include <wx/string.h>
#include <wx/grid.h>
#include <wx/propgrid/propgrid.h>
#include <wx/propgrid/advprops.h>
#include <wx/propgrid/manager.h>

#include <nau/event/ilistener.h>

class DlgViewports : public wxDialog, nau::event_::IListener
{
public:
	void updateDlg();
	static DlgViewports* Instance();
	static void SetParent(wxWindow *parent);
	static wxWindow *Parent;

	void updateInfo(std::string name);
	void eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt);
	std::string &getName();

protected:

	DlgViewports();
	DlgViewports(const DlgViewports&);
	DlgViewports& operator= (const DlgViewports&);
	static DlgViewports *Inst;

	/* GLOBAL STUFF */
	// active viewport
	std::string m_Active;
	// the class name
	std::string m_Name;

	/* VIEWPORTS */
	wxButton *m_BAdd;
	wxPropertyGridManager *m_PG;
	wxComboBox *m_List;

	/* EVENTS */
	void OnListSelect(wxCommandEvent& event);
	void OnAdd(wxCommandEvent& event);
	void OnPropsChange( wxPropertyGridEvent& e);

	void update();
	void updateList();
	void setupPanel(wxSizer *siz, wxWindow *parent);
	void setupGrid();

	enum {
		DLG_COMBO,
		DLG_BUTTON_ADD,
		DLG_PROPS

	};

	typedef enum {
		NEW_VIEWPORT,
		PROPS_CHANGED
	} Notification;

	void notifyUpdate(Notification aNot, std::string vpName, std::string value);

    DECLARE_EVENT_TABLE()
};


#endif



#ifndef __COMPOSER_DIALOGS_CAMERAS__
#define __COMPOSER_DIALOGS_CAMERAS__


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

#include "nau.h"
#include "nau/scene/camera.h"

class DlgCameras : public wxDialog, nau::event_::IListener
{
public:
	void updateDlg();
	static DlgCameras* Instance ();
	static void SetParent(wxWindow *parent);
	static wxWindow *Parent;

	void eventReceived(const std::string &sender, const std::string &eventType, 
		const std::shared_ptr<IEventData> &evt);
	void updateInfo(std::string name);
	std::string &getName();


protected:

	DlgCameras();
	DlgCameras(const DlgCameras&);
	DlgCameras& operator= (const DlgCameras&);
	static DlgCameras *Inst;

	/* GLOBAL STUFF */
	std::string m_Active;
	std::string m_Name;

	/* CAMERAS */
	wxButton *m_BAdd,*m_BActivate;
	wxPropertyGridManager *m_PG;
	wxComboBox *m_List;
	wxPGChoices m_ViewportLabels;

	/* EVENTS */
	void OnListSelect(wxCommandEvent& event);
	void OnAdd(wxCommandEvent& event);
	void OnActivate(wxCommandEvent &event);
	void OnPropsChange( wxPropertyGridEvent& e);

	void update();
	void setupPanel(wxSizer *siz, wxWindow *parent);
	void setupGrid();

	/* VIEWPORTS */
	void updateList();
	void updateViewportLabels();

	enum {
		DLG_COMBO,
		DLG_BUTTON_ADD,
		DLG_BUTTON_ACTIVATE,
		DLG_PROPS
	};

	typedef enum {
		NEW_CAMERA,
		PROPS_CHANGED
	} Notification;

	void notifyUpdate(Notification aNot, std::string camName, std::string value);

    DECLARE_EVENT_TABLE()
};






#endif



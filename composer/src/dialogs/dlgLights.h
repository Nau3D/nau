#ifndef __COMPOSER_DIALOGS_LIGHTS__
#define __COMPOSER_DIALOGS_LIGHTS__


#ifdef __GNUG__
#pragma implementation
#pragma interface
#endif

#include <nau.h>
#include <nau/scene/light.h>

// For compilers that support precompilation, includes "wx/wx.h".
#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif


#include <wx/grid.h>
#include <wx/propgrid/propgrid.h>
#include <wx/propgrid/advprops.h>
#include <wx/propgrid/manager.h>


class DlgLights : public wxDialog
{
public:
	void updateDlg();
	static DlgLights* Instance ();
	static void SetParent(wxWindow *parent);
	static wxWindow *Parent;

	void updateInfo(std::string name);


protected:

	DlgLights();
	DlgLights(const DlgLights&);
	DlgLights& operator= (const DlgLights&);
	static DlgLights *Inst;

	/* GLOBAL STUFF */
	std::string m_active;

	/* LIGHTS */
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
	//enum {
	//	/* LIGHT TYPES */
	//	DLG_MI_DIRECTIONAL=0,
	//	DLG_MI_POINT,
	//	DLG_MI_SPOT,
	//	DLG_MI_OMNI
	//}lightTypes;

	typedef enum {
		NEW_LIGHT,
		PROPS_CHANGED
	} Notification;

	void notifyUpdate(Notification aNot, std::string lightName, std::string value);

    DECLARE_EVENT_TABLE()
};






#endif



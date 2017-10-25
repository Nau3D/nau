#ifndef __COMPOSER_DIALOGS_RENDERER__
#define __COMPOSER_DIALOGS_RENDERER__


#ifdef __GNUG__
#pragma implementation
#pragma interface
#endif

#include <nau.h>
#include <nau/render/iRenderer.h>

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


class DlgRenderer : public wxDialog
{
public:
	void updateDlg();
	static DlgRenderer* Instance ();
	static void SetParent(wxWindow *parent);
	static wxWindow *Parent;

	void updateInfo(std::string name);


protected:

	DlgRenderer();
	DlgRenderer(const DlgRenderer&);
	DlgRenderer& operator= (const DlgRenderer&);
	static DlgRenderer *Inst;

	/* GLOBAL STUFF */
	wxPropertyGridManager *m_PG;
	wxComboBox *m_List;
	wxButton *m_BUpdate;


	/* EVENTS */
	void OnPropsChange( wxPropertyGridEvent& e);
	void OnUpdate(wxCommandEvent& event);
	void update();
	void setupPanel(wxSizer *siz, wxWindow *parent);
	void setupGrid();


	enum {
		DLG_COMBO,
		DLG_PROPS,
		DLG_BUTTON_UPDATE
	};

	typedef enum {
		PROPS_CHANGED
	} Notification;

	void notifyUpdate(Notification aNot, std::string lightName, std::string value);

    DECLARE_EVENT_TABLE()
};






#endif



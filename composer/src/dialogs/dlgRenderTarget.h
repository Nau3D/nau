#ifndef __DIALOGS_RENDERTARGET__
#define __DIALOGS_RENDERTARGET__

#ifdef __GNUG__
#pragma implementation
#pragma interface
#endif

#include <nau.h>
#include <nau/render/iRenderTarget.h>

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


class DlgRenderTargets : public wxDialog
{
public:
	void updateDlg();
	static DlgRenderTargets* Instance ();
	static void SetParent(wxWindow *parent);
	static wxWindow *Parent;

	void updateInfo(std::string name);


protected:

	DlgRenderTargets();
	DlgRenderTargets(const DlgRenderTargets&);
	DlgRenderTargets& operator= (const DlgRenderTargets&);
	static DlgRenderTargets *Inst;

	/* GLOBAL STUFF */
	std::string m_active;

	/* ITEMS */
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
		DLG_PROPS

	};

	typedef enum {
		PROPS_CHANGED
	} Notification;

	void notifyUpdate(Notification aNot, std::string itemName, std::string value);

    DECLARE_EVENT_TABLE()
};






#endif



#ifndef __COMPOSER_DIALOGS_PHYSICS__
#define __COMPOSER_DIALOGS_PHYSICS__


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
#include "nau/physics/physicsManager.h"

class DlgPhysics : public wxDialog
{
public:
	void updateDlg();
	static DlgPhysics* Instance ();
	static void SetParent(wxWindow *parent);
	static wxWindow *Parent;

	void updateInfo(std::string name);
	std::string &getName();


protected:

	DlgPhysics();
	DlgPhysics(const DlgPhysics&);
	DlgPhysics& operator= (const DlgPhysics&);
	static DlgPhysics *Inst;

	/* GLOBAL STUFF */
	std::string m_Active, m_Name;

	wxPropertyGridManager *m_PGGlobal, *m_PGMat;
	wxComboBox *m_List;

	/* EVENTS */
	void OnListSelect(wxCommandEvent& event);
	//void OnGlobalPropsChange( wxPropertyGridEvent& e);
	void OnMaterialPropsChange(wxPropertyGridEvent& e);

	void update();
	void updateList();
	void setupPanel(wxSizer *siz, wxWindow *parent);
	void setupGrid();

	enum {
		DLG_COMBO,
		DLG_PROPS_GLOBAL,
		DLG_PROPS_MAT
	};

    DECLARE_EVENT_TABLE()
};






#endif



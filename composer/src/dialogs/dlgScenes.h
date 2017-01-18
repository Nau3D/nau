#ifndef __COMPOSER_DIALOGS_SCENES__
#define __COMPOSER_DIALOGS_SCENES__


#ifdef __GNUG__
#pragma implementation
#pragma interface
#endif


#include <nau.h>

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



class DlgScenes : public wxDialog
{
public:
	void updateDlg();
	static DlgScenes* Instance ();
	static void SetParent(wxWindow *parent);

	void updateInfo(std::string name);


protected:

	DlgScenes();
	DlgScenes(const DlgScenes&);
	DlgScenes& operator= (const DlgScenes&);
	static DlgScenes *Inst;
	static wxWindow *Parent;
	wxWindow *m_Parent;


	/* GLOBAL STUFF */
	std::string m_Active;

	/* SPECIFIC */
	wxPropertyGridManager *m_PG;
	wxComboBox *m_List;
	wxListBox *m_Objects;
	wxStaticText *m_TType;//, *tBoundingBoxMin, *tBoundingBoxMax;


	/* EVENTS */
	void OnListSelect(wxCommandEvent& event);
	void OnAddFile(wxCommandEvent& WXUNUSED(event));
	void OnAddDir(wxCommandEvent& WXUNUSED(event));
	void OnNewScene(wxCommandEvent& WXUNUSED(event));
	void OnSaveScene(wxCommandEvent& WXUNUSED(event));
	void OnCompile( wxCommandEvent& WXUNUSED(event));
	void OnBuild( wxCommandEvent& WXUNUSED(event));

	void OnPropsChange( wxPropertyGridEvent& e);


	void update();
	void updateList();
	void setupPanel(wxSizer *siz, wxWindow *parent);
	void setupGrid();


	enum {
		DLG_COMBO,
		DLG_PROPS,
		NEW_SCENE,
		ADD_FILE,
		ADD_DIR,
		SAVE_SCENE,

		COMPILE,
		BUILD,

		TOOLBAR_ID
	};

	wxToolBar *m_toolbar;

    DECLARE_EVENT_TABLE()
};






#endif



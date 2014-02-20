#include "model.h"

#ifndef __SLANGER_DIALOGS__
#define __SLANGER_DIALOGS__


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

#include "wx/notebook.h"
#include "wx/grid.h"
#include "wx/colordlg.h"
#include "wx/propgrid/propgrid.h"
#include "wx/propgrid/advprops.h"
#include "wx/propgrid/manager.h"

#include "dlgModelInfo.h"


class TextEdit;

//--------------------------------------------------------------------
// Model Info modeless dialog











//--------------------------------------------------------------------

#include "defsext.h"
#include "edit.h"
#include "prefs.h"
#include <wx/settings.h>


class TextEdit: public wxFrame {

public:

	TextEdit(DlgModelInfo *parent, wxString filename, wxString title, int k, int type, 
				const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize);
//	TextEdit(DlgModelInfo *parent, wxString title);

    void OnClose(wxCloseEvent& event);

	void setFilename(wxString filename, wxString title);
	void saveFile(wxCommandEvent& event);

	wxString filename;
	DlgModelInfo *parent;
	Edit *txtShader;
	wxToolBar *toolbar;

private:

	void setup();
	int index;
	int type;

	enum {
		SHADER_SAVE = 0,
		SHADER_COMPILE
	};

	DECLARE_EVENT_TABLE()
};


//--------------------------------------------------------------------





//--------------------------------------------------------------------


//--------------------------------------------------------------------


// A custom modeless dialog
class MyModelessDialog : public wxDialog
{
public:
    MyModelessDialog(wxWindow *parent);

    void OnButton(wxCommandEvent& event);
    void OnClose(wxCloseEvent& event);

private:
    DECLARE_EVENT_TABLE()
};






// A custom modal dialog
class MyModalDialog : public wxDialog
{
public:
    MyModalDialog(wxWindow *parent);

    void OnButton(wxCommandEvent& event);

private:
    wxButton *m_btnFocused;
    wxButton *m_btnDelete;

    DECLARE_EVENT_TABLE()
};





#endif

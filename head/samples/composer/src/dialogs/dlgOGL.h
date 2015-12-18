#ifndef __SLANGER_DIALOGS_OGL__
#define __SLANGER_DIALOGS_OGL__


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

#include <wx/notebook.h>
#include <wx/grid.h>
#include <wx/colordlg.h>
#include <wx/propgrid/propgrid.h>
#include <wx/propgrid/advprops.h>
#include <wx/propgrid/manager.h>

//#ifndef _WX_OGL_H_

#include "glcanvas.h"




// A custom modeless dialog
class DlgOGL : public wxDialog
{
public:
	static DlgOGL* Instance ();
	static void SetParent(wxWindow *parent);
	static wxWindow *parent;


    DlgOGL(wxWindow *parent,GLCanvas *gls);
	~DlgOGL() {};

	void setupInfoPanel(wxSizer *siz, wxWindow *parent);
	void setupMIPanel(wxSizer *siz, wxWindow *parent);

protected: 

	DlgOGL();
	DlgOGL(const DlgOGL&);

	DlgOGL& operator= (const DlgOGL&);
	static DlgOGL *inst;

private:

	wxPropertyGridManager *pgmi,*pg;
//	GLCanvas *gls;

	

	enum {
		PGMI,
		DLG_UPDATE_EYE_COORDS,
		PGID,
	};

    DECLARE_EVENT_TABLE()
};

#endif



#ifndef __noname__
#define __noname__

#include <wx/propgrid/propgrid.h>
#include <wx/propgrid/propgrid.h>
#include <wx/propgrid/advprops.h>
#include <wx/propgrid/manager.h>
#include <wx/gdicmn.h>
#include <wx/font.h>
#include <wx/colour.h>
#include <wx/settings.h>
#include <wx/string.h>
#include <wx/sizer.h>
#include <wx/frame.h>

///////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
/// Class DlgAtomics
///////////////////////////////////////////////////////////////////////////////
class DlgAtomics : public wxDialog
{
	private:
	
	enum {
		DLG_ATOMICS

	};
	protected:
		wxPropertyGridManager* m_propertyGrid1;
		static DlgAtomics *inst;
	
		DlgAtomics();

	public:
		
		DlgAtomics( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("Slanger - Atomic Counters"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 500,300 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );
		~DlgAtomics();

		void updateDlg();
		void update();
	
		static DlgAtomics* Instance ();
		static void SetParent(wxWindow *parent);
		static wxWindow *parent;
};

#endif //__noname__

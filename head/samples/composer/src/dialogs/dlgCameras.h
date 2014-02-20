#ifndef __SLANGER_DIALOGS_CAMERAS__
#define __SLANGER_DIALOGS_CAMERAS__


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

class DlgCameras : public wxDialog
{
public:
	void updateDlg();
	static DlgCameras* Instance ();
	static void SetParent(wxWindow *parent);
	static wxWindow *parent;

	void updateInfo(std::string name);


protected:

	DlgCameras();
	DlgCameras(const DlgCameras&);
	DlgCameras& operator= (const DlgCameras&);
	static DlgCameras *inst;

	/* GLOBAL STUFF */
	std::string m_active;

	/* CAMERAS */
	wxButton *bAdd,*bActivate;
	wxPropertyGridManager *pg;
	wxComboBox *list;
	wxPGChoices viewportLabels;

	void addMatrix(wxPropertyGridManager *pg, Camera::Mat4Property m);
	void updateMatrix(Camera *cam, Camera::Mat4Property m) ;

	/* EVENTS */
	void OnListSelect(wxCommandEvent& event);
	void OnAdd(wxCommandEvent& event);
	void OnActivate(wxCommandEvent &event);
	void OnPropsChange( wxPropertyGridEvent& e);

	void update();
	void updateList();
	void updateViewportLabels();
	void setupPanel(wxSizer *siz, wxWindow *parent);


	enum {
		DLG_COMBO,
		DLG_BUTTON_ADD,
		DLG_BUTTON_ACTIVATE,
		DLG_PROPS

	};
	enum {
		/* CAMERA TYPES */
		DLG_MI_PERSPECTIVE=0,
		DLG_MI_ORTHOGONAL
	}cameraTypes;

	typedef enum {
		NEW_CAMERA,
		PROPS_CHANGED
	} Notification;

	void notifyUpdate(Notification aNot, std::string camName, std::string value);



    DECLARE_EVENT_TABLE()
};






#endif



#ifndef __COMPOSER_DIALOGS_SHADERS__
#define __COMPOSER_DIALOGS_SHADERS__


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


#include <wx/grid.h>
#include <wx/propgrid/propgrid.h>
#include <wx/propgrid/advprops.h>
#include <wx/propgrid/manager.h>

#include <nau.h>
#include <nau/material/iProgram.h>
#include <nau/material/programValue.h>
#include <nau/render/opengl/glProgram.h>

using namespace nau::material;

class DlgShaders : public wxDialog, IListener
{
public:
	void updateDlg();
	static DlgShaders* Instance ();
	static void SetParent(wxWindow *parent);
	static wxWindow *parent;

	void updateInfo(std::string name);

	std::string &getName() { return m_Name; };
	void eventReceived(const std::string &sender, const std::string &eventType,
		const std::shared_ptr<IEventData> &evt);

protected:

	DlgShaders();
	//DlgShaders(const DlgShaders&);
	//DlgShaders& operator= (const DlgShaders&);
	static DlgShaders *inst;
	std::string m_Name;

	/* GLOBAL STUFF */
	std::string m_Active;

	/* LIGHTS */
	wxButton *bAdd,*bActivate;
	wxComboBox *m_List;

	/* SPECIFIC */
	wxPropertyGridManager *m_PG;
	wxListBox *m_Log;
	wxPGProperty *m_Shader[IProgram::SHADER_COUNT];
	wxPGProperty *m_LinkStatus, *m_ValidateStatus,
		*m_ActiveAtomicBuffers,*m_ActiveAttributes, *m_ActiveUniforms ;
	wxButton *m_bValidate, *m_bCompileAndLink;


	/* EVENTS */
	void OnListSelect(wxCommandEvent& event);
	void OnAdd(wxCommandEvent& event);
	void OnPropsChange( wxPropertyGridEvent& e);

	void update();
	void updateList();
	void updateProgramProperties(GLProgram *p);
	void setupPanel(wxSizer *siz, wxWindow *parent);

	void OnProcessValidateShaders(wxCommandEvent& event);
	void OnProcessCompileAndLinkShaders(wxCommandEvent& event);
	void updateShaderAux();
	void updateLogAux(std::string aux);
	void addUniform(wxPGProperty *pid, wxString name, wxString type);
	wxString getUniformType(int type);


	enum {
		DLG_COMBO,
		DLG_BUTTON_ADD,
		DLG_PROPS,

		/* SHADERS */
		DLG_SHADER_FILES,
		DLG_SHADER_VALIDATE,
		DLG_SHADER_COMPILE_AND_LINK,
		DLG_SHADER_LOG,
		DLG_SHADER_UNIFORMS,

	};
		
	typedef enum {
		NEW_SHADER,
		PROPS_CHANGED
	} Notification;

	void notifyUpdate(Notification aNot, std::string shaderName, std::string value);


    DECLARE_EVENT_TABLE()
};






#endif



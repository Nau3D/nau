#ifndef __SLANGER_DIALOGS_SHADERS__
#define __SLANGER_DIALOGS_SHADERS__


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
#include <nau/render/iprogram.h>
#include <nau/render/opengl/glprogram.h>
#include <nau/material/programvalue.h>
//#include "dlgMaterials.h"

using namespace nau::material;

class DlgShaders : public wxDialog
{
public:
	void updateDlg();
	static DlgShaders* Instance ();
	static void SetParent(wxWindow *parent);
	static wxWindow *parent;

	void updateInfo(std::string name);


protected:

	DlgShaders();
	DlgShaders(const DlgShaders&);
	DlgShaders& operator= (const DlgShaders&);
	static DlgShaders *inst;

	/* GLOBAL STUFF */
	std::string m_active;

	/* LIGHTS */
	wxButton *bAdd,*bActivate;
	wxComboBox *list;

	/* SPECIFIC */
	wxPropertyGridManager *pg;
	wxCheckBox *m_cbUseShader; //,*m_cbShowGlobalU;
	wxListBox *m_log;
	wxPGProperty *m_Shader[IProgram::SHADER_COUNT];
	wxPGProperty *m_LinkStatus, *m_ValidateStatus,
		*m_ActiveAtomicBuffers,*m_ActiveAttributes, *m_ActiveUniforms ;
	wxButton *m_bValidate, *m_bCompile, *m_bLink;


	/* EVENTS */
	void OnListSelect(wxCommandEvent& event);
	void OnAdd(wxCommandEvent& event);
	void OnPropsChange( wxPropertyGridEvent& e);

	void update();
	void updateList();
	void updateProgramProperties(GlProgram *p);
	void setupPanel(wxSizer *siz, wxWindow *parent);

	void OnProcessValidateShaders(wxCommandEvent& event);
	void OnProcessCompileShaders(wxCommandEvent& event);
	void OnProcessLinkShaders(wxCommandEvent& event);
	void updateShaderAux();
	void updateLogAux(std::string aux);
	void addUniform(wxString name, wxString type);
	wxString getUniformType(int type);
//	void updateUniforms();


	enum {
/*		DLG_MI_VERTEX=0,
		DLG_MI_FRAGMENT,
		DLG_MI_GEOMETRY,
*/		DLG_COMBO,
		DLG_BUTTON_ADD,
		DLG_PROPS,


		/* SHADERS */
		DLG_SHADER_FILES,
		DLG_SHADER_VALIDATE,
		DLG_SHADER_COMPILE,
		DLG_SHADER_LINK,
		DLG_SHADER_USE,
		DLG_SHADER_LOG,
		DLG_SHADER_UNIFORMS,
		DLG_SHADER_SHOW_GLOBAL_UNIFORMS,

	};
		
	typedef enum {
		NEW_SHADER,
		PROPS_CHANGED
	} Notification;

	void notifyUpdate(Notification aNot, std::string shaderName, std::string value);


    DECLARE_EVENT_TABLE()
};






#endif



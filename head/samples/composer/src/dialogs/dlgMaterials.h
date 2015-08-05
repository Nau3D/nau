#ifndef __SLANGER_DIALOGS_MATERIALS__
#define __SLANGER_DIALOGS_MATERIALS__


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
#include <wx/notebook.h>
#include <wx/combobox.h>
#include <wx/grid.h>
#include <wx/colordlg.h>
#include <wx/frame.h>
#include <wx/propgrid/propgrid.h>
#include <wx/propgrid/advprops.h>
#include <wx/propgrid/manager.h>

class ImageGridCellRenderer;

#include <nau/material/materialLibManager.h>
#include <nau/material/material.h>
#include <nau/render/opengl/glProgram.h>
#include <nau.h>

#include <nau/event/ilistener.h>

//#ifndef _WX_OGL_H_
#include "dlgOGLpanels.h"

using namespace nau::material;


class DlgMaterials : public wxDialog, IListener
{


protected:
	DlgMaterials();
	DlgMaterials(const DlgMaterials&);
	DlgMaterials& operator= (const DlgMaterials&);

public:

	static DlgMaterials* Instance () ;
	static void SetParent(wxWindow *parent);

	std::string &getName () {return m_Name;}; 
	void eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt);

	void updateDlg();
	void updateTexture(Texture *tex);
	void updateTextureList();
	void updateActiveTexture();

private:
	static wxWindow *parent; 
//	static OGLCanvas *canvas;
	static DlgMaterials *inst;

	/* GLOBAL STUFF */
	wxWindow *m_parent;
	wxComboBox *materialList, *libList;
	Material *m_copyMat;

	std::string m_Name;

	void OnClose(wxCloseEvent& event);
	Material *getModelMaterial();
	void OnSelectMaterial(wxCommandEvent& event);
	void OnSelectLibMaterial(wxCommandEvent& event);
	void updateMaterialList();

	/* COLORS */
	wxPropertyGridManager *pgMaterial;

	void setupColorPanel(wxSizer *siz, wxWindow *parent);
	void OnProcessColorChange( wxPropertyGridEvent& e);
	void updateColors(Material *mm);

	/* TEXTURES */
	wxGrid *gridTextures;
	wxPGProperty *m_pgPropTextureList;
	wxPropertyGridManager *pgTextureProps;
	ImageGridCellRenderer *imagesGrid[8];
	wxPGChoices textureLabels;


	void setupTexturesPanel(wxSizer *siz, wxWindow *parent);
	void updateTextures(Material *mm,int index);

	void OnProcessTexturePropsChange( wxPropertyGridEvent& e);
	void setTextureUnit(int index);
	void OnprocessClickGrid(wxGridEvent &e);

	/* SHADERS */
	wxComboBox *shaderList;
	wxPropertyGridManager *pgShaderFiles,*pgShaderUniforms;
	wxCheckBox *m_cbUseShader; //,*m_cbShowGlobalU;
	wxPGProperty *m_vf,*m_ff;
	
	int aux;
	// These vars must be updated whenever Lights, 
	// Cameras or Textures (assigned to this material) are modified
	wxPGChoices m_pgCamList, m_pgLightList, m_pgTextureList, m_pgTextUnitsList;
	// This var keeps track of the units assigned to uniforms samplers
	std::vector<int> samplerUnits;

	void OnShaderListSelect(wxCommandEvent& event);
	void setupShaderPanel(wxSizer *siz, wxWindow *parent);
	void updateShader(Material *m);
	void OnProcessUseShader(wxCommandEvent& event);
	void OnProcessShaderUpdateUniforms( wxPropertyGridEvent& e);
//	void OnProcessShowGlobalUniforms(wxCommandEvent& event);
	void updateShaderAux(Material *m);
	void addUniform(ProgramValue &u,int showGlobal);
	void updateUniforms(Material *m);

	void auxSetMat4(wxPGProperty  *pid, wxPGProperty  *pid2, int edit, float *f);
	void auxSetMat3(wxPGProperty *pid, wxPGProperty *pid2, int edit, float *f);
	void auxSetVec4(wxPGProperty *pid, wxPGProperty *pid2, int edit, float *f);
	DlgOGLPanels panels;
	void OnProcessPanelChange( wxPropertyGridEvent& e);

	void setPropf4Aux(std::string propName,  vec4 &values);
	void setPropm4Aux(std::string propName,  mat4 &values);

	enum {
		DLG_MI_COMBO_MATERIAL,
		DLG_MI_COMBO_LIBMATERIAL,

		/* MATERIAL */
		DLG_MI_PGMAT,

		/* TEXTURES */
		DLG_MI_TEXTURE_GRID,
		DLG_MI_PGTEXTPROPS,
		PGUNIFORMS,
		PGTEXTURES,

		/* SHADERS */
		DLG_SHADER_COMBO,
		DLG_SHADER_FILES,
		DLG_SHADER_VALIDATE,
		DLG_SHADER_COMPILE,
		DLG_SHADER_LINK,
		DLG_SHADER_USE,
		DLG_SHADER_LOG,
		DLG_SHADER_UNIFORMS,
		DLG_SHADER_SHOW_GLOBAL_UNIFORMS,

		/* TOOLBAR */
		LIBMAT_NEW,
		LIBMAT_OPEN,
		LIBMAT_SAVE,
		LIBMAT_SAVEALL,

		MAT_NEW,
		MAT_CLONE,
		MAT_REMOVE,
		MAT_COPY,
		MAT_PASTE, 

		TOOLBAR_ID
	};

	wxToolBar *m_toolbar;
	void toolbarLibMatNew(wxCommandEvent& WXUNUSED(event) );
	void toolbarLibMatSave(wxCommandEvent& WXUNUSED(event) );
	void toolbarLibMatSaveAll(wxCommandEvent& WXUNUSED(event) );
	void toolbarLibMatOpen(wxCommandEvent& WXUNUSED(event) );

	void toolbarMatNew(wxCommandEvent& WXUNUSED(event) );
	void toolbarMatClone(wxCommandEvent& WXUNUSED(event) );
	void toolbarMatRemove(wxCommandEvent& WXUNUSED(event) );
	void toolbarMatCopy(wxCommandEvent& WXUNUSED(event) );
	void toolbarMatPaste(wxCommandEvent& WXUNUSED(event) );


	//typedef enum {CAMERA_POSITION, CAMERA_VIEW, CAMERA_UP, 
	//	LIGHT_POSITION, LIGHT_DIRECTION, LIGHT_COLOR, VEC3} vec3Comp;

	//typedef enum {LIGHT_ID, INT, TEXTURE_ID, TEXTURE_UNIT} intComp;

	//typedef enum {PROJECTIONMATRIX, MODELVIEWPROJECTIONMATRIX, TS05_MVPMATRIX, MAT4} mat4Comp;

	//typedef enum {
	//	NEW_TEXTURE,
	//} Notification;

	//void notifyUpdate(Notification aNot, std::string name, std::string value);



    DECLARE_EVENT_TABLE()
};

#include "imagegridcellrenderer.h"



#endif



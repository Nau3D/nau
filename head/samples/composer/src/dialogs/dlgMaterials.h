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

#include <nau/event/iListener.h>

//#ifndef _WX_OGL_H_
#include "dlgMatStatepanels.h"
#include "dlgMatBufferPanels.h"
#include "dlgMatITexPanels.h"

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

	~DlgMaterials();
	std::string &getName () {return m_Name;}; 
	void eventReceived(const std::string &sender, const std::string &eventType, 
		const std::shared_ptr<IEventData> &evt);

	void updateDlg();
	void updateTexture(ITexture *tex);
	void updateTextureList();
	void updateActiveTexture();

private:
	static wxWindow *parent; 
//	static OGLCanvas *canvas;
	static DlgMaterials *inst;

	wxBitmap *m_EmptyBitmap;

	/* GLOBAL STUFF */
	wxWindow *m_parent;
	wxComboBox *materialList, *libList;
	Material *m_copyMat;

	std::string m_Name;

	void OnClose(wxCloseEvent& event);
	std::shared_ptr<Material> &getModelMaterial();
	void OnSelectMaterial(wxCommandEvent& event);
	void OnSelectLibMaterial(wxCommandEvent& event);
	void updateMaterialList();

	/* COLORS */
	wxPropertyGridManager *pgMaterial;

	void setupColorPanel(wxSizer *siz, wxWindow *parent);
	void resetColorPanel();
	void OnProcessColorChange( wxPropertyGridEvent& e);
	void updateColors(std::shared_ptr<Material> &mm);

	/* TEXTURES */
	wxGrid *gridTextures;
	wxPGProperty *m_pgPropTextureList;
	wxPropertyGridManager *pgTextureProps;
	ImageGridCellRenderer *imagesGrid[8];
	wxPGChoices textureLabels;


	void setupTexturesPanel(wxSizer *siz, wxWindow *parent);
	void updateTextures(std::shared_ptr<Material> &mm,int index);

	void OnProcessTexturePropsChange( wxPropertyGridEvent& e);
	void resetTexturePropGrid();
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
	void updateShader(std::shared_ptr<Material> &m);
	void OnProcessUseShader(wxCommandEvent& event);
	void OnProcessShaderUpdateUniforms( wxPropertyGridEvent& e);
//	void OnProcessShowGlobalUniforms(wxCommandEvent& event);
	void updateShaderAux(std::shared_ptr<Material> &m);
	void addUniform(wxPGProperty *pid, ProgramValue &u,int showGlobal);
	void addBlockUniform(wxPGProperty *pid, ProgramBlockValue &u,int showGlobal);
	void updateUniforms(std::shared_ptr<Material> &m);

	void auxSetMat4(wxPGProperty  *pid, wxPGProperty  *pid2, int edit, float *f);
	void auxSetMat3(wxPGProperty *pid, wxPGProperty *pid2, int edit, float *f);
	void auxSetVec4(wxPGProperty *pid, wxPGProperty *pid2, int edit, float *f);

	DlgMatStatePanels panels;
	void OnProcessPanelChange( wxPropertyGridEvent& e);

	DlgMatBufferPanels m_BufferPanel;
	void OnProcessBufferPanelChange( wxPropertyGridEvent& e);
	void OnProcessBufferPanelSelect( wxCommandEvent& e);

	DlgMatImageTexturePanels m_ITexPanel;
	void OnProcessITexPanelChange( wxPropertyGridEvent& e);
	void OnProcessITexPanelSelect( wxCommandEvent& e);

	void setPropf4Aux(std::string &propName,  vec4 &values);
	void setPropm4Aux(std::string &propName,  mat4 &values);

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

	DECLARE_EVENT_TABLE()
};

#include "imagegridcellrenderer.h"



#endif



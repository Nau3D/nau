#ifndef __SLANGER_DIALOGS_TEXTURELIB__
#define __SLANGER_DIALOGS_TEXTURELIB__


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

#include <nau/event/iListener.h>
#include <nau/material/iTexture.h>

class ImageGridCellRenderer;
class DlgTextureLib : public wxDialog
{
public:
	void updateDlg();
	static DlgTextureLib* Instance ();
	static void SetParent(wxWindow *parent);
	static wxWindow *Parent;

	void updateTexInfo(int pos);
	std::string &getName () {return m_Name;}; 

	void eventReceived(const std::string &sender, const std::string &eventType, 
		const std::shared_ptr<IEventData> &evt);

	std::map<unsigned int, wxBitmap *> m_Bitmaps;

protected:

	DlgTextureLib();
	~DlgTextureLib();
	DlgTextureLib(const DlgTextureLib&);
	DlgTextureLib& operator= (const DlgTextureLib&);
	static DlgTextureLib *Inst;
	std::string m_Name;

	/* GLOBAL STUFF */
//	CProject *m_project;
	int m_ActiveTexture;

	/* TEXTURES */
	wxButton *m_BAddTex, *m_BSaveRaw, *m_BSavePNG, *m_BSaveHDR, *m_BUpdateIcons;
	wxGrid *m_GridTextures;
	wxPropertyGridManager *m_PGTextureProps;
	std::vector<ImageGridCellRenderer*> m_ImagesGrid;

	wxBitmap *m_EmptyBitmap;

	void setupTexturesPanel(wxSizer *siz, wxWindow *parent);
	void updateTextures(int index);
	void OnProcessTexturePropsChange( wxPropertyGridEvent& e);
	void setTextureProps(int index);
	//void loadTextureDialog(wxGridEvent &e);
	void OnprocessDClickGrid(wxGridEvent &e);
	void OnprocessClickGrid(wxGridEvent &e);
	void OnAddTex(wxCommandEvent& event);
	void OnSaveRaw(wxCommandEvent& event);
	void OnSavePNG(wxCommandEvent& event);
	void OnSaveHDR(wxCommandEvent& event);
	void OnUpdateIcons(wxCommandEvent &event);

	enum {
		/* TEXTURES */
		DLG_MI_TEXTURE_GRID,
		DLG_MI_PGTEXTPROPS,
		DLG_MI_BUTTON_ADD_TEX,
		DLG_MI_BUTTON_SAVE_RAW,
		DLG_MI_BUTTON_SAVE_PNG,
		DLG_MI_BUTTON_SAVE_HDR,
		DLG_MI_BUTTON_UPDATE_ICONS,
		PGTEXTURES,

	};

	typedef enum {
		NEW_TEXTURE,
		TEXTURE_ICON_UPDATE
	} Notification;

	void notifyUpdate(Notification aNot, std::string texName, std::string value);


    DECLARE_EVENT_TABLE()
};

#include "dialogs/imagegridcellrenderer.h"





#endif



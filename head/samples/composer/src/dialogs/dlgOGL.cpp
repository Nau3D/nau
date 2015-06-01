#pragma warning(disable:4018)
#include <GL/glew.h>
#include "dlgOGL.h"
#include <stdio.h>
#include <string.h>

BEGIN_EVENT_TABLE(DlgOGL, wxDialog)

END_EVENT_TABLE()

wxWindow *DlgOGL::parent = NULL;
DlgOGL *DlgOGL::inst = NULL;


void DlgOGL::SetParent(wxWindow *p) {

	parent = p;
}

DlgOGL* DlgOGL::Instance () {

	if (inst == NULL)
		inst = new DlgOGL();

	return inst;
}




DlgOGL::DlgOGL()
                : wxDialog(parent, -1, wxString("Nau - OpenGL State"),wxDefaultPosition,wxDefaultSize,wxRESIZE_BORDER|wxDEFAULT_DIALOG_STYLE)
{


//	this->gls = gl;

	wxBoxSizer *sizer = new wxBoxSizer(wxVERTICAL);

	wxNotebook *notebook = new wxNotebook(this,-1);
//	wxNotebookSizer *nbSizer = new wxNotebookSizer(notebook);

	/* Panels  */

	/* Info */
	wxPanel *pInfo = new wxPanel(notebook,-1);

	wxSizer *infos = new wxBoxSizer(wxVERTICAL);

	setupInfoPanel(infos,pInfo);

	pInfo->SetSizer(infos);
	pInfo->SetAutoLayout(TRUE);


	/* More Info */
	wxPanel *pMI = new wxPanel(notebook,-1);

	wxSizer *mis = new wxBoxSizer(wxVERTICAL);

	setupMIPanel(mis,pMI);
	
	pMI->SetSizer(mis);
	pMI->SetAutoLayout(TRUE);




	notebook->AddPage(pInfo,wxT("System"));
	notebook->AddPage(pMI,wxT("More info"));

//	sizer->Add(5,5,0, wxGROW | wxALL,5);
	sizer->Add(notebook,1,wxEXPAND | wxGROW | wxALL ,5);
//	sizer->Add(5,5,0, wxGROW | wxALL,5);

    SetAutoLayout(TRUE);
    SetSizer(sizer);

    sizer->SetSizeHints(this);
    sizer->Fit(this);
}


/* -----------------------------------------------------------------------

                                 INFO

  ------------------------------------------------------------------------ */


void DlgOGL::setupInfoPanel(wxSizer *siz, wxWindow *parent) {


	wxString s;
	wxStaticBox *sb = new wxStaticBox(parent,-1,wxT(" Driver "));
	wxSizer *sizer = new wxStaticBoxSizer(sb,wxHORIZONTAL);

		wxFlexGridSizer *gs = new wxFlexGridSizer(5,2,0,0);

		wxStaticText *st1 = new wxStaticText(parent,-1,wxT("GL_VENDOR:"));
		wxStaticText *st2 = new wxStaticText(parent,-1,wxString(glGetString(GL_VENDOR)));

		wxStaticText *st3 = new wxStaticText(parent,-1,wxT("GL_RENDERER:"));
		wxStaticText *st4 = new wxStaticText(parent,-1,wxString(glGetString(GL_RENDERER)));

		wxStaticText *st5 = new wxStaticText(parent,-1,wxT("GL_VERSION:"));
		wxStaticText *st6 = new wxStaticText(parent,-1,wxString(glGetString(GL_VERSION)));

		wxStaticText *st7 = new wxStaticText(parent,-1,wxT("GL_SHADING_LANGUAGE_VERSION:"));
		wxStaticText *st8 = new wxStaticText(parent,-1,wxString(glGetString(GL_SHADING_LANGUAGE_VERSION)));

		wxStaticText *st9 = new wxStaticText(parent, -1, wxT("Extension Count:"));
		int n;
		glGetIntegerv(GL_NUM_EXTENSIONS, &n);
		s.Printf(wxT("%d"), n);

		wxStaticText *st10= new wxStaticText(parent, -1, s);

		gs ->Add(st1,0,wxALL,3);
		gs ->Add(st2,0,wxALL,3);
		gs ->Add(st3,0,wxALL,3);
		gs ->Add(st4,0,wxALL,3);
		gs ->Add(st5,0,wxALL,3);
		gs ->Add(st6,0,wxALL,3);
		gs ->Add(st7,0,wxALL,3);
		gs ->Add(st8,0,wxALL,3);
		gs ->Add(st9,0,wxALL,3);
		gs ->Add(st10,0,wxALL,3);

	sizer->Add(gs,0,wxGROW|wxALL,5);

	wxStaticBox *sb3 = new wxStaticBox(parent,-1,wxT(" Extensions "));
	wxSizer *sizer3 = new wxStaticBoxSizer(sb3,wxVERTICAL);


		wxArrayString *li = new wxArrayString();
		
		//wschar *tmp,*text,*token;
		//tmp = (char *)glGetString(GL_EXTENSIONS);

		//if(tmp){

		//	text=_strdup(tmp);
		//	token=strtok(text," ");
		//	while(token!=NULL){
		//		s.Printf(wxT("%s"),token);
		//	  li->Add(s);
		//	  token=strtok(NULL," ");
		//	  }
		//}

		for (int i = 0; i < n ; ++i) {
		
			s.Printf(wxT("%s"), (char *)glGetStringi(GL_EXTENSIONS, i));
			li->Add(s);
		}
		//free(tmp);
		li->Sort();
		wxListBox *lb = new wxListBox(parent,-1);//,wxDefaultPosition,wxDefaultSize,li);
		lb->InsertItems(*li,0);


	sizer3->Add(lb,1,wxGROW|wxEXPAND|wxALL,5);

	siz->Add(sizer,0,wxGROW|wxALL,5);
//	siz->Add(sizer2,0,wxGROW|wxALL,5);
	siz->Add(sizer3,1,wxGROW|wxALL,5);
}


/* -----------------------------------------------------------------------

                                 MORE INFO

  ------------------------------------------------------------------------ */

void DlgOGL::setupMIPanel(wxSizer *siz, wxWindow *parent) {


	wxString s;
//	int i[4];

/*	wxStaticBox *sb2 = new wxStaticBox(parent,-1,wxT(" Buffers "));
	wxSizer *sizer2 = new wxStaticBoxSizer(sb2,wxHORIZONTAL);

		wxFlexGridSizer *gs2 = new wxFlexGridSizer(5,2,0,0);

		wxStaticText *st27 = new wxStaticText(parent,-1,wxT("RGBA:"));
		glGetIntegerv(GL_RED_BITS,i);glGetIntegerv(GL_GREEN_BITS,&i[1]);
		glGetIntegerv(GL_BLUE_BITS,&i[2]);glGetIntegerv(GL_ALPHA_BITS,&i[3]);
		s.Printf(wxT("%d,%d,%d,%d"),i[0],i[1],i[2],i[3]);
		wxStaticText *st28 = new wxStaticText(parent,-1,s);

		wxStaticText *st21 = new wxStaticText(parent,-1,wxT("GL_DEPTH_BITS:"));
		glGetIntegerv(GL_DEPTH_BITS,i); s.Printf(wxT("%d"),i[0]);
		wxStaticText *st22 = new wxStaticText(parent,-1,s);

		wxStaticText *st23 = new wxStaticText(parent,-1,wxT("GL_STENCIL_BITS:"));
		glGetIntegerv(GL_STENCIL_BITS,i); s.Printf(wxT("%d"),i[0]);
		wxStaticText *st24 = new wxStaticText(parent,-1,s);

		wxStaticText *st25 = new wxStaticText(parent,-1,wxT("GL_ACCUM_BITS:"));
		glGetIntegerv(GL_ACCUM_RED_BITS,i); glGetIntegerv(GL_ACCUM_GREEN_BITS,&i[1]);
		glGetIntegerv(GL_ACCUM_BLUE_BITS,&i[2]); glGetIntegerv(GL_ACCUM_ALPHA_BITS,&i[3]);
		s.Printf(wxT("%d"),i[0]+i[1]+i[2]+i[3]);
		wxStaticText *st26 = new wxStaticText(parent,-1,s);

		wxStaticText *st29 = new wxStaticText(parent,-1,wxT("GL_AUX_BUFFERS:"));
		glGetIntegerv(GL_ACCUM_RED_BITS,i); s.Printf(wxT("%d"),i[0]);
		wxStaticText *st30 = new wxStaticText(parent,-1,s);

		gs2 ->Add(st27,0,wxALL,3);
		gs2 ->Add(st28,0,wxALL,3);
		gs2 ->Add(st21,0,wxALL,3);
		gs2 ->Add(st22,0,wxALL,3);
		gs2 ->Add(st23,0,wxALL,3);
		gs2 ->Add(st24,0,wxALL,3);
		gs2 ->Add(st25,0,wxALL,3);
		gs2 ->Add(st26,0,wxALL,3);
		gs2 ->Add(st29,0,wxALL,3);
		gs2 ->Add(st30,0,wxALL,3);
	
	sizer2->Add(gs2,0,wxGROW|wxALL,5);

*/	wxStaticBox *sb3 = new wxStaticBox(parent,-1,wxT(" GL Limitations "));
	wxSizer *sizer3 = new wxStaticBoxSizer(sb3,wxHORIZONTAL);

	pgmi = new wxPropertyGridManager(parent, PGID,
				wxDefaultPosition, wxDefaultSize,
				// These and other similar styles are automatically
				// passed to the embedded wxPropertyGrid.
				wxPG_BOLD_MODIFIED|//wxPG_SPLITTER_AUTO_CENTER|
				// Plus defaults.
				wxPGMAN_DEFAULT_STYLE
           );
	pgmi->AddPage(wxT("Standard Items"));

	int d;

	glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &d);
	pgmi->Append(new  wxIntProperty(wxT("GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS"),wxPG_LABEL,d));

	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &d);
	pgmi->Append(new  wxIntProperty(wxT("GL_MAX_COMPUTE_WORK_GROUP_COUNT X"),wxPG_LABEL,d));

	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &d);
	pgmi->Append(new  wxIntProperty(wxT("GL_MAX_COMPUTE_WORK_GROUP_COUNT Y"),wxPG_LABEL,d));

	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &d);
	pgmi->Append(new  wxIntProperty(wxT("GL_MAX_COMPUTE_WORK_GROUP_COUNT Z"),wxPG_LABEL,d));

	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0,&d);
	pgmi->Append(new  wxIntProperty(wxT("GL_MAX_COMPUTE_WORK_GROUP_SIZE X"),wxPG_LABEL,d));

	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1,&d);
	pgmi->Append(new  wxIntProperty(wxT("GL_MAX_COMPUTE_WORK_GROUP_SIZE Y"),wxPG_LABEL,d));

	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2,&d);
	pgmi->Append(new  wxIntProperty(wxT("GL_MAX_COMPUTE_WORK_GROUP_SIZE Z"),wxPG_LABEL,d));

	glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS,&d);
	pgmi->Append(new  wxIntProperty(wxT("GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS"),wxPG_LABEL,d));

	glGetIntegerv(GL_MAX_DRAW_BUFFERS,&d);
	pgmi->Append(new  wxIntProperty(wxT("GL_MAX_DRAW_BUFFERS"),wxPG_LABEL,d));

	glGetIntegerv(GL_MAX_FRAGMENT_UNIFORM_COMPONENTS,&d);
	pgmi->Append(new  wxIntProperty(wxT("GL_MAX_FRAGMENT_UNIFORM_COMPONENTS"),wxPG_LABEL,d));

	glGetIntegerv(GL_MAX_TEXTURE_COORDS,&d);
	pgmi->Append(new  wxIntProperty(wxT("GL_MAX_TEXTURE_COORDS"),wxPG_LABEL,d));

	glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS,&d);
	pgmi->Append(new  wxIntProperty(wxT("GL_MAX_TEXTURE_IMAGE_UNITS"),wxPG_LABEL,d));

	glGetIntegerv(GL_MAX_VARYING_FLOATS,&d);
	pgmi->Append(new  wxIntProperty(wxT("GL_MAX_VARYING_FLOATS"),wxPG_LABEL,d));

	glGetIntegerv(GL_MAX_VERTEX_ATTRIBS,&d);
	pgmi->Append(new  wxIntProperty(wxT("GL_MAX_VERTEX_ATTRIBS"),wxPG_LABEL,d));

	glGetIntegerv(GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS,&d);
	pgmi->Append(new  wxIntProperty(wxT("GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS"),wxPG_LABEL,d));

	glGetIntegerv(GL_MAX_VERTEX_UNIFORM_COMPONENTS,&d);
	pgmi->Append(new  wxIntProperty(wxT("GL_MAX_VERTEX_UNIFORM_COMPONENTS"),wxPG_LABEL,d));

	glGetIntegerv(GL_MAX_TEXTURE_UNITS,&d);
	pgmi->Append(new  wxIntProperty(wxT("GL_MAX_TEXTURE_UNITS"),wxPG_LABEL,d));

	glGetIntegerv(GL_MAX_LIGHTS,&d);
	pgmi->Append(new  wxIntProperty(wxT("GL_MAX_LIGHTS"),wxPG_LABEL,d));

	glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS_EXT,&d);
	pgmi->Append(new  wxIntProperty(wxT("GL_MAX_COLOR_ATTACHMENTS_EXT"),wxPG_LABEL,d));

	glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE_EXT,&d);
	pgmi->Append(new  wxIntProperty(wxT("GL_MAX_RENDERBUFFER_SIZE_EXT"),wxPG_LABEL,d));

	glGetIntegerv(GL_MAX_COMBINED_ATOMIC_COUNTERS, &d);
	pgmi->Append(new  wxIntProperty(wxT("GL_MAX_COMBINED_ATOMIC_COUNTERS"),wxPG_LABEL,d));

	glGetIntegerv(GL_MAX_VERTEX_ATOMIC_COUNTERS, &d);
	pgmi->Append(new  wxIntProperty(wxT("GL_MAX_VERTEX_ATOMIC_COUNTERS"),wxPG_LABEL,d));

	glGetIntegerv(GL_MAX_TESS_CONTROL_ATOMIC_COUNTERS, &d);
	pgmi->Append(new  wxIntProperty(wxT("GL_MAX_TESS_CONTROL_ATOMIC_COUNTERS"),wxPG_LABEL,d));

	glGetIntegerv(GL_MAX_TESS_EVALUATION_ATOMIC_COUNTERS, &d);
	pgmi->Append(new  wxIntProperty(wxT("GL_MAX_TESS_EVALUATION_ATOMIC_COUNTERS"),wxPG_LABEL,d));

	glGetIntegerv(GL_MAX_GEOMETRY_ATOMIC_COUNTERS, &d);
	pgmi->Append(new  wxIntProperty(wxT("GL_MAX_GEOMETRY_ATOMIC_COUNTERS"),wxPG_LABEL,d));

	glGetIntegerv(GL_MAX_FRAGMENT_ATOMIC_COUNTERS, &d);
	pgmi->Append(new  wxIntProperty(wxT("GL_MAX_FRAGMENT_ATOMIC_COUNTERS"),wxPG_LABEL,d));

	glGetIntegerv(GL_MAX_ATOMIC_COUNTER_BUFFER_SIZE, &d);
	pgmi->Append(new  wxIntProperty(wxT("GL_MAX_ATOMIC_COUNTER_BUFFER_SIZE"),wxPG_LABEL,d));

	//	pgmi->SetSplitterPosition(300);

	sizer3->Add(pgmi,1,wxEXPAND|wxALL);
	sizer3->Add(5,5,0, wxEXPAND,5);

//	siz->Add(sizer2,0,wxGROW|wxALL,5);
	siz->Add(sizer3,1,wxGROW|wxALL,5);
	siz->Add(5,5,0, wxEXPAND|wxALL,5);

}



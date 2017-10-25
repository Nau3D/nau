#pragma warning( disable: 4049)
#pragma warning( disable: 4005)
#pragma warning( disable: 4290)
#pragma warning( disable: 4299)
#pragma warning( disable: 4099)

#ifdef _WIN32
#define APIENTRY __stdcall
#endif

//#include <vld.h>

#include <main.h>
#include <glcanvas.h>

#include <nau/errors.h>
#include <nau/clogger.h>
#include <nau/slogger.h>
#include <nau/debug/profile.h>
#include <nau/render/iAPISupport.h>

#include <wx/bitmap.h>
#include <wx/dir.h>
#include <wx/filename.h>
#include <wx/filesys.h>
#include <wx/splash.h>
#include <wx/image.h>

#include <fstream>

              
using namespace nau::math;

//helper functions
enum wxbuildinfoformat {
    short_f, long_f };

wxString wxbuildinfo(wxbuildinfoformat format)
{
    wxString wxbuild(wxVERSION_STRING);

    if (format == long_f )
    {
#if defined(__WXMSW__)
        wxbuild << _T("-Windows");
#elif defined(__UNIX__)
        wxbuild << _T("-Linux");
#endif

#if wxUSE_UNICODE
        wxbuild << _T("-unicode build");
#else
        wxbuild << _T("-ANSI build");
#endif // wxUSE_UNICODE
    }

    return wxbuild;
}

//wxApplication
IMPLEMENT_APP(WndComposer);

bool WndComposer::OnInit()
{
    FrmMainFrame* frame = new FrmMainFrame (0L, wxT("nau engine"));
    frame->Show();
//	frame->init();
    return true;
}



// Menu Options 
// File Menu
int idMenuProject = wxNewId();
int idMenuDir = wxNewId();
int idMenuModel = wxNewId();
int idMenuSaveProj = wxNewId();
//int idMenuModelAppend = wxNewId();
int idMenuProcess = wxNewId();
int idMenuQuit = wxNewId();
// Render Menu
int idMenuDlgRenderer = wxNewId();
int idMenuDlgPass = wxNewId();
int idMenuResetFrameCount = wxNewId();
int idMenuWireframe = wxNewId();
int idMenuPoint = wxNewId();
int idMenuSolid = wxNewId();
int idMenuMaterial = wxNewId();
int idMenuRenderFlagBoundingBox = wxNewId();
// Assets Menu
int idMenuDlgCameras = wxNewId();
int idMenuDlgViewports = wxNewId();
int idMenuDlgScenes = wxNewId();
int idMenuDlgLights = wxNewId();
// Materials Menu
int idMenuDlgTextures = wxNewId();
int idMenuDlgMaterials = wxNewId();
int idMenuDlgShaders = wxNewId();
int idMenuDlgAtomics = wxNewId();
int idMenuDlgBuffers = wxNewId();
int idMenuDlgRT = wxNewId();
// Physics Menu
int idMenuPhysicsOn = wxNewId();
int idMenuPhysicsOff = wxNewId();
int idMenuDlgPhysics = wxNewId();
// Debug Menu
int idMenuDbgBreak = wxNewId();
int idMenuDlgStep = wxNewId();
int idMenuDlgLog = wxNewId();
int idMenuDlgTrace = wxNewId();
int idMenuDlgProgramInfo = wxNewId();
int idMenuScreenShot = wxNewId();
int idMenuTraceSingle = wxNewId();
int idMenuTracing = wxNewId();
int idMenuProfileReset = wxNewId();
// About Menu
int idMenuAbout = wxNewId();
int idMenuDlgOGL = wxNewId();

BEGIN_EVENT_TABLE(FrmMainFrame, wxFrame)
// File Menu
EVT_MENU(idMenuProject, FrmMainFrame::OnProjectLoad)
EVT_MENU(idMenuSaveProj, FrmMainFrame::OnProjectSave)

EVT_MENU(idMenuDir, FrmMainFrame::OnDirectoryLoad)
EVT_MENU(idMenuModel, FrmMainFrame::OnModelLoad)
//EVT_MENU(idMenuModelAppend, FrmMainFrame::OnModelAppend)
EVT_MENU(idMenuProcess, FrmMainFrame::OnProcess)
EVT_MENU(idMenuQuit, FrmMainFrame::OnQuit)
// Render Menu

EVT_MENU(idMenuDlgRenderer, FrmMainFrame::OnDlgRenderer)
EVT_MENU(idMenuDlgPass, FrmMainFrame::OnDlgPass)
EVT_MENU(idMenuResetFrameCount, FrmMainFrame::OnResetFrameCount)
EVT_MENU_RANGE(idMenuWireframe, idMenuMaterial, FrmMainFrame::OnRenderMode)
EVT_MENU(idMenuRenderFlagBoundingBox, FrmMainFrame::OnSetRenderFlags)
// Assets Menu
EVT_MENU(idMenuDlgScenes, FrmMainFrame::OnDlgScenes)
EVT_MENU(idMenuDlgViewports, FrmMainFrame::OnDlgViewports)
EVT_MENU(idMenuDlgCameras, FrmMainFrame::OnDlgCameras)
EVT_MENU(idMenuDlgLights, FrmMainFrame::OnDlgLights)
// Materials Menu
EVT_MENU(idMenuDlgMaterials, FrmMainFrame::OnDlgMaterials)
EVT_MENU(idMenuDlgTextures, FrmMainFrame::OnDlgTextures)
EVT_MENU(idMenuDlgAtomics, FrmMainFrame::OnDlgAtomics)
EVT_MENU(idMenuDlgShaders, FrmMainFrame::OnDlgShaders)
EVT_MENU(idMenuDlgBuffers, FrmMainFrame::OnDlgBuffers)
EVT_MENU(idMenuDlgRT, FrmMainFrame::OnDlgRenderTargets)
// Physics Menu
EVT_MENU_RANGE(idMenuPhysicsOn, idMenuPhysicsOff, FrmMainFrame::OnPhysicsMode)
EVT_MENU(idMenuDlgPhysics, FrmMainFrame::OnDlgPhysics)
// Debug Menu
EVT_MENU(idMenuDlgLog, FrmMainFrame::OnDlgLog)
EVT_MENU(idMenuDbgBreak, FrmMainFrame::OnBreakResume)
EVT_MENU(idMenuDlgStep, FrmMainFrame::OnDlgDbgStep)
EVT_MENU(idMenuDlgTrace, FrmMainFrame::OnDlgDbgTraceRead)
EVT_MENU(idMenuDlgProgramInfo, FrmMainFrame::OnDlgDbgProgram)
EVT_MENU(idMenuScreenShot, FrmMainFrame::OnScreenShot)
EVT_MENU(idMenuTraceSingle, FrmMainFrame::OnTraceSingleFrame)
EVT_MENU(idMenuTracing, FrmMainFrame::OnTrace)
EVT_MENU(idMenuProfileReset, FrmMainFrame::OnProfileReset)
// About Menu
EVT_MENU(idMenuDlgOGL, FrmMainFrame::OnDlgOGL)
EVT_MENU(idMenuAbout, FrmMainFrame::OnAbout)

EVT_KEY_DOWN(FrmMainFrame::OnKeyDown)
	
//EVT_MENU(idMenuPhysicsBuild, FrmMainFrame::OnPhysicsBuild)
	
EVT_CLOSE(FrmMainFrame::OnClose)
	
 END_EVENT_TABLE()

BEGIN_EVENT_TABLE(GLCanvas, wxGLCanvas)
  EVT_SIZE(GLCanvas::OnSize)
  EVT_PAINT(GLCanvas::OnPaint)
  EVT_ERASE_BACKGROUND(GLCanvas::OnEraseBackground)
  EVT_ENTER_WINDOW(GLCanvas::OnEnterWindow)
  EVT_KEY_UP(GLCanvas::OnKeyUp)
  EVT_KEY_DOWN(GLCanvas::OnKeyDown)
  EVT_MOTION(GLCanvas::OnMouseMove)
  EVT_LEFT_DOWN (GLCanvas::OnLeftDown)
  EVT_LEFT_UP (GLCanvas::OnLeftUp)
#ifdef WIN32
  EVT_IDLE(GLCanvas::OnIdle)
#endif
  EVT_MIDDLE_UP(GLCanvas::OnMiddleUp)
  EVT_RIGHT_UP(GLCanvas::OnRightUp)
  EVT_RIGHT_DOWN(GLCanvas::OnRightDown)

END_EVENT_TABLE()


//#undef wxUSE_MENUS
//#undef wxUSE_STATUSBAR
//#define FINAL

FrmMainFrame::FrmMainFrame (wxFrame *frame, const wxString& title)
    : wxFrame(frame, -1, title), m_Canvas (0), materialsMenu (0), m_Width (0.0f), m_Height (0.0f), 
	m_Inited (false), m_Tracing(false)
{

#if wxUSE_MENUS
   // create a menu bar
	mbar = new wxMenuBar();

	// File Menu

	fileMenu = new wxMenu(_T(""));
	fileMenu->Append(idMenuProject, _("&Open Project\tCtrl-O"), _("Open a project file"));
	fileMenu->Append(idMenuDir, _("&Open Folder\tCtrl-F"), _("Loads all files in a folder"));
	fileMenu->Append(idMenuModel, _("&Open Model\tCtrl-M"), _("Reset and Loads a 3D Model"));
	fileMenu->Append(idMenuSaveProj, _("&Save Project (WIP)"), _("Save the project file"));
	//fileMenu->Append(idMenuModelAppend, _("&Append Model\tCtrl-+"), _("Appends a 3D Model to the scene"));
	fileMenu->Append(idMenuProcess, _("&Process Folder\tCtrl-P"), _("Convert all models in a folder to NBO format"));
	fileMenu->Append(idMenuQuit, _("&Quit\tAlt-F4"), _("Quit the application"));
	mbar->Append(fileMenu, _("&File"));

	// Render Menu

	renderMenu = new wxMenu(_T(""));
	renderMenu->Append(idMenuDlgRenderer, _("&Renderer Props\tCtrl-R"), _(""));
	renderMenu->Append(idMenuDlgPass, _("&Pass Library\tF2"), _("Show Pass Library"));
	renderMenu->AppendSeparator ();
	renderMenu->Append(idMenuResetFrameCount, _("&Reset Frame Count\tB"), _(""));
	renderMenu->AppendSeparator();
	renderMenu->AppendRadioItem (idMenuWireframe, _("&Wireframe"), _("Render in wireframe mode"));
	renderMenu->AppendRadioItem (idMenuSolid, _("&Solid"), _("Render in solid mode"));
	renderMenu->AppendRadioItem (idMenuMaterial, _("&Material"), _("Render with materials"));
	renderMenu->Check(idMenuMaterial, true);
	renderMenu->AppendSeparator ();
	renderMenu->AppendCheckItem(idMenuRenderFlagBoundingBox, _("Show &Bounding Boxes\tCtrl-B"));

	renderMenu->Enable(idMenuDlgPass, false);
	renderMenu->Enable(idMenuResetFrameCount, false);
	renderMenu->Enable(idMenuWireframe, false);
	renderMenu->Enable(idMenuSolid, false);
	renderMenu->Enable(idMenuMaterial, false);
	renderMenu->Enable(idMenuRenderFlagBoundingBox, false);

	mbar->Append (renderMenu, _("&Render"));

	// Assets Menu

	assetsMenu = new wxMenu(_T(""));
	assetsMenu->Append(idMenuDlgCameras, _("&Camera Library\tF3"), _("Show Camera Library"));
	assetsMenu->Append(idMenuDlgLights, _("&Light Library\tF4"), _("Show Light Library"));
	assetsMenu->Append(idMenuDlgViewports, _("&Viewports Library\tF5"), _("Show Viewport Library"));
    assetsMenu->Append(idMenuDlgScenes, _("&Scene Library\tF6"), _("Show Scene Library"));

	assetsMenu->Enable(idMenuDlgCameras, false);
	assetsMenu->Enable(idMenuDlgViewports, false);
	assetsMenu->Enable(idMenuDlgLights, false);
	assetsMenu->Enable(idMenuDlgScenes, false);

	mbar->Append (assetsMenu, _("&Assets"));

	// Materials Menu

	materialsMenu = new wxMenu(_T(""));
	materialsMenu->Append(idMenuDlgMaterials, _("Material Library Manager\tF7"), _("Show Material Libraries"));
    materialsMenu->Append(idMenuDlgTextures, _("Texture Library\tF8"), _("Show ITexture Library"));
    materialsMenu->Append(idMenuDlgShaders, _("Shader Library\tF9"), _("Show Shader Library"));
	materialsMenu->Append(idMenuDlgAtomics,  _("Atomics\tF10"), _("Show Atomics Info"));
	materialsMenu->Append(idMenuDlgBuffers, _("Buffer Info\tF11"), _("View Buffer information"));
	materialsMenu->Append(idMenuDlgRT, _("Render Targets\tF12"), _("View Render Target information"));

	materialsMenu->Enable(idMenuDlgMaterials, false);
	materialsMenu->Enable(idMenuDlgTextures, false);
	materialsMenu->Enable(idMenuDlgShaders, false);
	materialsMenu->Enable(idMenuDlgAtomics, false);
	materialsMenu->Enable(idMenuDlgBuffers, false);
	materialsMenu->Enable(idMenuDlgRT, false);
	
	mbar->Append (materialsMenu, _("&Materials"));

	// Physics Menu

	physicsMenu = new wxMenu(_T(""));
	physicsMenu->Append(idMenuDlgPhysics, _("Physics Property Manager"), _(""));
	physicsMenu->AppendRadioItem (idMenuPhysicsOn, _("Physics On"), _("Physics On"));
	physicsMenu->AppendRadioItem (idMenuPhysicsOff, _("Physics Off"), _("Physics Off"));
	physicsMenu->Check (idMenuPhysicsOff, true);
	mbar->Append (physicsMenu, _("Physics"));

	// Debug Menu

	debugMenu = new wxMenu(_T(""));
    debugMenu->Append(idMenuDlgLog, _("&Log\tCtrl-L"), _("Show Log"));
	debugMenu->Append(idMenuProfileReset, _("Reset Profiler\tCtrl-R"), _(""));
	debugMenu->AppendSeparator();
	debugMenu->Append(idMenuDbgBreak, _("&Pause"), _("Pauses or resumes rendering"));
	debugMenu->Append(idMenuDlgStep, _("&Advanced Pass Controller"), _("Aditional Pass control options"));
	debugMenu->AppendSeparator();
	debugMenu->Append(idMenuDlgProgramInfo, _("&Program Info\tCtrl-F1"), _("Shows Program information"));
	debugMenu->AppendSeparator();
	debugMenu->Append(idMenuDlgTrace, _("&Trace Log Window\tCtrl-T"), _("Displays Trace Info"));
	debugMenu->Append(idMenuTraceSingle, _("Trace Single Frame\tShift+T"), _("Trace 3D API calls for a single frame"));
	debugMenu->Append(idMenuTracing, _("Trace Start\tT"), _("Start/Stop tracing 3D API calls"));
	debugMenu->AppendSeparator();
	debugMenu->Append(idMenuScreenShot, _("Screen Shot\tCtrl-0"), _(""));

	debugMenu->Enable(idMenuDbgBreak, false);
	debugMenu->Enable(idMenuDlgStep, false);
	debugMenu->Enable(idMenuDlgProgramInfo, false);
	debugMenu->Enable(idMenuDlgTrace,false);
	debugMenu->Enable(idMenuTraceSingle, false);
	debugMenu->Enable(idMenuTracing, false);
	debugMenu->Enable(idMenuProfileReset, false);

	mbar->Append(debugMenu, _("&Debug"));

	// About Menu

	aboutMenu = new wxMenu(_T(""));
    aboutMenu->Append(idMenuDlgOGL, _("&OpenGL Properties\tCtrl-P"), _("Show info about OpenGL Context"));
    aboutMenu->Append(idMenuAbout, _("&About\tF1"), _("Show info about this application"));

    mbar->Append(aboutMenu, _("&Help"));


    SetMenuBar(mbar);


#endif // wxUSE_MENUS

#if wxUSE_STATUSBAR
    // create a status bar with some information about the used wxWidgets version
//   CreateStatusBar(2);
#endif // wxUSE_STATUSBAR

#ifdef FINAL
	//wxBitmap loaderBitmap;

	//wxInitAllImageHandlers();

	//wxSplashScreen *splash = 0;

	//if (loaderBitmap.LoadFile (wxT("loader.jpg"), wxBITMAP_TYPE_JPEG)) {
	//	splash = new wxSplashScreen (loaderBitmap, wxSPLASH_CENTRE_ON_SCREEN , 0, NULL, 
	//		-1, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER|wxSTAY_ON_TOP);
	//}
#endif

	//// Dialogs //
	//DlgOGL::SetParent(this);
	//DlgLog::Instance()->updateDlg();

	//m_pRoot = (Nau *)Nau::Create();


	//bool nauInit (false);
        
    Show(true);
	int attribList[] = {
		WX_GL_RGBA,
	//	WX_GL_DOUBLEBUFFER,
		WX_GL_DEPTH_SIZE, 24,
		WX_GL_STENCIL_SIZE, 8, 
		WX_GL_SAMPLE_BUFFERS,1,
		WX_GL_SAMPLES,1,
		0};


#define WGL_CONTEXT_MAJOR_VERSION_ARB		0x2091
#define WGL_CONTEXT_MINOR_VERSION_ARB		0x2092
#define	WGL_CONTEXT_LAYER_PLANE_ARB			0x2093
#define WGL_CONTEXT_FLAGS_ARB				0x2094
#define WGL_CONTEXT_PROFILE_MASK_ARB		0x9126

#define WGL_CONTEXT_DEBUG_BIT_ARB				0x0001
#define WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB	0x0002

#define WGL_CONTEXT_CORE_PROFILE_BIT_ARB			0x00000001
#define	WGL_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB	0x00000002
	//int major = NAU_OPENGL_VERSION / 100;
	//int minor = (NAU_OPENGL_VERSION - major * 100 ) / 10;
	int contextAttribList[] = {
			//WGL_CONTEXT_MAJOR_VERSION_ARB, major,
            //WGL_CONTEXT_MINOR_VERSION_ARB, minor, 
            WGL_CONTEXT_FLAGS_ARB, WGL_CONTEXT_DEBUG_BIT_ARB,
            WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
 
			0};

	m_Canvas = new GLCanvas (this , -1);//, NULL/*, contextAttribList*/);
    //m_Canvas->DoSetCurrent();

	//bool trace = false;
	//int framesToTrace = 1;

	//if (wxGetApp().argc > 1) {
	//	int param = 1;
	//	if (wxGetApp().argv[1].c_str()[0] != '-') {
	//		loadProject(wxGetApp().argv[1].c_str());
	//		param = 2;
	//	}
	//	for (int i = param; i < wxGetApp().argc; ++i) {
	//		if (wxGetApp().argv[i].ToStdString() == "-trace") {

	//			trace = true;
	//			long frames;
	//			if (i + 1 < wxGetApp().argc && wxGetApp().argv[i + 1].ToLong(&frames)) {
	//				framesToTrace = (int)frames;
	//				++i;
	//			}
	//		}
	//	}
	//}

	//try {
	//	wxMessageBox (_("Nau error!"), _("Kaput!")); 
	//	nauInit = m_pRoot->init(trace);
	//	wxMessageBox (_("Nau error!"), _("Kaput!")); 
	//	if (trace)
	//		m_pRoot->setTrace(framesToTrace);
	//	
	//}
	//catch(std::string &s) {
	//	wxMessageBox((wxString)s.c_str());
	//	exit(1);
	//}

	//if (true != nauInit){
	//	wxMessageBox (_("Nau error!"), _("Kaput!"));
	//	exit (1);
	//}
	// 
	//m_Canvas->setEngine (m_pRoot);
	//m_Canvas->setCamera();

	//// Dialogs //
	//DlgOGL::SetParent(this);
	//DlgLog::Instance()->updateDlg();
	//if (m_pRoot->getPhysicsManager()->isPhysicsAvailable())
	//	DlgPhysics::SetParent(this);
	//else
	//	mbar->EnableTop(mbar->FindMenu(_("Physics")), false);
	//DlgTextureLib::SetParent(this);
	//DlgCameras::SetParent(this);
	//DlgMaterials::SetParent(this);
	//DlgLights::SetParent(this);
	//DlgShaders::SetParent(this);
	//DlgScenes::SetParent(this);
	//DlgPass::SetParent(this);
	//DlgAtomics::SetParent(this);
	//DlgViewports::SetParent(this);
	//DlgTrace::SetParent(this);
	//DlgDbgPrograms::SetParent(this);
	//DlgDbgBuffers::SetParent(this);
	//DlgDbgStep::SetParent(this);
	//DlgDbgStep::SetCanvas(m_Canvas);
	//DlgRenderTargets::SetParent(this);
//#ifdef GLINTERCEPTDEBUG
//	gliSetIsGLIActive(true);
//#endif

#ifdef FINAL
	startStandAlone();
	//delete splash;
#endif
	//int w, h;
	//GetClientSize(&w, &h);
	//SetClientSize(w+1, h+1);

	//if (wxGetApp().argc > 1) {
	//	int param = 1;
	//	if (wxGetApp().argv[1].c_str()[0] != '-') {
	//		loadProject(wxGetApp().argv[1].c_str());
	//		param = 2;
	//	}
	//	for (int i = param; i < wxGetApp().argc; ++i) {
	//		if (wxGetApp().argv[i].ToStdString() == "-trace") {
	//			long frames;
	//			if (i + 1 < wxGetApp().argc && wxGetApp().argv[i + 1].ToLong(&frames)) {
	//				m_pRoot->setTrace((int)frames);
	//				++i;
	//			}
	//			else
	//				m_pRoot->setTrace(1);
	//		}
	//	}
	//}
	Show(true);
}


FrmMainFrame::~FrmMainFrame() {

//#ifdef GLINTERCEPTDEBUG
//	gliSetIsGLIActive(true);
//#endif
}


void
FrmMainFrame::init() {

	if (m_Inited)
		return;
	m_Inited= true;

	m_Canvas->DoSetCurrent();

	// Dialogs //
	DlgOGL::SetParent(this);
	DlgLog::Instance()->updateDlg();

	m_pRoot = (Nau *)Nau::Create();

	bool nauInit(false);

	bool loadProjectFlag = false;
	bool trace = false;
	bool pause = false;
	int framesToTrace = 1;
	bool invalidOptions = false;

	if (wxGetApp().argc > 1) {
		int param = 1;
		if (wxGetApp().argv[1].c_str()[0] != '-') {
			loadProjectFlag = true;
			param = 2;
		}
		for (int i = param; i < wxGetApp().argc; ++i) {
			if (wxGetApp().argv[i].ToStdString() == "-trace") {

				trace = true;
				long frames;
				if (i + 1 < wxGetApp().argc && wxGetApp().argv[i + 1].ToLong(&frames)) {
					framesToTrace = (int)frames;
					++i;
				}
			}
			else if (wxGetApp().argv[i].ToStdString() == "-pause") {
				pause = true;
			}
			else {
				invalidOptions = true;
				
			}
		}
	}
	if (invalidOptions)
		wxMessageBox("Invalid option. Valid options are -trace and -pause");

	try {
		nauInit = m_pRoot->init(trace);
		if (trace)
			m_pRoot->setTrace(framesToTrace);

	}
	catch (std::string &s) {
		wxMessageBox((wxString)s.c_str());
		exit(1);
	}

	if (true != nauInit) {
		wxMessageBox(_("Nau error!"), _("Kaput!"));
		exit(1);
	}

	m_Canvas->setEngine(m_pRoot);
	m_Canvas->setCamera();

	// Dialogs //
	DlgOGL::SetParent(this);
	DlgLog::Instance()->updateDlg();
	if (m_pRoot->getPhysicsManager()->isPhysicsAvailable())
		DlgPhysics::SetParent(this);
	else
		mbar->EnableTop(mbar->FindMenu(_("Physics")), false);
	DlgTextureLib::SetParent(this);
	DlgCameras::SetParent(this);
	DlgMaterials::SetParent(this);
	DlgLights::SetParent(this);
	DlgShaders::SetParent(this);
	DlgScenes::SetParent(this);
	DlgPass::SetParent(this);
	DlgAtomics::SetParent(this);
	DlgViewports::SetParent(this);
	DlgTrace::SetParent(this);
	DlgDbgPrograms::SetParent(this);
	DlgDbgBuffers::SetParent(this);
	DlgDbgStep::SetParent(this);
	DlgDbgStep::SetCanvas(m_Canvas);
	DlgRenderTargets::SetParent(this);
	DlgRenderer::SetParent(this);

	int w, h;
	GetClientSize(&w, &h);
	SetClientSize(w + 1, h + 1);

	//if (wxGetApp().argc > 1) {
	//	int param = 1;
	//	if (wxGetApp().argv[1].c_str()[0] != '-') {
	//		loadProjectFlag = true;
	//		//loadProject(wxGetApp().argv[1].c_str());
	//		param = 2;
	//	}
	//	for (int i = param; i < wxGetApp().argc; ++i) {
	//		if (wxGetApp().argv[i].ToStdString() == "-trace") {
	//			long frames;
	//			if (i + 1 < wxGetApp().argc && wxGetApp().argv[i + 1].ToLong(&frames)) {
	//				m_pRoot->setTrace((int)frames);
	//				++i;
	//			}
	//			else
	//				m_pRoot->setTrace(1);
	//		}
	//	}
	//}
	if (loadProjectFlag)
		loadProject(wxGetApp().argv[1].c_str());
	if (pause) {
		m_Canvas->BreakResume();
		debugMenu->Enable(idMenuDlgStep, true);
		debugMenu->SetLabel(idMenuDbgBreak, "Resume");
	}
}


void 
FrmMainFrame::OnClose(wxCloseEvent& event) {

	if (m_pRoot) {
		m_Canvas->setEngine(NULL);
		delete m_pRoot;
		m_pRoot = NULL;
		delete m_Canvas;
	}
	Destroy();
	wxExit();
}


void
FrmMainFrame::OnDlgScenes(wxCommandEvent& event) {

	DlgScenes::Instance()->Show(TRUE);
}


void
FrmMainFrame::OnDlgAtomics(wxCommandEvent& event) {

	DlgAtomics::Instance()->Show(TRUE);
}


void 
FrmMainFrame::OnDlgOGL(wxCommandEvent& event) {

	DlgOGL::Instance()->Show(TRUE);
}


void 
FrmMainFrame::OnDlgLog(wxCommandEvent& event) {

	DlgLog::Instance()->Show(TRUE);
}


void
FrmMainFrame::OnDlgViewports(wxCommandEvent& event) {

	DlgViewports::Instance()->Show(TRUE);
}


void 
FrmMainFrame::OnDlgTextures(wxCommandEvent& event) {

	DlgTextureLib::Instance()->Show();
}


void 
FrmMainFrame::OnDlgShaders(wxCommandEvent& event) {

	DlgShaders::Instance()->Show();
}


void 
FrmMainFrame::OnDlgCameras(wxCommandEvent& event) {

	DlgCameras::Instance()->Show();
}


void 
FrmMainFrame::OnDlgLights(wxCommandEvent& event) {

	DlgLights::Instance()->Show();
}


void 
FrmMainFrame::OnDlgMaterials(wxCommandEvent& event) {

	DlgMaterials::Instance()->Show();
}


void 
FrmMainFrame::OnDlgPass(wxCommandEvent& event) {

	DlgPass::Instance()->Show();
}


void 
FrmMainFrame::updateDlgs() {

	if (APISupport->apiSupport(IAPISupport::BUFFER_ATOMICS)) {
		DlgAtomics::Instance()->updateDlg();
		materialsMenu->Enable(idMenuDlgAtomics, true);
	}

	DlgCameras::Instance()->updateDlg();
	DlgTextureLib::Instance()->updateDlg();
	DlgMaterials::Instance()->updateDlg();
	DlgLog::Instance()->updateDlg();
	DlgShaders::Instance()->updateDlg();
	DlgLights::Instance()->updateDlg();
	DlgScenes::Instance()->updateDlg();
	DlgPass::Instance()->updateDlg();
	DlgViewports::Instance()->updateDlg();
	DlgDbgPrograms::Instance()->updateDlg();
	DlgTrace::Instance()->updateDlg();
	DlgRenderTargets::Instance()->updateDlg();
	DlgRenderer::Instance()->updateDlg();
	DlgDbgStep::Instance()->updateDlg();

	if (m_pRoot->getPhysicsManager()->isPhysicsAvailable())
		DlgPhysics::Instance()->updateDlg();

	renderMenu->Enable(idMenuDlgPass, true);
	renderMenu->Enable(idMenuWireframe, true);
	renderMenu->Enable(idMenuSolid, true);
	renderMenu->Enable(idMenuMaterial, true);
	renderMenu->Enable(idMenuRenderFlagBoundingBox, true);
	renderMenu->Enable(idMenuResetFrameCount, true);

	assetsMenu->Enable(idMenuDlgCameras,true);
	assetsMenu->Enable(idMenuDlgLights,true);
	assetsMenu->Enable(idMenuDlgScenes, true);
	assetsMenu->Enable(idMenuDlgViewports, true);

	materialsMenu->Enable(idMenuDlgMaterials, true);
	materialsMenu->Enable(idMenuDlgTextures, true);
	materialsMenu->Enable(idMenuDlgShaders, true);
	materialsMenu->Enable(idMenuDlgAtomics, true);
	materialsMenu->Enable(idMenuDlgBuffers, true);
	materialsMenu->Enable(idMenuDlgRT, true);

	debugMenu->Enable(idMenuDlgProgramInfo, true);
	debugMenu->Enable(idMenuDlgTrace, true);
	debugMenu->Enable(idMenuTraceSingle, true);
	debugMenu->Enable(idMenuTracing, true);
	debugMenu->Enable(idMenuProfileReset, true);
	debugMenu->Enable(idMenuDbgBreak, true);
	debugMenu->Enable(idMenuDlgStep, false);


	if (m_Canvas->IsPaused()) {

		debugMenu->Enable(idMenuDlgStep, true);
		debugMenu->SetLabel(idMenuDbgBreak, "Resume");
	}
	else {

		debugMenu->Enable(idMenuDlgStep, false);
		debugMenu->SetLabel(idMenuDbgBreak, "Pause");
	}

	//bool traceOn = m_pRoot->getTraceStatus();
	//if (traceOn)
	//	debugMenu->SetLabel(idMenuTracing, "Trace Stop\tT");
	//else 
		debugMenu->SetLabel(idMenuTracing, "Trace Start\tT");

	//m_pRoot->setTrace(0);
}


void 
FrmMainFrame::OnResetFrameCount(wxCommandEvent& event) {

	RENDERER->setPropui(IRenderer::FRAME_COUNT, 0);
}


void
FrmMainFrame::OnScreenShot(wxCommandEvent& event) {

	RENDERER->saveScreenShot();
}


void
FrmMainFrame::OnTraceSingleFrame(wxCommandEvent& event) {

	m_pRoot->setTrace(1);
}


void
FrmMainFrame::OnTrace(wxCommandEvent& event) {

	if (m_Tracing) {
		m_pRoot->setTrace(0);
		debugMenu->SetLabel(idMenuTracing, "Trace Start\tT");
	}
	else {
		m_pRoot->setTrace(-1);
		debugMenu->SetLabel(idMenuTracing, "Trace Stop\tT");
	}
	m_Tracing = !m_Tracing;
}


void
FrmMainFrame::OnProfileReset(wxCommandEvent& event) {

	Profile::Reset();
}


void
FrmMainFrame::OnDirectoryLoad (wxCommandEvent& event) {

	wxDirDialog *openDirDlg = new wxDirDialog (this);

	wxStopWatch aTimer;

	if (wxID_OK == openDirDlg->ShowModal()) {
		aTimer.Start();
		wxDir dir (openDirDlg->GetPath());

		try {
			DlgLog::Instance()->updateDlg();
			DlgLog::Instance()->clear();
			m_pRoot->readDirectory (std::string(dir.GetName().mb_str()));
			m_Canvas->setCamera();
			updateDlgs();
			SLOG("done reading");
#ifndef FINAL
			double t =  aTimer.Time()/1000.0;
			SLOG("Elapsed time: %f", t);
#endif		
		}
		catch(std::string &s) {
			wxMessageBox(wxString(s.c_str()));
		}
		catch (...) {
			wxMessageBox("An exception has occured during folder load");
		}
	}
	delete openDirDlg;
}

void 
FrmMainFrame::OnProjectSave(wxCommandEvent & event) {

	static const wxChar *fileTypes = _T("XML files|*.xml|All files|*.*");
	wxFileDialog *saveFileDlg = new wxFileDialog(this, _("Save File"), _(""), _(""), fileTypes, wxFD_OPEN, wxDefaultPosition);

	if (wxID_OK == saveFileDlg->ShowModal()) {
		wxString path = saveFileDlg->GetPath();
		m_pRoot->saveProject(std::string(path.mb_str()));
	}
	delete saveFileDlg;
}


void
FrmMainFrame::OnModelLoad (wxCommandEvent& event) {

	static const wxChar *fileTypes = _T("3D Files (*.nbo, *.3ds, *.dae, *.obj, *.xml, *.blend, *.ply, *.lwo, *.stl, *.cob, *.scn)|*.nbo;*.3ds;*.dae;*.obj;*.xml;*.blend;*.ply;*.lwo;*.stl;*.cob;*.scn|NBO files (*.nbo)|*.nbo|COLLADA files (*.dae)|*.dae|3DS files (*.3ds)|*.3ds|OBJ files (*.obj)|*.obj|Ogre XML Meshes (*.xml)|*.xml|Blender files (*.blend)|*.blend|Stanford Polygon Library (*.ply)|*.ply|Lightwave (*.lwo)|*.lwo|Stereolithography (*.stl)|*.stl|True Space Obj (*.cob)|*.cob|True Space Scene (*scn)|*.scn");
	wxFileDialog *openFileDlg = new wxFileDialog(this, _("Open File"), _(""), _(""), fileTypes, wxFD_OPEN, wxDefaultPosition);

	if (wxID_OK == openFileDlg->ShowModal ()) {
		wxStopWatch aTimer;
		aTimer.Start();
		wxString path = openFileDlg->GetPath ();

		try {
			DlgLog::Instance()->updateDlg();
			DlgLog::Instance()->clear();
			m_pRoot->readModel (std::string(path.mb_str()));	
			m_Canvas->setCamera();
			updateDlgs();
#ifndef FINAL
			float t =  aTimer.Time()/1000.0;
			SLOG("Elapsed time: %f", t);
#endif		
		}
		catch(std::string &s) {
			wxMessageBox(wxString(s.c_str()));
		}
	}
	delete openFileDlg;
}


void 
FrmMainFrame::OnProjectLoad(wxCommandEvent& event) {

	static const wxChar *fileTypes = _T( "XML files|*.xml|All files|*.*");
	wxFileDialog *openFileDlg = new wxFileDialog (this, _("Open File"), _(""), _(""), fileTypes, wxFD_OPEN, wxDefaultPosition);

	if (wxID_OK == openFileDlg->ShowModal ()) {
		wxString path = openFileDlg->GetPath (); 
		loadProject(path.c_str());
	}
	delete openFileDlg;
}


void 
FrmMainFrame::loadProject(const char *s) {

	wxStopWatch aTimer;
	aTimer.Start();

	LOG_trace("Loading project %s", s);

	try {
		m_pRoot->clear();
		DlgLog::Instance()->updateDlg();
		DlgLog::Instance()->clear();
		int width = 0, height = 0;
		std::string ProjectFile(s);
		m_pRoot->readProjectFile(ProjectFile, &width, &height);
		if (width)
			SetClientSize(width, height);
		m_Canvas->setCamera();
		updateDlgs();
#ifndef FINAL

		float t = aTimer.Time() / 1000.0;
		SLOG("Elapsed time: %f", t);
#endif

		DlgTrace::Instance()->clear();

	}
	catch (nau::ProjectLoaderError &e) {
		wxMessageBox(wxString(e.getException().c_str()));
	}
	catch (std::string s) {
		wxMessageBox(wxString(s.c_str()));
	}
	LOG_trace("Project loaded");
}


void 
FrmMainFrame::OnProcess (wxCommandEvent& event) {

	wxDirDialog *openDirDlg = new wxDirDialog (this);

	if (wxID_OK == openDirDlg->ShowModal()) {
		wxDir dir (openDirDlg->GetPath());

		wxDirDialog *openDirDlgDst = new wxDirDialog (this);

		if (wxID_OK == openDirDlgDst->ShowModal()) {

			wxString aFile;

			bool cont = dir.GetFirst (&aFile);
			while (cont){

				wxFileName fn (openDirDlg->GetPath() + wxT("/") + aFile);
				LOG_INFO ("Loading: %s", (fn.GetFullPath()).ToStdString().c_str());
                m_pRoot->loadAsset (std::string(fn.GetFullPath().mb_str()), "MainScene");
				fn.SetPath (openDirDlgDst->GetPath());
				fn.SetExt(wxT("nbo"));
				m_pRoot->writeAssets ("NBO", (const char *)(fn.GetFullPath()).c_str(), "MainScene");
				cont = dir.GetNext (&aFile);
			}
			wxMessageBox (_("Process done..."), _("Composer"));
		}
	}
	delete openDirDlg;
}


void FrmMainFrame::OnQuit(wxCommandEvent& event) {

	Close(true);
}


void FrmMainFrame::OnAbout(wxCommandEvent& event) {

    wxString msg = wxbuildinfo(long_f);
    wxMessageBox(_("Welcome to Composer - Nau3D's GUI\nhttps://github.com/Nau3D\nhttp://nau3d.di.uminho.pt"), _("About Composer"));
}


void
FrmMainFrame::OnRenderMode(wxCommandEvent& event) {

	if (event.GetId() == idMenuWireframe){
		RENDERMANAGER->setRenderMode (nau::render::IRenderer::WIREFRAME_MODE);
	}
	if (event.GetId() == idMenuPoint) {
		RENDERMANAGER->setRenderMode (nau::render::IRenderer::POINT_MODE);
	}
	if (event.GetId() == idMenuSolid){
		RENDERMANAGER->setRenderMode (nau::render::IRenderer::SOLID_MODE);
	}
	if (event.GetId() == idMenuMaterial){
		RENDERMANAGER->setRenderMode (nau::render::IRenderer::MATERIAL_MODE);
	}
}


void 
FrmMainFrame::OnSetRenderFlags(wxCommandEvent& event) {

	if (event.GetId() == idMenuRenderFlagBoundingBox) {
		NAU->setRenderFlag(nau::Nau::BOUNDING_BOX_RENDER_FLAG,event.IsChecked());
	}
}


void
FrmMainFrame::startStandAlone (void) {
/*
	char *cwd;

	cwd = _getcwd (0, 0);

	wxString path (_("C:\\pl3d\\nxtg"));

	wxString filename;
	wxArrayString files;

	wxDir::GetAllFiles (path, &files);
	
	files.Sort();

	wxArrayString::iterator filesIter;

	filesIter = files.begin();

	for ( ; filesIter != files.end(); filesIter++) {
	  wxFileName fn (path + _("/") + (*filesIter));

	  std::string AssetPath ((const char *) fn.GetFullPath().c_str());
	  m_pRoot->loadAsset (AssetPath, "MainScene");
	}

	RENDERMANAGER->getScene ("MainScene")->compile();

	m_pRoot->enablePhysics();

	nau::scene::Camera *cam = NAU->getActiveCamera ();
	cam->setPropb(Camera::DYNAMIC, true);
	m_Canvas->setCamera();
*/}


void 
FrmMainFrame::OnKeyDown(wxKeyEvent & event) {

	event.Skip();
}


void 
FrmMainFrame::OnKeyUp(wxKeyEvent & event) {

	event.Skip();
}


void 
FrmMainFrame::OnBreakResume(wxCommandEvent& event) {

	m_Canvas->BreakResume();
	if (m_Canvas->IsPaused()){
		
//#ifdef GLINTERCEPTDEBUG
//		FreezeGLI();
//#endif
		debugMenu->Enable(idMenuDlgStep, true);
		debugMenu->SetLabel(idMenuDbgBreak, "Resume");
	}
	else{

//#ifdef GLINTERCEPTDEBUG		
//		gliSetIsGLIActive(true);
//#endif
		debugMenu->Enable(idMenuDlgStep, false);
		debugMenu->SetLabel(idMenuDbgBreak, "Pause");
	}
}


//void
//FrmMainFrame::FreezeGLI(){
//#ifdef GLINTERCEPTDEBUG
//	gliSetIsGLIActive(false);
//#endif
//}


void
FrmMainFrame::OnDlgDbgTraceRead(wxCommandEvent& event) {

	DlgTrace::Instance()->Show(TRUE);
}


void
FrmMainFrame::OnDlgDbgProgram(wxCommandEvent& event) {

	DlgDbgPrograms::Instance()->Show(TRUE);
}


void
FrmMainFrame::OnDlgBuffers(wxCommandEvent& event) {

	DlgDbgBuffers::Instance()->updateDlg();
	DlgDbgBuffers::Instance()->Show(TRUE);
}


void 
FrmMainFrame::OnDlgRenderTargets(wxCommandEvent & event) {

	DlgRenderTargets::Instance()->Show(TRUE);
}


void
FrmMainFrame::OnDlgRenderer(wxCommandEvent & event) {

	DlgRenderer::Instance()->Show(TRUE);
}


void
FrmMainFrame::OnDlgDbgStep(wxCommandEvent& event) {

	DlgDbgStep::Instance()->Show(TRUE);
}


void
FrmMainFrame::OnDlgPhysics (wxCommandEvent &event) {

	DlgPhysics::Instance()->Show(TRUE);
}


void
FrmMainFrame::OnPhysicsMode (wxCommandEvent &event) {

	nau::scene::Camera *cam = NAU->getActiveCamera ();

	if (idMenuPhysicsOn == event.GetId()) {
		m_pRoot->enablePhysics();		
	}
	
	if (idMenuPhysicsOff == event.GetId()) {
		m_pRoot->disablePhysics();	
	}
}


/*
---------------------------------------------- CUT HERE (OLD CODE) ----------------------------------------------------------
*/



//FrmMainFrame::OnOctreeBuild(wxCommandEvent& event) {
//
//	if (0 != m_pRoot)
//		RENDERMANAGER->buildOctrees();
//}
//
//
//void
//FrmMainFrame::OnOctreeCompile(wxCommandEvent& event) {
//
//	if (0 != m_pRoot)
//		RENDERMANAGER->compile();
//}
//
//
//void FrmMainFrame::compile(IScene *scene) {
//
//	scene->compile();
//}
//
//
//void
//FrmMainFrame::OnOctreeWrite(wxCommandEvent& event) {
//
//	if (0 != m_pRoot) {
//		wxFileDialog *saveOctDlg =
//			new wxFileDialog(this, _("Save as NBO"), _(""), _(""), _(""), wxFD_SAVE, wxDefaultPosition);
//
//		if (wxID_OK == saveOctDlg->ShowModal()) {
//			m_pRoot->writeAssets("NBO", (const char *)(saveOctDlg->GetPath()).c_str(), "MainScene");
//		}
//	}
//}






//void
//FrmMainFrame::LoadDebugData() {
//
//	DlgTrace::Instance()->loadLog();
//	DlgDbgPrograms::Instance()->clear();
//	DlgDbgStep::Instance()->updateDlg();
//}



//void
//FrmMainFrame::OnModelAppend (wxCommandEvent& event) {
//
//	static const wxChar *fileTypes = _T("3D Files (*.nbo, *.3ds, *.dae, *.obj, *.xml, *.blend, *.ply, *.lwo, *.stl, *.cob, *.scn)|*.nbo;*.3ds;*.dae;*.obj;*.xml;*.blend;*.ply;*.lwo;*.stl;*.cob;*.scn|CBO files (*.nbo)|*.nbo|COLLADA files (*.dae)|*.dae|3DS files (*.3ds)|*.3ds|OBJ files (*.obj)|*.obj|Ogre XML Meshes (*.xml)|*.xml|Blender files (*.blend)|*.blend|Stanford Polygon Library (*.ply)|*.ply|Lightwave (*.lwo)|*.lwo|Stereolithography (*.stl)|*.stl|True Space Obj (*.cob)|*.cob|True Space Scene (*scn)|*.scn");
//	wxFileDialog *openFileDlg = new wxFileDialog(this, _("Open File"), _(""), _(""), fileTypes, wxFD_OPEN, wxDefaultPosition);
//
//	if (wxID_OK == openFileDlg->ShowModal ()) {
//		wxStopWatch aTimer;
//		aTimer.Start();
//		wxString path = openFileDlg->GetPath ();
//
//		try {
//			DlgLog::Instance()->updateDlg();
//			DlgLog::Instance()->clear();
//			m_pRoot->appendModel (std::string(path.mb_str()));
//			m_Canvas->setCamera();
//			updateDlgs();
//#ifndef FINAL
//			float t =  aTimer.Time()/1000.0;
//			SLOG("Elapsed time: %f", t);
//#endif		
//		}
//		catch (std::string &s) {
//			wxMessageBox((wxChar *)s.c_str());
//		}
//	}
//}






//void
//FrmMainFrame::initScene (void)
//{
	//m_CurrentProject->addPass(); /***MARK***/
	//m_CurrentProject->setModel (0, static_cast<CEditorModel*>(&m_pRoot->getScene())); /***MARK***/

	/****Scene Initialization****/

	//RENDERMANAGER->createScene ("MainScene");

	//nau::scene::Camera *aCamera = RENDERMANAGER->getCamera ("MainCamera"); //new nau::scene::Camera ("MainCamera");

	//nau::render::Viewport* MainViewport = m_pRoot->createViewport();
	//MainViewport->setBackgroundColor (nau::math::vec4 (0.0f, 0.0f, 0.0f, 1.0f));

	//aCamera->setViewport (MainViewport);
	//aCamera->setPerspective (60.0f, 0.5f, 10000.0f);

	//aCamera->setCamera (vec3 (-2.0f, 18.0f, 1.0f), vec3 (0.0f, 0.0f, -1.0f), vec3 (0.0f, 1.0f, 0.0f));

	//nau::scene::Light* sunLight = new nau::scene::Light ("Sun", vec3 (150.0f, 150.0f, 0.0f),
	//																		vec3 (-1.0f, 1.0f, 0.0f),
	//																		vec4 (0.9f, 0.9f, 0.9f, 0.0f), 
	//																		nau::scene::Light::DIRECTIONAL);

	//m_pRoot->getScene().addLight (sunLight);

	//nau::scene::Light* sunLight = RENDERMANAGER->getLight ("Sun");
	//sunLight->init (vec3 (150.0f, 150.0f, 0.0f),
	//				vec3 (-1.0f, 1.0f, 0.0f),
	//				vec4 (0.9f, 0.9f, 0.9f, 0.0f),
	//				nau::scene::Light::DIRECTIONAL);

	/***Material Lib Setup***/
	//MATERIALLIBMANAGER->getLib ("model");
	//MATERIALLIBMANAGER->setActiveLib ("model");

	/*** Pipeline Initialization ***/

	//nau::render::Pipeline* pip = RENDERMANAGER->getPipeline ("FixedFunction");
	//nau::render::Pass* p = pip->createPass (/*"fogwithcaustics"*/); //Adds default pass to pipeline	
	//p->setDoColorClear (true);
	//p->setDoDepthClear (true); 
	//p->addScene ("MainScene");
	//p->setCamera ("MainCamera");


	/*** Second pipeline ***/
	//pip = RENDERMANAGER->getPipeline ("Shadow32F");
	//p = pip->createPass ("depthmap");
	//p->setDoColorClear (true);
	//p->setDoDepthClear (true);
	//p->setRTMode (nau::render::ITexture::RGBA32F);
	//p->setFBOs (1);


	//p->setFBOs (1);

	//p = pip->createPass ("quad");
	//p->setClearColor (true);
	//p->setClearDepth (true);name << "(" << program << ", " << location << ")
	//p->setMaterialLib ("model");

	//nau::material::Material *mat = new nau::material::Material;
	//mat->setName (p->getName() + "#quad");
	//mat->attachTexture (0, "FixedFunction#Pass1#rt#color0");
	//MATERIALLIBMANAGER->addMaterial (mat);

	/*RENDERMANAGER->setActivePipeline ("FixedFunction");*/



/***MARK***/ //This needs refactoring! The way camera lights are handled must change
	//nau::scene::Camera *sunCamera = RENDERMANAGER->getCamera ("SunCamera0"); 
	//sunCamera->deactivate();
	////m_pRoot->getScene().addCamera (sunCamera);
	//
	//nau::render::Viewport* sunViewport = m_pRoot->createViewport();
	//sunViewport->setBackgroundColor (nau::math::vec4 (0.0f, 0.0f, 0.0f, 1.0f));

	//sunCamera->setViewport (sunViewport);

	//sunCamera = RENDERMANAGER->getCamera ("SunCamera1"); 
	//sunCamera->deactivate();
	//sunCamera->setViewport (sunViewport);

	//sunCamera = RENDERMANAGER->getCamera ("SunCamera2"); 
	//sunCamera->deactivate();
	//sunCamera->setViewport (sunViewport);

	//sunCamera = RENDERMANAGER->getCamera ("SunCamera3"); 
	//sunCamera->deactivate();
	//sunCamera->setViewport (sunViewport);


	//nau::scene::Camera* quadCamera = RENDERMANAGER->getCamera ("QuadCamera");// new nau::scene::Camera ("QuadCamera");
	//quadCamera->deactivate();
	////m_pRoot->getScene().addCamera (quadCamera);


	//nau::render::Viewport* quadViewport = m_pRoot->createViewport();
	//quadViewport->setBackgroundColor (nau::math::vec4 (0.0f, 0.0f, 0.0f, 1.0f));
	//quadViewport->setSize (512, 512);
	//quadViewport->setFixed (true);

	//quadCamera->setViewport (MainViewport);
/***MARK***/

	//m_Canvas->setCamera (aCamera); /***MARK***/
//}

/*
void
FrmMainFrame::OnIdle (wxIdleEvent& event)
{
	if (0 != m_Canvas) {
		m_Canvas->Render();
		event.Skip();
	}
}
*/



/*
void 
FrmMainFrame::OnFileLoad(wxCommandEvent& event)
{


	static const wxChar *fileTypes = _T( "XML files|*.dae|"
									"All files|*.*|"
									);

	wxFileDialog *openFileDlg = new wxFileDialog (this, _("Open File"), _(""), _(""), fileTypes, wxOPEN, wxDefaultPosition);

	if (wxID_OK == openFileDlg->ShowModal ()) {
	  SetStatusText (openFileDlg->GetPath ());
	  m_CurrentProject = CProject::Instance();

	  m_CurrentProject->setCamera(new CCamera(), new CProjection(), new CViewport());
	  
	  if (0 != m_canvas){
			delete m_canvas;
			m_canvas = 0;
		}

		m_canvas = new CGlCanvas (this);
		SetClientSize (1024, 1024);
	}

	CResourceManager &manager = CResourceManager::getInstance ();
	wxString path = openFileDlg->GetPath ();

	wxFilename file (path);

	wxString url(wxFileSystem::FilenameToURL (file));

	bool result = false;
	
	CTextureManager *tm = CTextureManager::Instance();
	tm->setPath((openFileDlg->GetDirectory()).fn_str());
	
	try{
		if (file.GetExt() == "dae"){
			result = manager.loadWorld ((const char *) url.fn_str ()); 
		} else if (file.GetExt() == "oct") {
			result = manager.loadWorld ((const char*) file.GetFullPath().fn_str());
		}
	} catch(nau::CWorldFactoryError exception) {
		LOG_CRITICAL("Exception: %s", (exception.getException()).c_str());
	} 
	
	
	if (false == result){
		wxMessageBox(_("Unable to load: " + openFileDlg->GetFilename ()), _("Error"));
	} else {


		IWorld& world = CResourceManager::getInstance().getWorld();
		IRenderer &render = CRenderManager::getInstance().getRenderer();

		world.prepare();
		
		//Create the project to render materials

		m_CurrentProject->addPass();
		m_CurrentProject->setModel (0, static_cast<CEditorModel*>(&world));

		//build materials menu
		std::vector<std::string>& materialsNames = render.getMaterialsNamesList ();
		std::vector<std::string>::iterator materialsNamesIter = materialsNames.begin ();
	
		if (materialsNamesIter != materialsNames.end()){
			long int initId = wxNewId();
			long int finalId = initId;
			
			materialsMenu->AppendCheckItem(finalId, wxString((*materialsNamesIter).c_str(), wxConvUTF8));
			materialsMenu->Check(finalId, true);

			while (++materialsNamesIter != materialsNames.end()){
				finalId = wxNewId();
				materialsMenu->AppendCheckItem(finalId, wxString((*materialsNamesIter).c_str(), wxConvUTF8));
				materialsMenu->Check(finalId, true);
			}
			
			this->Connect (initId, finalId, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler(FrmMainFrame::OnMaterialSelect));
		}
	}
}
*/

//void
//FrmMainFrame::setupMaterials (void)
//{

	//nau::material::Material* aMat = 0;

	///*** Vidro ***/
	//aMat = MATERIALLIBMANAGER->getMaterial ("Vidro");

	//aMat->setPriority (1);
	//aMat->setTransparent (true);
	//
	//aMat->OGLstate.m_glBlend = 1;
	//aMat->OGLstate.m_glBlendSrc = GL_SRC_ALPHA;
	//aMat->OGLstate.m_glBlendDst = GL_ONE_MINUS_SRC_ALPHA;

	//const float* diffuse = aMat->getColor().getDiffuse();
	//
	//aMat->getColor().setDiffuse (diffuse[0], diffuse[1], diffuse[2], 0.75f);		


	///*** Agua Compose ***/

	//aMat = MATERIALLIBMANAGER->getMaterial ("AguaComposer");
	//aMat->setTransparent (true);
	//aMat->setPriority (1);
	//aMat->OGLstate.m_glBlend = 1;
	//aMat->OGLstate.m_glBlendSrc = GL_SRC_ALPHA;
	//aMat->OGLstate.m_glBlendDst = GL_ONE_MINUS_SRC_ALPHA;

	//diffuse = aMat->getColor().getDiffuse();

	//aMat->getColor().setDiffuse (diffuse[0], diffuse[1], diffuse[2], 0.75f);				  

	///*** Grade ***/
	//std::vector<std::string> *materials = MATERIALLIBMANAGER->getMaterialNames ("model");

	//for (int i = 0; i < materials->size(); i++) {
	//	/*** All type of 'Grade' ***/
	//	int pos = materials->at (i).find ("Grade");
	//	if (std::string::npos != pos) {
	//		aMat = MATERIALLIBMANAGER->getMaterial (materials->at (i));
	//		aMat->setTransparent (true);
	//		aMat->setPriority (2);
	//		aMat->OGLstate.m_glBlend = 1;
	//		aMat->OGLstate.m_glBlendSrc = GL_SRC_ALPHA;
	//		aMat->OGLstate.m_glBlendDst = GL_ONE_MINUS_SRC_ALPHA;
	//	}
	//}

	//delete materials;
//}

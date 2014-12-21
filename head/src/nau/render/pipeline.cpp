#include <nau/render/pipeline.h>

#include <nau/config.h>
#include <nau.h>
#include <nau/render/rendermanager.h>

#include <nau/geometry/frustum.h>
#include <nau/render/passfactory.h>

#include <nau/slogger.h>

#include <nau/debug/profile.h>


#ifdef GLINTERCEPTDEBUG
#include <nau/loader/projectloaderdebuglinker.h>
#endif //GLINTERCEPTDEBUG

using namespace nau::geometry;
using namespace nau::render;
using namespace nau::scene;

Pipeline::Pipeline (std::string pipelineName) :
	m_Name (pipelineName),
	m_Active (true),
	m_Passes(0),
	m_DefaultCamera(""),
	m_CurrentPass(0),
	m_NextPass(0)
{

}

std::string 
Pipeline::GetName(){
	return m_Name;
}


const std::string &
Pipeline::getLastPassCameraName() 
{
	// There must be at least one item in the passes queue
	assert(m_Passes.size() > 0);
	// The last item can't be NULL
	assert(m_Passes.at(m_Passes.size()-1) != NULL);

	return(m_Passes.at(m_Passes.size()-1)->getCameraName());
}


const std::string &
Pipeline::getDefaultCameraName()
{
	if (m_DefaultCamera == "")
		return(m_Passes.at(m_Passes.size()-1)->getCameraName());
	else
		return m_DefaultCamera;
}


void
Pipeline::setDefaultCamera(const std::string &defCam)
{
	// The camera must be defined in the render manager
	assert(defCam == "" || RENDERMANAGER->hasCamera(defCam));

	m_DefaultCamera = defCam;
}


int 
Pipeline::getNumberOfPasses()
{
	return (int)m_Passes.size();
}


std::vector<std::string> * 
Pipeline::getPassNames(){

	std::vector<std::string> *names = new std::vector<std::string>; 

	for( std::deque<nau::render::Pass*>::iterator iter = m_Passes.begin(); iter != m_Passes.end(); ++iter ) {
      names->push_back((*iter)->getName()); 
    }
	return names;
}


void 
Pipeline::addPass (Pass* aPass, int PassIndex)
{
	// Pass index must be valid
	assert(PassIndex > -2 && PassIndex < (int)m_Passes.size());

	if (PassIndex < -1) {
		return;
	}

	switch (PassIndex) {
  
		case -1: 
			m_Passes.push_back (aPass);
			break;
		case 0:
		    m_Passes.push_front (aPass);
			break;
		default:
			unsigned int pos = static_cast<unsigned int>(PassIndex);
			if (pos < m_Passes.size()) {
				m_Passes.insert (m_Passes.begin() + pos, aPass);
			}
	}
}


Pass* 
Pipeline::createPass (const std::string &name, const std::string &passType) 
{
	// name must not be empty
	assert(name != "");
	// type must also be a valid pass class
	assert(PassFactory::isClass(passType));

	std::stringstream s;

	s << m_Name;
	s << "#" << name;

	Pass *pass = PassFactory::create (passType, s.str());
	m_Passes.push_back(pass);

	return pass;
}


bool
Pipeline::hasPass(const std::string &passName)
{
	std::deque<Pass*>::iterator passIter;
	passIter = m_Passes.begin();

	for ( ; passIter != m_Passes.end(); ++passIter) {
		if ((*passIter)->getName() == passName) {
			return true;
		}
	}
	return false;
}


Pass* 
Pipeline::getPass (const std::string &passName)
{
	// Pass must exist
	assert(hasPass(passName));

	std::deque<Pass*>::iterator passIter;
	passIter = m_Passes.begin();

	for ( ; passIter != m_Passes.end(); passIter++) {
		if ((*passIter)->getName() == passName) {
			return (*passIter);
		}
	}
	return 0;
}


Pass* 
Pipeline::getPass (int n)
{
	// n must be with range
	assert(n < (int)m_Passes.size());

	return m_Passes.at (n);
}


const std::string &
Pipeline::getCurrentCamera()
{
	// pipeline must be in execution
	assert(m_CurrentPass != NULL);

	return(m_CurrentPass->getCameraName());
}


int 
Pipeline::getPassCounter() {

	return m_NextPass;
}

void
Pipeline::execute() {

	//For each pass....

	try {
		PROFILE("Pipeline execute");

		for ( auto pass:m_Passes) {
#ifdef GLINTERCEPTDEBUG
			addMessageToGLILog(("\n#NAU(PASS,START," + pass->getName() + ")").c_str());
#endif //GLINTERCEPTDEBUG

			PROFILE (pass->getName());
			m_CurrentPass = pass;
			RENDERER->setDefaultState();			
			pass->prepare();

			if (true == pass->renderTest()) {	

				pass->doPass();
			}
			pass->restore();
#ifdef GLINTERCEPTDEBUG
			addMessageToGLILog(("\n#NAU(PASS,END," + pass->getName() + ")").c_str());
#endif //GLINTERCEPTDEBUG
		}
	}
	catch (Exception &e) {
		SLOG(e.getException().c_str());
	}
}


void 
Pipeline::executeNextPass() {

	try {
		Pass *p = m_Passes[m_NextPass];
		p->prepare();
		if (true == p->renderTest())
			p->doPass();
		p->restore();

		m_NextPass++;
		if (m_NextPass == m_Passes.size())
			m_NextPass = 0;
	}
	catch (Exception &e) {
		SLOG(e.getException().c_str());
	}
}


//unsigned char
//Pipeline::executePass (unsigned int p)
//{
//	//For each pass....
//
//	try {
//		PROFILE("Pipeline execute");
//
//		//std::deque<Pass*>::iterator passIter;
//		//passIter = m_Passes.begin();
//
//		//for ( ; passIter != m_Passes.end(); passIter++) {
//
//		//PROFILE ((*passIter)->getName().c_str());
//		//m_CurrentPass = *passIter;
//		//RENDERER->setDefaultState();			
//		//(*passIter)->prepare();
//
//		//if (true == (*passIter)->renderTest()) {	
//
//		//	(*passIter)->doPass();
//		//}
//		//(*passIter)->restore();
//		//}
//		{
//			PROFILE(m_Passes[m_NextPass]->getName().c_str());
//			m_CurrentPass = m_Passes[m_NextPass];
//			RENDERER->setDefaultState();
//			m_Passes[m_NextPass]->prepare();
//
//			if (true == m_Passes[m_NextPass]->renderTest()) {
//
//				m_Passes[m_NextPass]->doPass();
//			}
//			m_Passes[m_NextPass]->restore();
//		}
//
//		m_NextPass++;
//
//		if (m_NextPass == 1){
//			if (m_NextPass >= m_Passes.size()){
//				m_NextPass = 0;
//				m_CurrentPass = m_Passes[m_NextPass];
//				return PIPE_PASS_STARTEND;
//			}
//			else{
//				m_CurrentPass = m_Passes[m_NextPass];
//				return PIPE_PASS_START;
//			}
//		} 
//		else if (m_NextPass >= m_Passes.size()){
//			m_NextPass = 0;
//			m_CurrentPass = m_Passes[m_NextPass];
//			return PIPE_PASS_END;
//		}
//
//	}
//	catch(Exception &e) {
//		SLOG(e.getException().c_str());
//	}
//
//	m_CurrentPass = m_Passes[m_NextPass];
//	return PIPE_PASS_MIDDLE;
//}


Pass *
Pipeline::getCurrentPass() {
	if (m_Passes.size() > m_NextPass){
		m_CurrentPass = m_Passes[m_NextPass];
	}
	return m_CurrentPass;
}

/*
---------------------------------------------- CUT HERE (OLD CODE) ----------------------------------------------------------
*/


//void CProject::setModel(int pass, CEditorModel *model) {
//
//	m_passes[pass]->m_model = model;
//	m_passes[pass]->m_model->m_materialLib = m_materialLibManager;
//	m_passes[pass]->m_model->mapMaterials(pass);
//	buildMaterialList(pass);
//	m_passes[pass]->setup();
//	setRTLabels(pass);
//
//}
//
//
//void CProject::addPass(std::string file) {
//
//	std::string p,f;
//	p = CFilename::GetPath(file);
//	f = CFilename::GetFilename(file);
//	CPass *pass = new CPass(p,f );
//	m_passes.push_back(pass);
//
//	m_activeCameraIndex = 0;
//	m_defaultCamera = pass->m_camera;
//
//	int passNumber = m_passes.size() - 1;
//	pass->m_model->m_materialLib = m_materialLibManager;
//	pass->m_model->mapMaterials(passNumber);
//	buildMaterialList(passNumber);
//	pass->setup();
//	setRTLabels(passNumber);
//}
//
//CTextureDevIL *CProject::getTexture(int pass, int rt) {
//
//	return &(m_passes[pass]->m_renderTargets[rt]);
//}
//
//void CProject::setFBOs(int pass, int fbos) {
//	
//	m_passes[pass]->m_color = fbos;
//	m_passes[pass]->m_useRT = 1;
//	m_passes[pass]->setup();
//	setRTLabels(pass);
//}
//
//void CProject::setRTLabels(int passNumber) {
//
//	std::stringstream numpass;
//	std::string auxpass;
//
//	numpass << passNumber;
//	numpass >> auxpass;
//	for (int i = 0; i < m_passes[passNumber]->m_color; i++) {
//		std::stringstream numrt;
//		std::string auxrt;
//		numrt << i;
//		numrt >> auxrt;
//		std::string s = "Pass #" + auxpass + ": RT " + auxrt;
//		m_passes[passNumber]->m_renderTargets[i].m_label = s;
//		m_passes[passNumber]->m_renderTargets[i].m_width = m_passes[passNumber]->m_rtSize;
//		m_passes[passNumber]->m_renderTargets[i].m_height = m_passes[passNumber]->m_rtSize;
//		m_passes[passNumber]->m_renderTargets[i].m_depth = 1;
//	}
//		
//}
//
//void CProject::setModel(int pass, std::string path, std::string filename) {
//
//	m_passes[pass]->m_model = CEditorModel::loadModel(path,filename);
//	m_passes[pass]->m_model->m_materialLib = m_materialLibManager;
//	m_passes[pass]->m_model->mapMaterials(pass);
//	buildMaterialList(pass);
//}
//
//void CProject::drawAxis() {
//
//	float red[4] = {1,0,0,1};
//	float green[4] = {0,1,0,1};
//	float blue[4] = {0,0,1,1};
//	float black[4] = {0,0,0,1};
//	float lgrey[4] = {0.7f,0.7f,0.7f,1.0f};
//
//	glPushAttrib(GL_LIGHTING);
//	glDisable(GL_LIGHTING);
//
//	glColor3fv(red);
//	glBegin(GL_LINES);
//		glVertex3f(0,0,0);
//		glVertex3f(10,0,0);
//	glEnd();
//
//	glColor3fv(green);
//	glBegin(GL_LINES);
//		glVertex3f(0,0,0);
//		glVertex3f(0,10,0);
//	glEnd();
//
//	glColor3fv(blue);
//	glBegin(GL_LINES);
//		glVertex3f(0,0,0);
//		glVertex3f(0,0,10);
//	glEnd();
//
//	glPopAttrib();
//
//}
//
//void CProject::setInputs(int pass,int t,int id) {
//
//	if (id != 0) {
//		int auxp = (id-1) / 4;
//		int auxt = (id-1) % 4;
//		m_passes[pass]->m_inputTexID[t] = m_passes[auxp]->m_texId[auxt-1];
//	}
//	else
//		m_passes[pass]->m_inputTexID[t] = 0;
//
//	m_passes[pass]->setInputs(t,id);
//
//}
//
//void CProject::render() {
//
//	int first=1;
//
//	CProgram::FixedFunction();
//	m_glCurrState.setDiff(*m_glDefState);
////	CTexture::NoTexture();
//
//	//tb.setViewport(0,0,m_defaultViewport->m_viewport[2],m_defaultViewport->m_viewport[3]);
//
//	std::vector<CPass *>::iterator it;
//	CPass *pass;
//	for(it = m_passes.begin(); it != m_passes.end() ; it++ ) {
//		
//		pass = (CPass *)*it;
//
//
//		// defines the viewport
//		//if (pass->m_color || pass->m_depth)
//		//	glViewport(0,0,pass->m_rtSize,pass->m_rtSize);
//		//else
//		//	m_defaultViewport->use();
//
//		// defines the projection
//		//pass->m_projection->m_ratio = m_defaultProjection->m_ratio;
//		//pass->m_projection->use();
//
//
//		//glLoadIdentity();
//
//		//int i;
//		//for(i=0; i < 8;i++) {
//		//	if (lightWhere[i] == 0)
//		//		glLightfv(GL_LIGHT0+i, GL_POSITION, lightPos[i]);
//		//}
//
//		//pass->m_camera->use();
//		
//		//if (pass->m_camera == m_defaultCamera)
//		//	tb.cameraUpdate();
//
//		//for(i=0; i < 8;i++) {
//		//	if (lightWhere[i] == 1)
//		//		glLightfv(GL_LIGHT0+i, GL_POSITION, lightPos[i]);
//		//}
//		
//		//if (pass->m_camera == m_defaultCamera)
//		//	tb.use(;)
//
//		//for(i=0; i < 8;i++) {
//		//	if (lightWhere[i] == 2)
//		//		glLightfv(GL_LIGHT0+i, GL_POSITION, lightPos[i]);
//		//}
//
//// TESTING
//
///*	if (first) {
//		glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
//		first = !first;
//	}
//	else {
//		glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
//		first = !first;
//	}
//*/
//// END TESTING
//
//		//glutSolidTeapot(1);
//		(*it)->render();
//
//	}
//}


//void CProject::initState(COGLState *glState) {
//
//	m_glDefState = glState;
//}
//
//
//
//int CProject::getNumMaterials() {
//
//	return(m_mats.size());
//}
//
//CMaterial *CProject::getMaterial(int i) {
//
//	return(m_mats[i]);
//}
//
//std::string CProject::getMaterialName(int i) {
//
//	return(m_mats[i]->getName());
//}
//
//void CProject::save( std::string name) {
//
//
//}


//void CProject::buildMaterialList(int pass) {
//
//	CEditorModel *m;
//	int numMat;
//	std::string s;
//	std::stringstream aux;
//
//	CMaterialLib *ml;
//	ml = new CMaterialLib();
//	m = m_passes[pass]->m_model;
//	numMat = m->getNumMaterials();
//	for(int i=0; i < numMat; i++) 
//		ml->add(m->getMaterial(i));
//
//	aux << pass;
//	aux >> s;
//	s = " Pass #" + s;
//	ml->m_filename = s;
//	m_materialLibManager->addLib(ml);
//
//	/*
//	m_mats.clear();
//	std::vector<CPass *>::iterator it;
//	for(it = m_passes.begin(); it != m_passes.end() ; it++ ) {
//		m = (*it)->m_model;
//		numMat = m->getNumMaterials();
//		for(int i=0; i < numMat; i++) 
//			m_mats.push_back((m->getMaterial(i)));
//	}
//	*/
//		
//}


//void CProject::setMaterialLibManager(CMaterialLibManager *mlm) {
//
//	m_materialLibManager = mlm;
//
//}
//
//
//int CProject::materialInUse(CMaterialID &mid) {
//
//	for (int i = 0; i - m_passes.size(); i++) {
//		
//		if (m_passes[i]->m_model != NULL)
//			if (m_passes[i]->m_model->materialInUse(mid))
//				return 1;
//	}
//	return 0;
//}
//
//void CProject::save() {
//
//	int i;
//	std::string filename = m_path + "/" + m_filename;
//	std::ofstream outf(filename.c_str());
//
//	outf << "PROJECT\n";
//
//	// project data 
//	outf << "PROPERTIES\n";
//	outf << "ACTIVE CAMERA PASS=" << m_activeCameraIndex << "\n";
//	outf << "END PROPERTIES\n";
//
//	// LIGHTS
//	outf << "LIGHTS\n";
//
//	for (i = 0; i < 8; i++) {
//		outf << "LIGHT #" << i << "\n";
//		outf << "	POSITION=" << lightPos[i][0] << "," << lightPos[i][1] << ","  
//								<< lightPos[i][2] << ","  << lightPos[i][3] << "\n";
//		outf << "	SPOTDIR=" << spotDir[i][0] << "," << spotDir[i][1] << "," << spotDir[i][2] << "\n";
//		outf << "	LIGHTWHERE=" << lightWhere[i] << "\n";
//		outf << "END LIGHT\n";
//	}
//	outf << "END LIGHTS\n";
//
////	saveLibs(outf);
//
//	outf << "PASSES\n";
//	CPass *pass;
//	for (i = 0 ; i < m_passes.size(); i++) {
//		outf << "PASS #" << i << "\n";
//		pass = m_passes[i];
//		pass->save(outf, m_path);
//	}
//	outf << "END PASSES\n";
//
//	outf << "MATERIAL LIBS\n";
//	
//	std::vector<std::string> *libNames = m_materialLibManager->getLibNames();
//	std::vector<std::string>::iterator iter;
//	std::string libName;
//	CMaterialLib *matLib;
//    for(iter = libNames->begin(); iter != libNames->end(); ++iter)
//    {
//		libName = iter->c_str();
//		if (libName.substr(0,1) == " ") { // Its a Pass Lib
//			matLib = m_materialLibManager->getLib(libName);
//			outf << "LIB\n";
//			outf << "NAME= " << libName << "\n";
//			matLib->save(outf,m_path);
//			outf << "END LIB\n";
//		}
//    }
//
//	outf << "END MATERIAL LIBS\n";
//
//	std::set<std::string> libs;
//	std::set<std::string>::iterator iter2;
//	for (i = 0 ; i < m_passes.size(); i++) {
//
//		for (int j = 0; j < m_passes[i]->m_model->mapMat.size(); j++) {
//			libs.insert(m_passes[i]->m_model->mapMat[j]->libName);
//		}
//	}
//
//	if (libs.size() > 0) {
//
//		outf << "USE LIBS\n";
//		std::string lib;
//		for(iter2 = libs.begin(); iter2 != libs.end(); iter2++) {
//			lib = iter2->c_str();
//			if (lib.substr(0,1) != " ")
//				outf << "LIB= " << lib << "\n";
//		}
//
//		outf << "END USE LIBS\n";
//
//	}
//	outf << "END PROJECT\n";
//
//	outf.close();
//}
//
//
//void CProject::open(std::string path, std::string file) {
//
//
//	freeProject();
//
//	std::string text;
//	char textc[256],name [256],line[256];
//	m_path = path;
//	m_filename = file;
//	std::string filename = m_path + "/" + m_filename;
//	std::ifstream inf(filename.c_str());
//	CMaterialID mid;
//	CMaterialLib *matLib;
//	CMaterial *mat;
//
//	do { 
//		inf.getline(textc,256);
//		text = textc;
//		
//		if (text == "PROJECT") {
//			// do nothing
//		}
//		else if (text == "LIGHTS") {
//			readLights(inf);
//		}
//		else if (text == "PASSES") {
//			readPasses(inf);
//			setActiveCameraIndex(m_passes.size()-1);
//		} 
//		else if (text == "MATERIAL LIBS") {
//
//			inf.getline(line,256); // LIB
//			text = line;
//			do {
//
//
//				inf.getline(line,256); // NAME=
//				text = line;
//				mid.libName = text.substr(6);
//				matLib = m_materialLibManager->getLib(mid.libName);
//
//				inf.getline(line,256);
//				text = line;
//
//				do { // read materials
//
//					inf.getline(line,256);
//					text = line;
//					sscanf(line,"NAME= %s",name);
//					mid.matName = name;
//					mat = matLib->getMaterial(mid.matName);
//					mat->loadTo(inf,m_path);
//
//					inf.getline(line,256);
//					text = line;
//
//				} while (text != "END LIB");
//
//				inf.getline(line,256); // LIB
//				text = line;
//			}
//			while (text != "END MATERIAL LIBS");
//		}
//
//		else if (text == "USE LIBS") {
//			std::string filename;
//			inf.getline(line,256);
//			text = line;
//			do {
//				sscanf(line,"LIB= %s",name);
//				filename = name;
//				filename += ".spl";
//				m_materialLibManager->load(filename, path);
//
//				inf.getline(line,256);
//				text = line;
//
//			}while (text != "END USE LIBS");
//		}
//		else if (text == "PROPERTIES") {
//
//
//		}
//		/*
//		else {
//			do {
//				inf.getline(textc,256);
//				text = textc;
//			}
//			while (text.substr(0,3) != "END");
//			inf.getline(textc,256);
//			text = textc;
//		}*/
//
//	} while (text != "END PROJECT");
//}
//
//
//void CProject::readPasses(std::ifstream &inf) {
//
//	std::string text,item;
//	char textc[256];
//	int i;
//	CPass *pass;
//
//	inf.getline(textc,256); // should be "PASS #i"
//	do {
//		sscanf(textc,"PASS #%d",&i);
//
//		pass = new CPass();
//		
//		pass->load(inf,m_path);
//
//		m_passes.push_back(pass);
//		pass->m_model->m_materialLib = m_materialLibManager;
//		pass->m_model->mapMaterials(i);
//		buildMaterialList(i);
//		setRTLabels(i);
//
//		inf.getline(textc,256);
//		text = textc;
//
//	} while (text != "END PASSES");
//}
//
//
//
//
//
//
//void CProject::readLights(std::ifstream &inf) {
//	
//	std::string text,item;
//	char textc[256];
//	int i;
//
//	inf.getline(textc,256); // should be "LIGHT #i"
//	do {
//		sscanf(textc,"LIGHT #%d",&i);
//		do {
//			inf.getline(textc,256);
//			text = textc;
//			item = text.substr(1,text.find("=",1)-1);
//			if (item == "POSITION")
//				sscanf(text.c_str(),"\tPOSITION=%f,%f,%f,%f",&lightPos[i][0],&lightPos[i][1],
//															&lightPos[i][2],&lightPos[i][3]);
//			else if (item == "SPOTDIR")
//				sscanf(text.c_str(),"\tSPOTDIR=%f,%f,%f",&spotDir[i][0],&spotDir[i][1],
//															&spotDir[i][2]);
//			else if (item == "LIGHTWHERE")
//				sscanf(text.c_str(),"\tLIGHTWHERE=%d",&lightWhere[i]);
//
//		} while (text != "END LIGHT");
//
//		inf.getline(textc,256);
//		text = textc;
//
//	} while (text != "END LIGHTS");
//}


//void CProject::freeProject() 
//{
//
//
//}
//
//CProject* CProject::Instance () {
//	static CProject inst;
//	return &inst;
//}

//void 
//Pipeline::setCamera(CCamera *cam,CProjection *proj,CViewport *vp) {
//	
//	m_defaultCamera = cam;
//	m_defaultProjection = proj;
//	m_defaultViewport = vp;
//}
//
//void CProject::newProject(std::string path, std::string filename) {
//
//	m_path = path;
//	m_filename = filename;
//
//}
//
//void CProject::setActiveCameraIndex(int i) {
//
//	m_activeCameraIndex = i;
//	m_defaultCamera = m_passes[i]->m_camera;
//}
//
//int CProject::getActiveCameraIndex() {
//
//	return(m_activeCameraIndex);
//}
//
//
//void CProject::setPassCamera(int pass, CCamera *cam) {
//	
//	m_passes[pass]->m_camera = cam;
//	if (pass == m_activeCameraIndex)
//			m_defaultCamera = cam;
//
//}

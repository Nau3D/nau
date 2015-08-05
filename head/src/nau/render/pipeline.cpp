#include "nau/render/pipeline.h"

#include "nau.h"
#include "nau/config.h"
#include "nau/slogger.h"
#include "nau/debug/profile.h"
#include "nau/geometry/frustum.h"
#include "nau/render/passFactory.h"
#include "nau/render/renderManager.h"

#ifdef GLINTERCEPTDEBUG
#include "nau/loader/projectLoaderDebugLinker.h"
#endif 

#include <GL/glew.h>

using namespace nau::geometry;
using namespace nau::render;
using namespace nau::scene;

Pipeline::Pipeline (std::string pipelineName) :
	m_Name (pipelineName),
	m_Passes(0),
	m_DefaultCamera(""),
	m_CurrentPass(0),
	m_NextPass(0),
	m_PreScriptFile(""), 
	m_PreScriptName(""),
	m_PostScriptFile(""),
	m_PostScriptName("") {

}


std::string 
Pipeline::getName() {
	return m_Name;
}


const std::string &
Pipeline::getLastPassCameraName() {

	// There must be at least one item in the passes queue
	assert(m_Passes.size() > 0);
	// The last item can't be NULL
	assert(m_Passes.at(m_Passes.size()-1) != NULL);

	return(m_Passes.at(m_Passes.size()-1)->getCameraName());
}


const std::string &
Pipeline::getDefaultCameraName() {

	if (m_DefaultCamera == "")
		return(m_Passes.at(m_Passes.size()-1)->getCameraName());
	else
		return m_DefaultCamera;
}


void
Pipeline::setDefaultCamera(const std::string &defCam) {

	// The camera must be defined in the render manager
	assert(defCam == "" || RENDERMANAGER->hasCamera(defCam));

	m_DefaultCamera = defCam;
}


void 
Pipeline::setFrameCount(unsigned int k) {

	m_FrameCount = k;
}


unsigned int 
Pipeline::getFrameCount() {

	return m_FrameCount;
}


int 
Pipeline::getNumberOfPasses() {

	return (int)m_Passes.size();
}


std::vector<std::string> * 
Pipeline::getPassNames() {

	std::vector<std::string> *names = new std::vector<std::string>; 

	for( std::deque<nau::render::Pass*>::iterator iter = m_Passes.begin(); iter != m_Passes.end(); ++iter ) {
      names->push_back((*iter)->getName()); 
    }
	return names;
}


void 
Pipeline::addPass (Pass* aPass, int PassIndex) {

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
Pipeline::executePass(Pass *pass) {

	m_CurrentPass = pass;
	pass->callPreScript();
	bool keepRunning = false;

	if (pass->getPrope(Pass::TEST_MODE) == Pass::RUN_WHILE)
		keepRunning = true;

	do {
#ifdef GLINTERCEPTDEBUG
		addMessageToGLILog(("\n#NAU(PASS,START," + pass->getName() + ")").c_str());
#endif //GLINTERCEPTDEBUG

		if (RENDERER->getPropb(IRenderer::DEBUG_DRAW_CALL))
			SLOG("Pass: %s", pass->getName().c_str());

		PROFILE(pass->getName());

		bool run = pass->renderTest();
		if (run) {
			pass->prepare();
			pass->doPass();
			pass->restore();
		}

#ifdef GLINTERCEPTDEBUG
		addMessageToGLILog(("\n#NAU(PASS,END," + pass->getName() + ")").c_str());
#endif //GLINTERCEPTDEBUG

		keepRunning = keepRunning && run;
	} while (keepRunning);
	pass->callPostScript();

}


void
Pipeline::execute() {

	unsigned int n = RENDERER->getPropui(IRenderer::FRAME_COUNT);
	if (m_FrameCount == 0 || n == 0)
		callScript(m_PreScriptName);

	try {
		PROFILE("Pipeline execute");

		RENDERER->setDefaultState();			
		for ( auto pass:m_Passes) {

			int mode = pass->getPrope(Pass::RUN_MODE);
			// most common case: run pass in all frames
			if (mode == Pass::RUN_ALWAYS)
				executePass(pass);

			else {
				unsigned int f = RENDERER->getPropui(IRenderer::FRAME_COUNT);
				bool even = (f % 2 == 0);
				if (mode == Pass::RUN_EVEN && !even)
					continue;
				else if (mode == Pass::RUN_ODD && even)
					continue;
				// check for skip_first and run_once cases
				else if ((mode == Pass::SKIP_FIRST_FRAME && (f == 0)) || (mode == Pass::RUN_ONCE && (f > 0)))
					continue;
				else
					executePass(pass);
			}
		}
	}
	catch (Exception &e) {
		SLOG(e.getException().c_str());
	}

	if (m_FrameCount == 0 || n == m_FrameCount-1)
		callScript(m_PostScriptName);
}


void 
Pipeline::executeNextPass() {

	if (m_NextPass == 0)
		callScript(m_PreScriptName);

	try {
		Pass *p = m_Passes[m_NextPass];
		executePass(p);

		m_NextPass++;
		if (m_NextPass == m_Passes.size())
			m_NextPass = 0;
	}
	catch (Exception &e) {
		SLOG(e.getException().c_str());
	}
	if (m_NextPass == 0)
		callScript(m_PostScriptName);

}


Pass *
Pipeline::getCurrentPass() {
	//if (m_Passes.size() > m_NextPass){
	//	m_CurrentPass = m_Passes[m_NextPass];
	//}
	return m_CurrentPass;
}


// -----------------------------------------------------------------
//		PRE POST SCRIPTS
// -----------------------------------------------------------------

void 
Pipeline::setPreScript(std::string file, std::string name) {

	m_PreScriptFile = file;
	m_PreScriptName = name;
	if (file != "" && name != "")
		NAU->initLuaScript(file, name);
}


void
Pipeline::setPostScript(std::string file, std::string name) {

	m_PostScriptFile = file;
	m_PostScriptName = name;
	if (file != "" && name != "")
		NAU->initLuaScript(file, name);
}


void 
Pipeline::callScript(std::string &name) {

#ifdef NAU_LUA
	if (name != "") {
		NAU->callLuaScript(name);

	}
#endif
}
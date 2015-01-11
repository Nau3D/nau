#include <nau/resource/resourcemanager.h>

#include <nau/geometry/box.h>
#include <nau/geometry/sphere.h>
#include <nau/geometry/square.h>
#include <nau/geometry/axis.h>
#include <nau/geometry/bbox.h>
#include <nau/geometry/meshwithpose.h>
#include <nau/geometry/meshbones.h>

#include <sstream>

using namespace nau::resource;
using namespace nau::render;
using namespace nau::scene;
using namespace nau::geometry;

int ResourceManager::renderableCount = 0;


ResourceManager::ResourceManager (std::string path) :
	m_pTextureManager (0),
	m_Path (path),
	m_RenderTargets(),
	m_Meshes(),
	m_States(),
	m_Programs()
{
	m_pTextureManager = new TextureManager (m_Path);
}


ResourceManager::~ResourceManager(void) {

	/***DESTROY ALL ALOCATED RESOURCES***/

	while (!m_States.empty()) {
	
		m_States.erase(m_States.begin());
	}

	if (0 != m_pTextureManager) {
		delete m_pTextureManager;
	}

	while (!m_Meshes.empty()) {
	
		m_Meshes.erase(m_Meshes.begin());
	}
	ResourceManager::renderableCount = 0;

	while (!m_RenderTargets.empty()) {
	
		m_RenderTargets.erase(m_RenderTargets.begin());
	}

	while (!m_Programs.empty()) {
	
		m_Programs.erase(m_Programs.begin());
	}

	while (!m_Buffers.empty()) {

		m_Buffers.erase(m_Buffers.begin());
	}
}


void
ResourceManager::clear() {

	m_pTextureManager->clear();

	while (!m_States.empty()){
		delete((*m_States.begin()).second);
		m_States.erase(m_States.begin());
	}
	while (!m_Meshes.empty()){
		delete((*m_Meshes.begin()).second);
		m_Meshes.erase(m_Meshes.begin());
	}

	while (!m_RenderTargets.empty()){
		delete((*m_RenderTargets.begin()).second);
		m_RenderTargets.erase(m_RenderTargets.begin());
	}

	while (!m_Programs.empty()){
		delete((*m_Programs.begin()).second);
		m_Programs.erase(m_Programs.begin());
	}
	while (!m_Buffers.empty()){
		delete((*m_Buffers.begin()).second);
		m_Buffers.erase(m_Buffers.begin());
	}
}
//-------------------------------------	

//			TEXIMAGES

//-------------------------------------


nau::material::TexImage* 
ResourceManager::createTexImage(nau::render::Texture *t) {

	return m_pTextureManager->createTexImage(t);
}


nau::material::TexImage* 
ResourceManager::getTexImage(std::string aTextureName) {

	return m_pTextureManager->getTexImage(aTextureName);
}

//-------------------------------------	

//			TEXTURES

//-------------------------------------

bool
ResourceManager::hasTexture(std::string name) {

	return m_pTextureManager->hasTexture(name);
}


//nau::render::Texture*
//ResourceManager::createTexture (std::string label, 
//				std::string internalFormat, 
//				std::string aFormat, 
//				std::string aType, int width, int height,
//				unsigned char* data)
//{
//	return (m_pTextureManager->createTexture (label, internalFormat, aFormat, aType, width, height, data));
//}

nau::render::Texture * 
ResourceManager::createTexture(std::string label) {

	return (m_pTextureManager->createTexture(label));
}


nau::render::Texture*
ResourceManager::createTexture (std::string label, 
				std::string internalFormat, 
				int width, int height, int depth, int layers, int levels, int samples)
{
	return (m_pTextureManager->createTexture (label, internalFormat, width, height, depth, layers, levels, samples));
}


//nau::render::Texture*
//ResourceManager::createTextureMS (std::string label, 
//				std::string internalFormat, 
//				int width, int height,
//				int samples)
//{
//	return (m_pTextureManager->createTextureMS (label, internalFormat, width, height, samples));
//}


nau::render::Texture* 
ResourceManager::addTexture (std::string fn, std::string label, bool mipmap) {

	return (m_pTextureManager->addTexture (fn, label, mipmap));
}


nau::render::Texture* 
ResourceManager::addTexture (std::vector<std::string> fn, std::string label, bool mipmap) {

	return (m_pTextureManager->addTexture (fn, label, mipmap));
}


void 
ResourceManager::removeTexture (std::string name) {

	m_pTextureManager->removeTexture (name);
}


Texture*
ResourceManager::getTexture (std::string name) {

	return (m_pTextureManager->getTexture (name));
}


Texture*
ResourceManager::getTextureByID (unsigned int id) {

	return (m_pTextureManager->getTexture (id));
}


Texture *
ResourceManager::getTexture(unsigned int i) {

	if (i < m_pTextureManager->m_Lib.size())
		return (m_pTextureManager->m_Lib[i]);
	else
		return NULL;
}


int 
ResourceManager::getNumTextures() {

	return(m_pTextureManager->m_Lib.size());
}

//-------------------------------------	

//			RENDER TARGETS

//-------------------------------------

nau::render::RenderTarget* 
ResourceManager::createRenderTarget (std::string name, int width, int height) {

	if (m_RenderTargets.count (name) > 0) {
		return 0;
	}
	m_RenderTargets[name] = RenderTarget::Create (name, width, height);
	return m_RenderTargets[name];
}


void 
ResourceManager::removeRenderTarget (std::string name) {

	if (m_RenderTargets.count (name) > 0) {
		delete m_RenderTargets[name];
	}
}


int 
ResourceManager::getNumRenderTargets() {

	return m_RenderTargets.size();
}


nau::render::RenderTarget *
ResourceManager::getRenderTarget(const std::string &name) {

	if (m_RenderTargets.count (name) > 0)
		return m_RenderTargets[name];
	else
		return 0;
}


std::vector<std::string>* 
ResourceManager::getRenderTargetNames() {

  std::map<std::string, nau::render::RenderTarget*>::iterator iter;
  std::vector<std::string> *names = new std::vector<std::string>;

  for(iter = m_RenderTargets.begin(); iter != m_RenderTargets.end(); ++iter) {
    names->push_back(iter->first);   
  }
	
  return names;
}


//-------------------------------------	

//			RENDERABLES

//-------------------------------------

using namespace nau::geometry;

nau::render::IRenderable* 
ResourceManager::createRenderable(std::string type, std::string name, std::string filename) {

	IRenderable *r = NULL;

	if ("" == name) {
		std::stringstream z;
		z << "Unnamed";
		z << ResourceManager::renderableCount++;
		name = z.str();
	}

	if (hasRenderable(name,filename))
		return(getRenderable(name, filename));

	if (0 == type.compare ("Mesh")) 
		r = new Mesh();
	
	else if (0 == type.compare("MeshPose")) 
		r = new MeshPose();
	
	else if ("MeshBones" == type) 
		r = new MeshBones();
	
	else if ("Box" == type) 
		r = new Box();

	else if ("Square" == type) 
		r = new Square();

	else if ("Sphere" == type) 
		r = new Sphere();

	else if ("Axis" == type)
		r = new Axis();

	else if ("BoundingBox" == type)
		r = new BBox();

	r->setName(name);
	addRenderable(r,filename);
	return r;
}


bool 
ResourceManager::hasRenderable (std::string meshName, std::string filename) {

	std::string key (filename);

	if("" != key)
		key += "#" + meshName;
	else
		key = meshName;

	if (m_Meshes.count (key) > 0) {
		return true;
	}
	return false;
}


nau::render::IRenderable* 
ResourceManager::getRenderable (std::string meshName, std::string filename) {	
	
	std::string key (filename);

	if (key != "")
		key += "#" + meshName;
	else
		key = meshName;

	if (m_Meshes.count (key) > 0) {
		return m_Meshes[key];
	}
	return 0;
}


nau::render::IRenderable* 
ResourceManager::addRenderable (nau::render::IRenderable* aMesh, std::string filename) {

	std::string key (filename);
	std::string meshName = aMesh->getName();

	if (key != "") { 
		key += "#" + meshName;
		aMesh->setName(key);
	}
	else
		key = meshName;

	m_Meshes[key] = aMesh;

	return aMesh;
}
 

void
ResourceManager::removeRenderable(std::string name) {

	if (m_Meshes.count(name) != 0)
		m_Meshes.erase(name);
}


//-------------------------------------	

//			STATES

//-------------------------------------

bool 
ResourceManager::hasState (std::string stateName) {

	if (m_States.count (stateName) > 0) {
		return true;
	}
	return false;
}


nau::render::IState* 
ResourceManager::getState (std::string stateName) {

	if (m_States.count (stateName) > 0) {
		return m_States[stateName];
	}
	return NULL;
}


void 
ResourceManager::addState (nau::render::IState* aState) {

	m_States[aState->getName()] = aState;
}


bool 
ResourceManager::hasProgram (std::string programName) {

	if (m_Programs.count (programName) > 0){
		return true;
	}
	return false;
}


//-------------------------------------	

//			SHADERS

//-------------------------------------

IProgram* 
ResourceManager::getProgram (std::string programName) {

	if (false == hasProgram (programName)) {
		m_Programs[programName] = IProgram::create();
		m_Programs[programName]->setName(programName);
	}
	return m_Programs[programName];
}


unsigned int 
ResourceManager::getNumPrograms() {

	return m_Programs.size();
}


std::vector<std::string> * 
ResourceManager::getProgramNames() {

	std::vector<std::string> *names = new std::vector<std::string>; 

	for( std::map<std::string, nau::render::IProgram*>::iterator iter = m_Programs.begin(); iter != m_Programs.end(); ++iter ) {
      names->push_back((*iter).first); 
    }
	return names;
}


//-------------------------------------	

//			BUFFERS

//-------------------------------------


void
ResourceManager::clearBuffers() {

	for (auto b : m_Buffers) {

		if (b.second->getPrope(IBuffer::CLEAR) == IBuffer::BY_FRAME)
			b.second->clear();
	}
}



nau::render::IBuffer* 
ResourceManager::getBuffer(std::string name) {

	if (m_Buffers.count(name) != 0)
		return m_Buffers[name];
	else
		return NULL;
}


nau::render::IBuffer*
ResourceManager::createBuffer(std::string name) {

	IBuffer *b = IBuffer::Create(name);
	m_Buffers[name] = b;
	return b;
}


bool 
ResourceManager::hasBuffer(std::string name) {

	if (m_Buffers.count(name) != 0)
		return true;
	else
		return false;
}


nau::render::IBuffer* 
ResourceManager::getBufferByID(int id) {

	for (auto b : m_Buffers) {

		if (b.second->getPropi(IBuffer::ID) == id)

			return b.second;
	}
	return NULL;
}


void 
ResourceManager::getBufferNames(std::vector<std::string> &names) {

	names.clear();
	for (auto b : m_Buffers) {
		names.push_back(b.second->getLabel());
	}
}
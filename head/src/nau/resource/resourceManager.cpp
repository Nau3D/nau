#include "nau/resource/resourceManager.h"

#include "nau/geometry/box.h"
#include "nau/geometry/sphere.h"
#include "nau/geometry/square.h"
#include "nau/geometry/axis.h"
#include "nau/geometry/terrain.h"
#include "nau/geometry/boundingBoxPrimitive.h"
#include "nau/geometry/meshWithPose.h"
#include "nau/geometry/meshBones.h"
#include "nau/material/iTextureCubeMap.h"
#include "nau/system/file.h"

#include <sstream>

using namespace nau::geometry;
using namespace nau::material;
using namespace nau::render;
using namespace nau::resource;
using namespace nau::scene;
using namespace nau::system;

int ResourceManager::renderableCount = 0;


ResourceManager::ResourceManager(std::string path) :
		m_Path(path),
		m_RenderTargets(),
		m_Meshes(),
		m_States(),
		m_Programs(),
		m_Textures()
{
	m_EmptyMesh = NULL;
}


ResourceManager::~ResourceManager(void) {

	clear();
}


void
ResourceManager::clear() {

	while(!m_Textures.empty()) {
		delete(*m_Textures.begin());
		m_Textures.erase(m_Textures.begin());
	}

	while (!m_States.empty()){
		delete((*m_States.begin()).second);
		m_States.erase(m_States.begin());
	}

	m_Meshes.clear();
	//while (!m_Meshes.empty()){
	//	delete((*m_Meshes.begin()).second);
	//	m_Meshes.erase(m_Meshes.begin());
	//}

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

//			TEXTURES

//-------------------------------------

bool
ResourceManager::hasTexture(std::string name) {

//	return m_pTextureManager->hasTexture(name);
	size_t siz = m_Textures.size();
	ITexture *tex;

	for (unsigned int i = 0; i < siz; i++) {
		tex = m_Textures[i];
		if (name == tex->getLabel()) {
			return(true);
		}
	}
	return(false);
}


nau::material::ITexture * 
ResourceManager::createTexture(std::string label) {

	ITexture *tex;

	if (true == hasTexture(label)) {
		tex = getTexture(label); 
		return(tex);
	}

	tex = ITexture::Create(label);
	m_Textures.push_back(tex);

	return(tex);
}


nau::material::ITexture*
ResourceManager::createTexture (std::string label, 
				std::string internalFormat, 
				int width, int height, int depth, 
				int layers, int levels, int samples) {

	ITexture *tex;

	if (true == hasTexture (label)) {
		tex = getTexture (label); 
		return(tex);
	}

	tex = ITexture::Create(label, internalFormat, width, height, depth, layers, levels, samples);
	m_Textures.push_back(tex);

	return(tex);
}


nau::material::ITexture* 
ResourceManager::addTexture (std::string filename, std::string label, bool mipmap) {

	size_t siz = m_Textures.size();
	ITexture *tex;

	File path (m_Path);
	File f (filename);

	if ("" == label) {
		label = path.getRelativeTo (f);
	}

	for (unsigned int i = 0; i < siz; i++) {
		tex = m_Textures[i];
		if (label == tex->getLabel()) {
			return(tex);
		}
	}

	// if the texture does not exist yet
	tex = ITexture::Create (filename, label, mipmap);
	if (tex)
		m_Textures.push_back(tex);

	return(tex);
}


nau::material::ITexture* 
ResourceManager::addTexture (std::vector<std::string> filenames, std::string label, bool mipmap) {

	size_t siz = m_Textures.size();
	ITexture *tex;

	for (unsigned int i = 0; i < siz; i++) {
		tex = m_Textures[i];
		if (label == tex->getLabel()) {
			return(tex);
		}
	}

	// if the texture does not exist yet
	tex = ITextureCubeMap::Create (filenames, label, mipmap);
	m_Textures.push_back(tex);

	return(tex);
}


void 
ResourceManager::removeTexture (std::string name) {

	std::vector<ITexture*>::iterator texIter;

	texIter = m_Textures.begin();

	while (texIter != m_Textures.end() && (*texIter)->getLabel() != name) {
		texIter++;
	}
	if (texIter != m_Textures.end()) {
		delete (*texIter);
		m_Textures.erase (texIter);
	}
}


ITexture*
ResourceManager::getTexture (std::string name) {

	size_t siz = m_Textures.size();
	ITexture *tex;

	for (unsigned int i = 0; i < siz; i++) {
		tex = m_Textures[i];
		if (name == tex->getLabel()) {
			return(tex);
		}
	}

	return(NULL);
}


ITexture*
ResourceManager::getTextureByID (unsigned int id) {

	size_t siz = m_Textures.size();
	ITexture *tex;

	for (unsigned int i = 0; i < siz; i++) {
		tex = m_Textures[i];
		if (id == tex->getPropi(ITexture::ID)) {
			return(tex);
		}
	}

	return(NULL);
}


ITexture *
ResourceManager::getTexture(unsigned int i) {

	if (i < m_Textures.size())
		return (m_Textures[i]);
	else
		return NULL;
}


int 
ResourceManager::getNumTextures() {

	return (unsigned int)m_Textures.size();
}

//-------------------------------------	
//
//			RENDER TARGETS
//
//-------------------------------------

nau::render::IRenderTarget* 
ResourceManager::createRenderTarget (std::string name) {

	if (m_RenderTargets.count (name) > 0) {
		return 0;
	}
	m_RenderTargets[name] = IRenderTarget::Create (name);
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

	return (unsigned int)m_RenderTargets.size();
}


nau::render::IRenderTarget *
ResourceManager::getRenderTarget(const std::string &name) {

	if (m_RenderTargets.count (name) > 0)
		return m_RenderTargets[name];
	else
		return 0;
}


bool
ResourceManager::hasRenderTarget(const std::string &name) {

	if (m_RenderTargets.count (name) > 0)
		return true;
	else
		return false;
}


std::vector<std::string>* 
ResourceManager::getRenderTargetNames() {

  std::map<std::string, nau::render::IRenderTarget*>::iterator iter;
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

std::shared_ptr<nau::render::IRenderable>
ResourceManager::createRenderable(std::string type, std::string name, std::string filename) {

	std::shared_ptr<nau::render::IRenderable> r;

	if ("" == name) {
		std::stringstream z;
		z << "Unnamed";
		z << ResourceManager::renderableCount++;
		name = z.str();
	}

	if (hasRenderable(name,filename))
		return getRenderable(name, filename);

	if (0 == type.compare ("Mesh")) 
		r =  std::shared_ptr<nau::render::IRenderable>(new Mesh());
	
	else if (0 == type.compare("MeshPose")) 
		r = std::shared_ptr<nau::render::IRenderable>(new MeshPose());
	
	else if ("MeshBones" == type) 
		r = std::shared_ptr<nau::render::IRenderable>(new MeshBones());
	
	else if ("BOX" == type) 
		r = std::shared_ptr<nau::render::IRenderable>(new Box());

	else if ("SQUARE" == type) 
		r = std::shared_ptr<nau::render::IRenderable>(new Square());

	else if ("SPHERE" == type) 
		r = std::shared_ptr<nau::render::IRenderable>(new Sphere());

	else if ("AXIS" == type)
		r = std::shared_ptr<nau::render::IRenderable>(new Axis());

	else if ("BoundingBox" == type)
		r = std::shared_ptr<nau::render::IRenderable>(new BBox());

	else if ("Terrain" == type)
		r = std::shared_ptr<nau::render::IRenderable>(new Terrain());
	else
		return NULL;

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


std::shared_ptr<nau::render::IRenderable> &
ResourceManager::getRenderable (std::string meshName, std::string filename) {	
	
	std::string key (filename);

	if (key != "")
		key += "#" + meshName;
	else
		key = meshName;

	if (m_Meshes.count (key) > 0) {
		return m_Meshes[key];
	}
	return m_EmptyMesh;
}


std::shared_ptr<nau::render::IRenderable> &
ResourceManager::addRenderable (std::shared_ptr<nau::render::IRenderable> &aMesh, std::string filename) {

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


IState *
ResourceManager::createState(const std::string &stateName) {

	if (!m_States.count(stateName)) {
		m_States[stateName] = IState::create();
		m_States[stateName]->setName(stateName);
	}
	return m_States[stateName];
}


bool 
ResourceManager::hasState (const std::string &stateName) {

	if (m_States.count (stateName) > 0) {
		return true;
	}
	return false;
}


nau::material::IState* 
ResourceManager::getState (const std::string &stateName) {

	if (m_States.count (stateName) > 0) {
		return m_States[stateName];
	}
	return NULL;
}


void 
ResourceManager::addState (nau::material::IState* aState) {

	m_States[aState->getName()] = aState;
}


//-------------------------------------	

//			SHADERS

//-------------------------------------

bool 
ResourceManager::hasProgram (std::string programName) {

	if (m_Programs.count (programName) > 0){
		return true;
	}
	return false;
}


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

	return (unsigned int)m_Programs.size();
}


std::vector<std::string> * 
ResourceManager::getProgramNames() {

	std::vector<std::string> *names = new std::vector<std::string>; 

	for( std::map<std::string, nau::material::IProgram*>::iterator iter = m_Programs.begin(); iter != m_Programs.end(); ++iter ) {
      names->push_back((*iter).first); 
    }
	return names;
}


//-------------------------------------	
//			BUFFERS
//-------------------------------------

void
ResourceManager::clearBuffers() {

	for (auto &b : m_Buffers) {

		if (b.second->getPrope(IBuffer::CLEAR) == IBuffer::BY_FRAME)
			b.second->clear();
	}
}


nau::material::IBuffer* 
ResourceManager::getBuffer(std::string name) {

	if (m_Buffers.count(name) != 0)
		return m_Buffers[name];
	else
		return NULL;
}


nau::material::IBuffer*
ResourceManager::createBuffer(std::string name) {

	if (m_Buffers.count(name))
		return m_Buffers[name];
	else {
		IBuffer *b = IBuffer::Create(name);
		m_Buffers[name] = b;
		return b;
	}
}


bool 
ResourceManager::hasBuffer(std::string name) {

	if (m_Buffers.count(name) != 0)
		return true;
	else
		return false;
}


nau::material::IBuffer* 
ResourceManager::getBufferByID(int id) {

	for (auto b : m_Buffers) {

		if (b.second->getPropi(IBuffer::ID) == id)

			return b.second;
	}
	return NULL;
}


void 
ResourceManager::getBufferNames(std::vector<std::string> *names) {

	for (auto b : m_Buffers) {
		names->push_back(b.second->getLabel());
	}
}
#include "nau/material/material.h"

#include "nau/debug/profile.h"
#include "nau/slogger.h"
#include "nau.h"

using namespace nau::material;
using namespace nau::render;
using namespace nau::resource;

Material::Material() : 
   m_Color (),
   //m_Texmat (0),
   m_Shader (NULL),
   m_ProgramValues(),
   m_UniformValues(),
   m_Enabled (true),
   m_Name ("__Default"),
   m_useShader(true)
{
	m_State = IState::create(); 
}


Material::~Material()
{
   if (0 != m_State) {
		delete m_State;
		m_State = 0;
   }
   // Must delete textures

}



Material*
Material::clone() { // check clone Program Values

   Material *mat;

   mat = new Material();

   mat->setName(m_Name);

   mat->m_Enabled = m_Enabled;
   mat->m_Shader = m_Shader;
   mat->m_useShader = m_useShader;

   mat->m_ProgramValues = m_ProgramValues;

   mat->m_Color.clone(m_Color);

   mat->m_Buffers = m_Buffers;

   for (auto mt : m_Textures) {

	   mat->m_Textures[mt.first] = mt.second;
   }

#if NAU_OPENGL_VERSION >=  420
   mat->m_ImageTextures = m_ImageTextures;
#endif

   if (m_State != 0)
		mat->m_State = m_State->clone();

   return mat;
}

			
void 
Material::enableShader(bool value) { 

	m_useShader = value;
}

			
bool 
Material::isShaderEnabled() {

	return(m_useShader);
}

			
std::map<std::string, nau::material::ProgramValue>& 
Material::getProgramValues() {

	return m_ProgramValues;
}


std::map<std::string, nau::material::ProgramValue>& 
Material::getUniformValues() {

	return m_UniformValues;
}


void 
Material::setName (std::string name) {

   m_Name = name;
}


std::string& 
Material::getName () {

   return m_Name;
}


void 
Material::getValidProgramValueNames(std::vector<std::string> *names) {

	// valid program value names are program values that are not part of the uniforms map
	// this is because there may be a program value specified in the project file whose type
	// does not match the shader variable type

	std::map<std::string,ProgramValue>::iterator progValIter;
    progValIter = m_ProgramValues.begin();

    for (; progValIter != m_ProgramValues.end(); progValIter++) {
		if (m_UniformValues.count(progValIter->first) == 0)
		names->push_back(progValIter->first); 
	}
}


void
Material::getUniformNames(std::vector<std::string> *names) {

	std::map<std::string,ProgramValue>::iterator progValIter;
    progValIter = m_UniformValues.begin();

    for (; progValIter != m_UniformValues.end(); progValIter++) {
		names->push_back(progValIter->first); 
	}
}


ProgramValue *
Material::getProgramValue(std::string name) {

	if (m_ProgramValues.count(name) > 0)
		return &m_ProgramValues[name];
	else
		return NULL;
}


void
Material::getTextureNames(std::vector<std::string> *vs) {

	for (auto t : m_Textures) {

		vs->push_back(t.second->getTexture()->getLabel());
	}
}


void
Material::getTextureUnits(std::vector<int> *vi) {

	for (auto t : m_Textures) {

		vi->push_back(t.second->getPropi(MaterialTexture::UNIT));
	}
}


void 
Material::setUniformValues() {

	PROFILE("Set Uniforms");
	std::map<std::string,ProgramValue>::iterator progValIter;

	progValIter = m_ProgramValues.begin();

	for (; progValIter != m_ProgramValues.end(); ++progValIter) {
			
		void *v = progValIter->second.getValues();
		m_Shader->setValueOfUniform(progValIter->first, v);
	}
}


void 
Material::checkProgramValuesAndUniforms() {

	int loc;
	IUniform iu;
	std::string s;
	// get the location of the ProgramValue in the shader
	// loc == -1 means that ProgramValue is not an active uniform
	std::map<std::string, ProgramValue>::iterator progValIter;

	progValIter = m_ProgramValues.begin();
	for (; progValIter != m_ProgramValues.end(); ++progValIter) {
		loc = m_Shader->getUniformLocation(progValIter->first);
		progValIter->second.setLoc(loc);
		if (loc == -1)
			SLOG("Material %s: material uniform %s is not active in shader %s", m_Name.c_str(), progValIter->first.c_str(), m_Shader->getName().c_str());
	}

	int k = m_Shader->getNumberOfUniforms();
	for (int i = 0; i < k; ++i) {
		
		iu = m_Shader->getIUniform(i);
		s = iu.getName();
		if (m_ProgramValues.count(s) == 0) {
			SLOG("Material %s: shader uniform %s from shader %s is not defined", m_Name.c_str(), s.c_str(), m_Shader->getName().c_str());
		}
		else if (! Enums::isCompatible(m_ProgramValues[s].getValueType(), iu.getSimpleType())) {
	
			SLOG("Material %s: uniform %s types are not compatiple (%s, %s)", m_Name.c_str(), s.c_str(), iu.getStringSimpleType().c_str(), Enums::DataTypeToString[m_ProgramValues[s].getValueType()].c_str());

		}
	}
}


void 
Material::prepareNoShaders ()
{
	RENDERER->setState (m_State);

	m_Color.prepare();

	for (auto t : m_Textures)
		t.second->bind();

#if NAU_OPENGL_VERSION >=  420
	for (auto it: m_ImageTextures) 
		it.second->prepare();
#endif

	for (auto b : m_Buffers) {
		b.second->bind();
	}
}


void 
Material::prepare () {

	{
		PROFILE("Buffers");
		for (auto b : m_Buffers) {
			b.second->bind();
		}
	}
	{
		PROFILE("State");
		RENDERER->setState (m_State);
	}
	{
		PROFILE("Color");
		m_Color.prepare();
	}
	{	PROFILE("Texture");
		for (auto t : m_Textures) {
			t.second->bind();
		}
	}

#if NAU_OPENGL_VERSION >=  420
	{
		for (auto it : m_ImageTextures)
			it.second->prepare();
	}
#endif
	{
		PROFILE("Shaders");
		if (NULL != m_Shader && m_useShader) {

			m_Shader->prepare();
			RENDERER->setShader(m_Shader);
			setUniformValues();
		}
		else
			RENDERER->setShader(NULL);
	}

}


void 
Material::restore() {

   m_Color.restore();

   if (NULL != m_Shader && m_useShader) {
	   m_Shader->restore();
    }

   for (auto t : m_Textures)
	   t.second->unbind();

   for (auto b : m_Buffers) 
	   b.second->unbind();
   
#if NAU_OPENGL_VERSION >=  420
   for (auto b : m_ImageTextures) 
	   b.second->restore();
#endif

}



void 
Material::restoreNoShaders() {

   m_Color.restore();

   for (auto t : m_Textures)
	   t.second->unbind();

   for (auto b : m_Buffers)
		b.second->unbind();

#if NAU_OPENGL_VERSION >=  420
	for (auto b : m_ImageTextures) 
		b.second->restore();
#endif
}



void 
Material::setState(IState *s) {

	m_State = s;
}

#if NAU_OPENGL_VERSION >=  420

void
Material::attachImageTexture(std::string label, unsigned int unit, unsigned int texID) {

	ImageTexture *it = ImageTexture::Create(label, unit, texID);
	m_ImageTextures[unit] = it;
}


ImageTexture *
Material::getImageTexture(unsigned int unit) {

	if (m_ImageTextures.count(unit))
		return m_ImageTextures[unit];
	else
		return NULL;
}

#endif // NAU_OPENGL_VERSION >=  420



void 
Material::attachBuffer(IMaterialBuffer *b) {

	int bp = b->getPropi(IMaterialBuffer::BINDING_POINT);
	m_Buffers[bp] = b;
}


IMaterialBuffer *
Material::getBuffer(int id) {

	if (m_Buffers.count(id))
		return m_Buffers[id];
	else
		return NULL;
}


bool
Material::hasBuffer(int id) {

	return  0 != m_Buffers.count(id);
}


void
Material::getBufferBindings(std::vector<int> *vi) {

	for (auto t : m_Buffers) {

		vi->push_back(t.second->getPropi(IMaterialBuffer::BINDING_POINT));
	}
}


bool
Material::createTexture (int unit, std::string fn) {

	Texture *tex = RESOURCEMANAGER->addTexture (fn);
	if (tex) {
		MaterialTexture *t = new MaterialTexture(unit);
		t->setSampler(TextureSampler::create(tex));
		t->setTexture(tex);
		m_Textures[unit] = t;
		return(true);
   }
   else {
	   SLOG("Texture not found: %s", fn.c_str());
	   return(false);
   }
}


void 
Material::unsetTexture(int unit) {

	m_Textures.erase(unit);
}


void
Material::attachTexture (int unit, Texture *tex) {

	MaterialTexture *t = new MaterialTexture(unit);
	t->setSampler(TextureSampler::create(tex));
	t->setTexture(tex);
	m_Textures[unit] = t;
}


void
Material::attachTexture (int unit, std::string label) {

	Texture *tex = RESOURCEMANAGER->getTexture (label);

	assert(tex != NULL);

	MaterialTexture *t = new MaterialTexture(unit);
	t->setSampler(TextureSampler::create(tex));
	t->setTexture(tex);
	m_Textures[unit] = t;
}


Texture*
Material::getTexture(int unit) {

	if (m_Textures.count(unit))
		return m_Textures[unit]->getTexture() ;
	else
		return(NULL);
}


TextureSampler*
Material::getTextureSampler(unsigned int unit) {

	if (m_Textures.count(unit))
		return m_Textures[unit]->getSampler();
	else
		return(NULL);
}


MaterialTexture *
Material::getMaterialTexture(int unit) {

	if (m_Textures.count(unit))
		return m_Textures[unit];
	else
		return(NULL);

}


void 
Material::attachProgram (std::string shaderName)
{
	m_Shader = RESOURCEMANAGER->getProgram(shaderName);
	//m_ProgramValues.clear();
}


void
Material::cloneProgramFromMaterial(Material *mat) {

	m_Shader = mat->getProgram();

	m_ProgramValues.clear();
	m_UniformValues.clear();

	std::map<std::string, nau::material::ProgramValue>::iterator iter;

	iter = mat->m_ProgramValues.begin();
	for( ; iter != mat->m_ProgramValues.end(); ++iter) {
	
		m_ProgramValues[(*iter).first] = (*iter).second;
	}

	iter = mat->m_UniformValues.begin();
	for( ; iter != mat->m_UniformValues.end(); ++iter) {
	
		m_UniformValues[(*iter).first] = (*iter).second;
	}
}


std::string 
Material::getProgramName() {

	if (m_Shader)
		return m_Shader->getName();
	else
		return "";
}


nau::render::IProgram * 
Material::getProgram() {

	return m_Shader;
}

			
void 
Material::setValueOfUniform(std::string name, void *values) {

	if (m_ProgramValues.count(name))
		m_ProgramValues[name].setValueOfUniform(values);
}


void
Material::clearProgramValues() {

	m_ProgramValues.clear();
}


void 
Material::addProgramValue (std::string name, nau::material::ProgramValue progVal) {

	// if specified in the material lib, add it to the program values
	if (progVal.isInSpecML())
		m_ProgramValues[name] = progVal;
	else {
		// if the name is not part of m_ProgramValues or if it is but the type does not match
		if (m_ProgramValues.count(name) == 0 || 
			!Enums::isCompatible(m_ProgramValues[name].getValueType(),progVal.getValueType())) {

			// if there is already a uniform with the same name, and a different type remove it
			if (m_UniformValues.count(name) 
				 || m_UniformValues[name].getValueType() != progVal.getValueType()) {
					
						 m_UniformValues.erase(name);
			}
			// add it to the uniform values
				m_UniformValues[name] = progVal;
		}
	}
}


nau::render::IState*
Material::getState (void) {

   return m_State;
}


nau::material::ColorMaterial& 
Material::getColor (void) {

   return m_Color;
}


//nau::material::TextureMat* 
//Material::getTextures (void) {
//
//   return m_Texmat;
//}


void 
Material::clear() {

   m_Color.clear();
   m_Buffers.clear();
   m_Textures.clear();

#if NAU_OPENGL_VERSION >=  420
   m_ImageTextures.clear();
#endif

   m_Shader = NULL; 
   m_ProgramValues.clear();
   m_Enabled = true;
   //m_State->clear();
   m_State->setDefault();
   m_Name = "Default";
}


void
Material::enable (void) {

   m_Enabled = true;
}


void
Material::disable (void) {

   m_Enabled = false;
}


bool
Material::isEnabled (void) {

   return m_Enabled;
}



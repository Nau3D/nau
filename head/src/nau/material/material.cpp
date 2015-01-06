#include <nau/material/material.h>

#include <nau/debug/profile.h>
#include <nau/slogger.h>
#include <nau.h>

using namespace nau::material;
using namespace nau::render;
using namespace nau::resource;

Material::Material() : 
   m_Color (),
   m_Texmat (0),
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
   if (0 != m_Texmat) {
      delete m_Texmat;
	  m_Texmat = 0;
   }
   if (0 != m_State) {
		delete m_State;
		m_State = 0;
   }
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

   if (m_Texmat != 0)
		mat->m_Texmat = m_Texmat->clone();

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
Material::setName (std::string name)
{
   m_Name = name;
}

std::string& 
Material::getName ()
{
   return m_Name;
}


std::vector<std::string> *
Material::getValidProgramValueNames() 
{
	// valid program value names are program values that are not part of the uniforms map
	// this is because there may be a program value specified in the project file whose type
	// does not match the shader variable type

	std::vector<std::string> *names = new std::vector<std::string>; 

	std::map<std::string,ProgramValue>::iterator progValIter;
    progValIter = m_ProgramValues.begin();

    for (; progValIter != m_ProgramValues.end(); progValIter++) {
		if (m_UniformValues.count(progValIter->first) == 0)
		names->push_back(progValIter->first); 
	}
	return names;

}

std::vector<std::string> *
Material::getUniformNames() 
{
	std::vector<std::string> *names = new std::vector<std::string>; 

	std::map<std::string,ProgramValue>::iterator progValIter;
    progValIter = m_UniformValues.begin();

    for (; progValIter != m_UniformValues.end(); progValIter++) {
		names->push_back(progValIter->first); 
	}
	return names;

}


ProgramValue *
Material::getProgramValue(std::string name) {

	if (m_ProgramValues.count(name) > 0)
		return &m_ProgramValues[name];
	else
		return NULL;
}

std::vector<std::string> *
Material::getTextureNames() {

	if (m_Texmat)
		return m_Texmat->getTextureNames();
	else
		return NULL;
}


std::vector<int> *
Material::getTextureUnits() {

	if (m_Texmat)
		return m_Texmat->getTextureUnits();
	else
		return NULL;
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

	if (0 != m_Texmat) {
		m_Texmat->prepare(m_State);
	}

#if NAU_OPENGL_VERSION >=  420
	if (m_ImageTexture.size() != 0) {
		std::map<int, ImageTexture*>::iterator it = m_ImageTexture.begin();
		for ( ; it != m_ImageTexture.end(); ++it)
			it->second->prepare(it->first);
	}
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
		if (0 != m_Texmat) {
			m_Texmat->prepare(m_State);
		}
	}

#if NAU_OPENGL_VERSION >=  420
	{
		if (m_ImageTexture.size() != 0) {
			std::map<int, ImageTexture*>::iterator it = m_ImageTexture.begin();
			for ( ; it != m_ImageTexture.end(); ++it)
				it->second->prepare(it->first);
		}
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
   if (0 != m_Texmat) {
      m_Texmat->restore(m_State);
   }

   for (auto b : m_Buffers) {

	   b.second->unbind();
   }
#if NAU_OPENGL_VERSION >=  420
   for (auto b : m_ImageTexture) {

	   b.second->restore();
   }
#endif

}



void 
Material::restoreNoShaders() {

   m_Color.restore();
   if (0 != m_Texmat) {
      m_Texmat->restore(m_State);
   }
   for (auto b : m_Buffers) {

	   b.second->unbind();
   }
#if NAU_OPENGL_VERSION >=  420
   for (auto b : m_ImageTexture) {

	   b.second->restore();
   }
#endif
}



void 
Material::setState(IState *s) {

	m_State = s;
}

#if NAU_OPENGL_VERSION >=  420

void
Material::attachImageTexture(std::string label, unsigned int unit, unsigned int texID) {

	ImageTexture *it = ImageTexture::Create(label, texID);
	m_ImageTexture[unit] = it;
}


ImageTexture *
Material::getImageTexture(unsigned int unit) {

	if (m_ImageTexture.count(unit))
		return m_ImageTexture[unit];
	else
		return NULL;
}

#endif // NAU_OPENGL_VERSION >=  420



void 
Material::attachBuffer(IMaterialBuffer *b) {

	int bp = b->getPropi(IMaterialBuffer::BINDING_POINT);
	m_Buffers[bp] = b;
}


//IBuffer *
//Material::getBuffer(int id) {
//
//	if (m_Buffers.count(id))
//		return m_Buffers[id].second;
//	else
//		return NULL;
//}
//
//
//int
//Material::getBufferBindingPoint(int id) {
//
//	if (m_Buffers.count(id))
//		return m_Buffers[id].first;
//	else
//		return -1;
//}



bool
Material::createTexture (int unit, std::string fn)
{
   if (0 == m_Texmat) {
      m_Texmat = new TextureMat;
   }

   Texture *tex = RESOURCEMANAGER->addTexture (fn);
   if (tex) {
		m_Texmat->setTexture (unit, tex);
		return(true);
   }
   else {
	   SLOG("Texture not found: %s", fn.c_str());
	   return(false);
   }
}


void 
Material::unsetTexture(int unit) {

	m_Texmat->unset(unit);
}



void
Material::attachTexture (int unit, Texture *t)
{
	if (0 == m_Texmat) {
      m_Texmat = new TextureMat;
   }

	m_Texmat->setTexture (unit, t);
}


void
Material::attachTexture (int unit, std::string label)
{
	if (0 == m_Texmat) {
	  m_Texmat = new TextureMat;
	}
	
	Texture *tex = RESOURCEMANAGER->getTexture (label);

	assert(tex != NULL);
	// if (tex == NULL)
	//	   tex = RESOURCEMANAGER->newEmptyTexture(label);

	m_Texmat->setTexture (unit, tex);
}


Texture*
Material::getTexture(int unit) {

	if (m_Texmat)
		return(m_Texmat->getTexture(unit));
	else
		return(NULL);
}


// unit must be in [0,7]
TextureSampler*
Material::getTextureSampler(unsigned int unit)
{
	if (m_Texmat)
		return m_Texmat->getTextureSampler(unit);
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
Material::getProgramName() 
{
	if (m_Shader)
		return m_Shader->getName();
	else
		return "";
}



nau::render::IProgram * 
Material::getProgram() 
{
	return m_Shader;
}

			
//bool 
//Material::isInSpecML(std::string name) 
//{
//	if (m_ProgramValues.count(name) == 0)
//		return false;
//	else 
//		return m_ProgramValues[name].isInSpecML();
//}

			
void 
Material::setValueOfUniform(std::string name, void *values) {

	if (m_ProgramValues.count(name))
		m_ProgramValues[name].setValueOfUniform(values);
}


void
Material::clearProgramValues() 
{
	m_ProgramValues.clear();
}


void 
Material::addProgramValue (std::string name, nau::material::ProgramValue progVal)
{
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
Material::getState (void)
{
   return m_State;
}


nau::material::ColorMaterial& 
Material::getColor (void)
{
   return m_Color;
}


nau::material::TextureMat* 
Material::getTextures (void)
{
   return m_Texmat;
}


void 
Material::clear()
{
   m_Color.clear();
   if (m_Texmat != 0)
		m_Texmat->clear();
   m_Shader = NULL; 
   m_ProgramValues.clear();
   m_Enabled = true;
   //m_State->clear();
   m_State->setDefault();
   m_Name = "Default";
}


void
Material::enable (void)
{
   m_Enabled = true;
}


void
Material::disable (void)
{
   m_Enabled = false;
}


bool
Material::isEnabled (void)
{
   return m_Enabled;
}



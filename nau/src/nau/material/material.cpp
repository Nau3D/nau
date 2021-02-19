#include "nau/material/material.h"

#include "nau.h"
#include "nau/slogger.h"
#include "nau/debug/profile.h"
#include "nau/material/uniformBlockManager.h"

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
	m_State(NULL) {

}


Material::~Material() {

	while (!m_ImageTextures.empty()) {
		delete((*m_ImageTextures.begin()).second);
		m_ImageTextures.erase(m_ImageTextures.begin());
	}
	while (!m_Textures.empty()) {
		delete((*m_Textures.begin()).second);
		m_Textures.erase(m_Textures.begin());
	}
	while (!m_Buffers.empty()) {
		delete((*m_Buffers.begin()).second);
		m_Buffers.erase(m_Buffers.begin());
	}
}



std::shared_ptr<Material> 
Material::clone() { // check clone Program Values

   Material *mat;

   mat = new Material();

   mat->setName(m_Name);

   mat->m_Enabled = m_Enabled;
   mat->m_Shader = m_Shader;
 
   mat->m_ProgramValues = m_ProgramValues;
   mat->m_ProgramBlockValues = m_ProgramBlockValues;

   mat->m_Color = m_Color;
   mat->m_Buffers = m_Buffers;
   mat->m_Textures = m_Textures;

   for (auto &it : m_ImageTextures) {
	   mat->attachImageTexture(it.second->getLabel(), it.second->getPropi(IImageTexture::UNIT),
		   it.second->getPropui(IImageTexture::TEX_ID));
   }

   for (auto &it : m_Textures) {
	   mat->attachTexture(it.first, it.second->getTexture());
   }

 	mat->m_State = m_State;

	mat->m_ArrayOfTextures = m_ArrayOfTextures;
	//mat->m_ArrayOfImageTextures = new MaterialArrayOfTextures(m_ArrayOfImageTextures);

   return std::shared_ptr<Material>(mat);
}

			
		
bool 
Material::isShaderLinked() {

	return(m_Shader->isLinked());
}

			
std::map<std::string, nau::material::ProgramValue>& 
Material::getProgramValues() {

	return m_ProgramValues;
}


std::map<std::string, nau::material::ProgramValue>& 
Material::getUniformValues() {

	return m_UniformValues;
}


std::map<std::pair<std::string, std::string>, nau::material::ProgramBlockValue>& 
Material::getProgramBlockValues() {

	return m_ProgramBlockValues;
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
Material::getTextureUnits(std::vector<unsigned int> *vi) {

	for (auto t : m_Textures) {

		vi->push_back(t.second->getPropi(MaterialTexture::UNIT));
	}
}


void 
Material::setUniformValues() {

	{ // explicit block for profiler
		PROFILE_GL("Set Uniforms");
		std::map<std::string, ProgramValue>::iterator progValIter;

		progValIter = m_ProgramValues.begin();

		for (; progValIter != m_ProgramValues.end(); ++progValIter) {

			void *v = progValIter->second.getValues();
			m_Shader->setValueOfUniform(progValIter->first, v);
		}
	}
	int x = 3;
}


void 
Material::setUniformBlockValues() {

	{ // explicit block for profiler
		PROFILE_GL("Set Blocks");

		std::set<std::string> blocks;
		for (auto pbv : m_ProgramBlockValues) {

			void *v = pbv.second.getValues();
			const std::string &block = pbv.first.first;
			std::string uniform = pbv.first.second;
			IUniformBlock *b = UNIFORMBLOCKMANAGER->getBlock(block);
			if (b) {
				b->setUniform(uniform, v);
				blocks.insert(block);
			}
		}
		m_Shader->prepareBlocks();
	}
}

#include <algorithm>

void 
Material::checkProgramValuesAndUniforms(std::string &result) {

	int loc;
	IUniform iu;
	std::string s,aux;

	std::vector<std::string> names, otherNames;
	m_Shader->getAttributeNames(&names);
	for (auto n : names) {

		if (VertexData::GetAttribIndex(n) == VertexData::MaxAttribs) {

			aux = "Material " + m_Name + ": attribute " + n + " -  not valid";
			result += aux + "\n";
			SLOG("Material %s: attribute %s - not valid", m_Name.c_str(), n.c_str());
		}
	}

	names.clear();
	m_Shader->getUniformBlockNames(&names);

	// check if uniforms defined in the project are active in the shader
	std::string aName;
	for (auto bi : m_ProgramBlockValues) {
		aName = bi.first.first;
		if (std::any_of(names.begin(), names.end(), [aName](std::string i) {return i == aName; })) {
			IUniformBlock *b = UNIFORMBLOCKMANAGER->getBlock(aName);
			otherNames.clear();
			b->getUniformNames(&otherNames);
			std::string uniName = bi.first.second;
			if (!std::any_of(otherNames.begin(), otherNames.end(),	[uniName](std::string i) {return i == uniName; })) {
				aux = "Material " + m_Name + ": block: " + bi.first.first + "  uniform: " + uniName + " - not active in shader";
				result += aux + "\n";
				SLOG("Material %s: block %s: uniform %s not active in shader",
					m_Name.c_str(), bi.first.first.c_str(), uniName.c_str());
			}
		}
		else {
			SLOG("Material %s: block %s is not active in shader", m_Name.c_str(), bi.first.first.c_str());
			aux = "Material " + m_Name + ": block: " + bi.first.first + " -  not active in shader";
			result += aux + "\n";
		}
	}

	// check if uniforms used in the shader are defined in the project
	// for each block
	for (auto name : names) {
		IUniformBlock *b = UNIFORMBLOCKMANAGER->getBlock(name);
		otherNames.clear();
		b->getUniformNames(&otherNames);
		// for each uniform in the block
		for (auto uni : otherNames) {
			if (uni.find(".") == std::string::npos  && !std::any_of(m_ProgramBlockValues.begin(), m_ProgramBlockValues.end(),
					[&](std::pair<std::pair<std::string , std::string >, ProgramBlockValue> k) 
					{return k.first == std::pair<std::string, std::string>(name, uni); })) {
				
				SLOG("Material %s: block %s: uniform %s is not defined in the material",
					m_Name.c_str(), name.c_str(), uni.c_str());
				aux = "Material " + m_Name + ": shader block: " + name + " uniform: " + uni + " - not defined in the material";
				result += aux + "\n";

			}
		}

	}

	// get the location of the ProgramValue in the shader
	// loc == -1 means that ProgramValue is not an active uniform
	std::map<std::string, ProgramValue>::iterator progValIter;

	progValIter = m_ProgramValues.begin();
	for (; progValIter != m_ProgramValues.end(); ++progValIter) {
		loc = m_Shader->getUniformLocation(progValIter->first);
		progValIter->second.setLoc(loc);
		if (loc == -1) {
			SLOG("Material %s: material uniform %s is not active in shader %s",
				m_Name.c_str(), progValIter->first.c_str(), m_Shader->getName().c_str());
			aux = "Material: " + m_Name + " uniform: " + progValIter->first + " - not active in shader " + m_Shader->getName();
			result += aux + "\n";

		}
	}

	int k = m_Shader->getNumberOfUniforms();
	for (int i = 0; i < k; ++i) {
		
		iu = m_Shader->getIUniform(i);
		s = iu.getName();
		if (m_ProgramValues.count(s) == 0) {
			SLOG("Material %s: shader uniform %s from shader %s is not defined", 
				m_Name.c_str(), s.c_str(), m_Shader->getName().c_str());
			aux = "Material: " + m_Name + " shader : " + m_Shader->getName() + " + shader uniform: " + s + " - not defined in material";
			result += aux + "\n";
		}
		else if (! Enums::isCompatible(m_ProgramValues[s].getValueType(), iu.getSimpleType())) {
	
			aux = "Material: " + m_Name + " uniform: " + s + " - types are not compatible(" + iu.getStringSimpleType() + "," + 
				Enums::DataTypeToString[m_ProgramValues[s].getValueType()] + ")";
			result += aux + "\n";
			SLOG("Material %s: uniform %s types are not compatiple (%s, %s)",
				m_Name.c_str(), s.c_str(), iu.getStringSimpleType().c_str(), 
				Enums::DataTypeToString[m_ProgramValues[s].getValueType()].c_str());
		}
	}
}


void 
Material::prepareNoShaders ()
{

	RENDERER->setState (getState());

	m_Color.prepare();

	for (auto t : m_Textures)
		t.second->bind();

	if (APISupport->apiSupport(IAPISupport::APIFeatureSupport::IMAGE_TEXTURE)) {
		for (auto it : m_ImageTextures)
			it.second->prepare();
	}

	for (auto b : m_Buffers) {
		b.second->bind();
	}
}


void 
Material::prepare () {

	{
		PROFILE("Buffers");
		for (auto &b : m_Buffers) {
			b.second->bind();
		}
	}
	{
		PROFILE("State");
		RENDERER->setState (getState());
	}
	{
		PROFILE("Color");
		m_Color.prepare();
	}
	{	
		PROFILE("Texture");
		for (auto &t : m_Textures) {
			t.second->bind();
		}
	}

	{
		PROFILE("Image Textures");
		if (APISupport->apiSupport(IAPISupport::APIFeatureSupport::IMAGE_TEXTURE)) {
			for (auto &it : m_ImageTextures)
				it.second->prepare();
		}
	}

	if (m_ArrayOfTextures.size()) {
		PROFILE("Array Of Textures");
		for (auto &at : m_ArrayOfTextures)
			at.bind();
	}
	if (m_ArrayOfImageTextures.size()) {
		PROFILE("Array Of Image Textures");
		for (auto at : m_ArrayOfImageTextures)
			at->bind();
	}
	{
		PROFILE("Shaders");
		if (NULL != m_Shader) {

			bool prepared = m_Shader->prepare();
			RENDERER->setShader(m_Shader);

			if (prepared) {
				setUniformValues();
				setUniformBlockValues();
			}
		}
		else
			RENDERER->setShader(NULL);
	}

}


void 
Material::restore() {

	//m_Color.restore();
	
	//if (NULL != m_Shader && m_useShader) {
	//	m_Shader->restore();
	//}
	if (m_Shader)
		m_Shader->restore();
	
	RENDERER->resetTextures(m_Textures);
	//for (auto t : m_Textures)
	//	t.second->unbind();

	for (auto b : m_Buffers) 
		b.second->unbind();
	
	if (APISupport->apiSupport(IAPISupport::APIFeatureSupport::IMAGE_TEXTURE)) {
		for (auto &b : m_ImageTextures)
			b.second->restore();
	}
	for (auto &b: m_ArrayOfTextures)
		b.unbind();
}



void 
Material::restoreNoShaders() {

   m_Color.restore();

   for (auto t : m_Textures)
	   t.second->unbind();

   for (auto b : m_Buffers)
		b.second->unbind();

	if (APISupport->apiSupport(IAPISupport::APIFeatureSupport::IMAGE_TEXTURE)) {
		for (auto b : m_ImageTextures)
			b.second->restore();
	}
}


void 
Material::addArrayOfTextures(IArrayOfTextures *at, int unit) {

	size_t k = m_ArrayOfTextures.size();
	m_ArrayOfTextures.resize(k+1);
	m_ArrayOfTextures[k].setArrayOfTextures(at);
	m_ArrayOfTextures[k].setPropi(MaterialArrayOfTextures::FIRST_UNIT, unit);
//	m_ArrayOfTextures.setArrayOfTextures(at);
//	m_ArrayOfTextures.setPropi(MaterialArrayOfTextures::FIRST_UNIT, unit);
}


MaterialArrayOfTextures *
Material::getMaterialArrayOfTextures(int k) {

	if (k < m_ArrayOfTextures.size())
		return &m_ArrayOfTextures[k];
	else
		return NULL;
};



void 
Material::addArrayOfImageTextures(MaterialArrayOfImageTextures *m) {

	m_ArrayOfImageTextures.push_back(m);
}


MaterialArrayOfImageTextures * 
Material::getArrayOfImageTextures(int id) {

	if (id < m_ArrayOfImageTextures.size())
		return m_ArrayOfImageTextures[id];
	else
		return NULL;
}


void 
Material::setState(IState *s) {

	m_State = s;
}


void
Material::attachImageTexture(std::string label, unsigned int unit, unsigned int texID) {

	assert(APISupport->apiSupport(IAPISupport::APIFeatureSupport::IMAGE_TEXTURE) && "No image texture support");
	IImageTexture *it = IImageTexture::Create(label, unit, texID);
	m_ImageTextures[unit] = it;
}


IImageTexture *
Material::getImageTexture(unsigned int unit) {

	if (m_ImageTextures.count(unit))
		return m_ImageTextures[unit];
	else
		return NULL;
}


void 
Material::getImageTextureUnits(std::vector<unsigned int> *v) {

	for (auto i : m_ImageTextures) {
		v->push_back(i.first);
	}
}


void 
Material::getTextureIDs(std::vector<unsigned int>* v) {

	for (auto i : m_Textures) {
		v->push_back(i.second->getTexture()->getPropi(ITexture::ID));
	}
}


void 
Material::attachBuffer(IMaterialBuffer *b) {

	int bp = b->getPropi(IMaterialBuffer::BINDING_POINT);
	m_Buffers[bp] = b;
}


IMaterialBuffer *
Material::getMaterialBuffer(int id) {

	if (m_Buffers.count(id))
		return m_Buffers[id];
	else
		return NULL;
}


IBuffer *
Material::getBuffer(int id) {

	if (m_Buffers.count(id))
		return m_Buffers[id]->getBuffer();
	else
		return NULL;
}


bool
Material::hasBuffer(int id) {

	return  0 != m_Buffers.count(id);
}


void
Material::getBufferBindings(std::vector<unsigned int> *vi) {

	for (auto t : m_Buffers) {

		vi->push_back(t.second->getPropi(IMaterialBuffer::BINDING_POINT));
	}
}


bool
Material::createTexture (int unit, std::string fn) {

	ITexture *tex = RESOURCEMANAGER->addTexture (fn);
	if (tex) {
		MaterialTexture *t = new MaterialTexture(unit);
		t->setTexture(tex);
//		t->setSampler(ITextureSampler::create(tex));
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
Material::attachTexture (int unit, ITexture *tex) {

	MaterialTexture *t = new MaterialTexture(unit);
	t->setTexture(tex);
//	t->setSampler(ITextureSampler::create(tex));
	m_Textures[unit] = t;
}


void
Material::attachTexture (int unit, std::string label) {

	ITexture *tex = RESOURCEMANAGER->getTexture (label);

	assert(tex != NULL);

	MaterialTexture *t = new MaterialTexture(unit);
	t->setTexture(tex);
//	t->setSampler(ITextureSampler::create(tex));
	m_Textures[unit] = t;
}


ITexture*
Material::getTexture(int unit) {

	if (m_Textures.count(unit))
		return m_Textures[unit]->getTexture() ;
	else
		return(NULL);
}


ITextureSampler*
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
Material::cloneProgramFromMaterial(std::shared_ptr<Material> &mat) {

	m_Shader = mat->getProgram();

	m_ProgramValues.clear();
	m_ProgramBlockValues.clear();
	m_UniformValues.clear();

	std::map<std::string, nau::material::ProgramValue>::iterator iter;

	iter = mat->m_ProgramValues.begin();
	for( ; iter != mat->m_ProgramValues.end(); ++iter) {
	
		m_ProgramValues[(*iter).first] = (*iter).second;
	}

	for (auto pbv : mat->m_ProgramBlockValues) {

		m_ProgramBlockValues[pbv.first] = pbv.second;
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


IProgram * 
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


void 
Material::addProgramBlockValue (std::string block, std::string name, nau::material::ProgramBlockValue progVal) {

	m_ProgramBlockValues[std::pair<std::string, std::string>(block,name)] = progVal;
}


IState*
Material::getState (void) {

	if (m_State == NULL) {
		m_State = RESOURCEMANAGER->createState(m_Name);
	}
   return m_State;
}


ColorMaterial& 
Material::getColor (void) {

   return m_Color;
}


//nau::material::TextureMat* 
//Material::getTextures (void) {
//
//   return m_Texmat;
//}


//void 
//Material::clear() {
//
//   m_Color.clear();
//   m_Buffers.clear();
//   m_Textures.clear();
//
//   m_ImageTextures.clear();
//
//   m_Shader = NULL; 
//   m_ProgramValues.clear();
//   m_Enabled = true;
//   //m_State->clear();
//   m_State->setDefault();
//   m_Name = "Default";
//}


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



#include "nau/render/opengl/glProgram.h"

#include "nau.h"
#include "nau/config.h"
#include "nau/slogger.h"
#include "nau/geometry/vertexData.h"
#include "nau/material/uniformBlockManager.h"
#include "nau/render/iRenderer.h"
#include "nau/system/file.h"
#include "nau/system/textutil.h"

//#include <GL/glew.h>

using namespace nau::render;
using namespace nau::system;

// STATIC METHOD


GLenum GLProgram::ShaderGLId[IProgram::SHADER_COUNT] = 
	{GL_VERTEX_SHADER, GL_GEOMETRY_SHADER, GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER, GL_FRAGMENT_SHADER, GL_COMPUTE_SHADER, GL_TASK_SHADER_NV, GL_MESH_SHADER_NV };

// CONSTRUCTORS

GLProgram::GLProgram() : 
	m_NumUniforms (0), 
	m_MaxLength (0),
	m_PLinked (false),
	m_ShowGlobalUniforms (false),
	m_Name("default") {

	m_P = glCreateProgram();
	m_Shaders.resize(SHADER_COUNT);
}


GLProgram::~GLProgram() {

	if (m_P != 0)
		glDeleteProgram(m_P);

	for (int i = 0; i < SHADER_COUNT; ++i) {
			if (m_Shaders[i].id != 0)
				glDeleteShader(m_Shaders[i].id);
	}
}


bool
GLProgram::areCompiled() {
	
	bool res = true;

	for (int i = 0; i < SHADER_COUNT; ++i) {
			if (m_Shaders[i].id != 0)
				res = res && m_Shaders[i].compiled;
	}
	return(res);
}


bool 
GLProgram::isCompiled(IProgram::ShaderType type) {
	
	bool res = true;

	if (m_Shaders[type].id != 0)
		res = res && m_Shaders[type].compiled;

	return res;
}


bool 
GLProgram::isLinked() {

	return(m_PLinked);
}


void 
GLProgram::getAttributeNames(std::vector<std::string>* s) {

	int k;
	GLsizei len, siz;
	GLenum typ;
	char name[256];
	glGetProgramiv(m_P, GL_ACTIVE_ATTRIBUTES, &k);
	for (unsigned int i = 0; i < (unsigned int)k; ++i) {
		glGetActiveAttrib(m_P, i, 256, &len, &siz, &typ, name);
		s->push_back(name);
	}
}


void 
GLProgram::setName(const std::string &name) {

	m_Name = name;
}


const std::string &
GLProgram::getName() {

	return(m_Name);
}


bool
GLProgram::loadShader (IProgram::ShaderType type, const std::vector<std::string> &files) {

	if (!isShaderSupported(type))
		return false;

	if (type == TESS_CONTROL_SHADER || type == TESS_EVALUATION_SHADER)
		m_HasTessShader = true;

	if (true == setShaderFiles(type,files)) {
		m_Shaders[type].compiled = compileShader(type);
		return m_Shaders[type].compiled;
	}
	else
		return false;
}


const std::vector<std::string> &
GLProgram::getShaderFiles(ShaderType type) {

	return m_Shaders[type].files;
}


void 
GLProgram::files2source(IProgram::ShaderType type) {

	// loop on filenames to create string array
	std::string source;
	m_Shaders[type].source = (char **)malloc(sizeof (char*) * m_Shaders[type].files.size());
	for (int i = 0; i < m_Shaders[type].files.size(); ++i) {
		if (i > 0)
			source = "#line 1 " + std::to_string(i) + "\n";
		source += nau::system::File::TextRead(m_Shaders[type].files[i]);
		m_Shaders[type].source[i] = (char *)malloc(sizeof(char) * source.size() + 2);
		memcpy(m_Shaders[type].source[i], source.c_str(), source.size());
		char *aux = m_Shaders[type].source[i];
		aux[source.size()] = '\n';
		aux[source.size()+1] = '\0';
	}
}


bool
GLProgram::setShaderFiles (IProgram::ShaderType type, const std::vector<std::string> &files) {

	if (!isShaderSupported(type))
		return false;

	// MUST SPLIT FILENAMES - separator: comma
	m_Shaders[type].files = files;
	// check if files exist
	for (int i = 0; i < m_Shaders[type].files.size(); ++i)
		if (!nau::system::File::Exists(m_Shaders[type].files[i]))
			return false;

	// reset shader
	if (files.size() == 0 && m_Shaders[type].id != 0) {
		glDetachShader(m_P, (GLuint)ShaderGLId[type]);
		glDeleteShader((GLuint)ShaderGLId[type]);
		m_Shaders[type].id = 0;
		m_Shaders[type].attached = false;
		m_Shaders[type].compiled = false;
		for (int i = 0; i < m_Shaders[type].files.size(); ++i) {
			free(m_Shaders[type].source[i]);
		}
		free(m_Shaders[type].source);
		m_Shaders[type].source = NULL;
		m_Shaders[type].files.clear();
		return true;
	}

	// if first time
	if (m_Shaders[type].id == 0) {
		m_Shaders[type].id = glCreateShader(ShaderGLId[type]);
	}

	// init shader variables
	m_Shaders[type].attached = false;
	m_Shaders[type].compiled = false;
	m_PLinked = false;

	files2source(type);
	// set shader source
	glShaderSource (m_Shaders[type].id, (GLsizei)m_Shaders[type].files.size(), m_Shaders[type].source, NULL);
	return true;
}


bool 
GLProgram::setValueOfUniform (const std::string &name, void *values) {

	int i;

	i = findUniform (name);
 	
	if (-1 == i) {
		return false;
	}
	m_Uniforms[i].setValues (values);
	setValueOfUniform(i);

	return true;
}


bool
GLProgram::setValueOfUniform(int loc, void *values) {

	int i = findUniformByLocation(loc);
	if (-1 == i) {
		return false;
	}
	m_Uniforms[i].setValues(values);
	setValueOfUniform(i);
	return true;
}


int
GLProgram::getUniformLocation(std::string name) {

	int i = findUniform(name);
	if (-1 == i)
		return -1;
	else
		return m_Uniforms[i].getLoc();

}


bool
GLProgram::reloadShaderFile (IProgram::ShaderType type) {

	if (!isShaderSupported(type))
		return false;

	m_Shaders[type].compiled = false;
	m_PLinked = false;
	for (int i = 0; i < m_Shaders[type].files.size(); ++i) {
		free(m_Shaders[type].source[i]);
	}
	free(m_Shaders[type].source);
	m_Shaders[type].source = NULL;

	files2source(type);
	glShaderSource(m_Shaders[type].id, (GLsizei)m_Shaders[type].files.size(), m_Shaders[type].source, NULL);
	return true;
}


bool 
GLProgram::reload (void) 
{
	for (int i = 0; i < SHADER_COUNT; ++i) {
		reloadShaderFile((IProgram::ShaderType)i);
		m_Shaders[i].compiled = compileShader((IProgram::ShaderType)i);
	}
	
	if (areCompiled()) {
		m_PLinked = linkProgram();
	}
	
	return m_PLinked;
}


int 
GLProgram::programValidate() {

	int v;

	glGetProgramiv(m_P,GL_VALIDATE_STATUS,&v);
	return v;
}


bool
GLProgram::compileShader (IProgram::ShaderType type) {

	int r;

	if (m_Shaders[type].id != 0) {
		glCompileShader (m_Shaders[type].id);
		
		glGetShaderiv (m_Shaders[type].id, GL_COMPILE_STATUS, &r);
		m_Shaders[type].compiled = (1 == r);

		m_PLinked = false;

		return (m_Shaders[type].compiled);
	}
	else
		return true;
}


bool 
GLProgram::linkProgram() 
{
	int r;

	if (!areCompiled()) {
		return false;
	}

	unsigned int index;
	for (index = 0; index < VertexData::MaxAttribs; index++) {
		glBindAttribLocation(m_P, index , VertexData::Syntax[index].c_str());
	}

	for (int i = 0; i < SHADER_COUNT; ++i) {
		if (m_Shaders[i].id != 0 && m_Shaders[i].attached == false) {
			glAttachShader(m_P, m_Shaders[i].id);
			m_Shaders[i].attached = true;
		}
	}
	glLinkProgram (m_P);
	glUseProgram (m_P);

	glGetProgramiv (m_P, GL_LINK_STATUS, &r);
	m_PLinked = (1 == r);

	glGetProgramiv (m_P, GL_ACTIVE_UNIFORMS, &m_NumUniforms);
	glGetProgramiv (m_P, GL_ACTIVE_UNIFORM_MAX_LENGTH, &m_MaxLength);

	setUniforms();
	setBlocks();

	glUseProgram(0);

	return (m_PLinked);
}


int
GLProgram::getNumberOfUniforms() {

	if (true == m_PLinked) {
		return (m_NumUniforms);
	} else {
		return (-1);
	}
}


int
GLProgram::getAttributeLocation (const std::string &name) {

	return glGetAttribLocation (m_P, name.c_str());
}


void 
GLProgram::useProgram (void) {

	if (m_Shaders[FRAGMENT_SHADER].id == 0 && m_Shaders[COMPUTE_SHADER].id == 0)
		glEnable(GL_RASTERIZER_DISCARD);
	else
		glDisable(GL_RASTERIZER_DISCARD);

	if (true == m_PLinked) {
		glUseProgram (m_P);
	} else {
		glUseProgram (0);
	}
}


unsigned int 
GLProgram::getProgramID() {

	return m_P;
}


void 
GLProgram::showGlobalUniforms (void) {

	m_ShowGlobalUniforms = !m_ShowGlobalUniforms;
}


bool 
GLProgram::prepare (void) {

//	glUseProgram(m_P);
//	return true;
	if (m_PLinked) {
		useProgram();
		return true;
	}
	else
		return false;
}


void 
GLProgram::prepareBlocks(void) {

	UniformBlockManager *blockMan = UNIFORMBLOCKMANAGER;
	IUniformBlock *block;
	std::string blockName;
	for (auto b : m_Blocks)  {
		blockName = b.first;
		block = blockMan->getBlock(blockName);
		block->useBlock();
	}
}


bool
GLProgram::restore (void) {

	glUseProgram (0);
	return true;
}


int 
GLProgram::findUniform (const std::string &name) {

	GLUniform uni;
	int i = 0;
	bool found (false);

	std::vector<GLUniform>::iterator it;
	for (it = m_Uniforms.begin(); it != m_Uniforms.end() && !found; it++) {
		//uni = *it;
		if ((*it).getName() == name) {
			found = true;
		} else {
			i++;
		}
	}

	if (true == found) {
		return (i);
	} else {
		return (-1);
	}
}


int
GLProgram::findUniformByLocation(int loc) {

	GLUniform uni;
	int i = 0;
	bool found(false);

	std::vector<GLUniform>::iterator it;
	for (it = m_Uniforms.begin(); it != m_Uniforms.end() && !found; it++) {
		//uni = *it;
		if ((*it).getLoc() == loc) {
			found = true;
		}
		else {
			i++;
		}
	}

	if (true == found) {
		return (i);
	}
	else {
		return (-1);
	}
}


const GLUniform& 
GLProgram::getUniform(const std::string &name) {

	int i = findUniform (name);
	if (-1 == i) {
		i = 0;
	}
	return (m_Uniforms[i]);
}


void 
GLProgram::getUniformBlockNames(std::vector<std::string>* s) {

	for (auto b : m_Blocks) {

		s->push_back(b.first);
	}
}


const IUniform&
GLProgram::getIUniform(int i) {

	assert((unsigned int) i < m_Uniforms.size());
	return (m_Uniforms[i]);
}


void 
GLProgram::setValueOfUniform (int i) {

	m_Uniforms[i].setValueInProgram();
}


void
GLProgram::setBlocks() {

	int count, dataSize, actualLen, activeUnif, maxUniLength;
	int uniType, uniSize, uniOffset, uniMatStride, uniArrayStride, auxSize;
	char name[256], name2[256];
	unsigned int indices[256];

	IUniformBlock *block;
	UniformBlockManager *blockMan = UNIFORMBLOCKMANAGER;

	glGetProgramiv(m_P, GL_ACTIVE_UNIFORM_BLOCKS, &count);

	for (int i = 0; i < count; ++i) {
		// Get buffers name
		glGetActiveUniformBlockiv(m_P, i, GL_UNIFORM_BLOCK_NAME_LENGTH, &actualLen);
		//name = (char *)malloc(sizeof(char) * actualLen);
		glGetActiveUniformBlockName(m_P, i, actualLen, NULL, name);
		glGetActiveUniformBlockiv(m_P, i, GL_UNIFORM_BLOCK_DATA_SIZE, &dataSize);
		bool newBlock = true;
		std::string sName = name;
		if (blockMan->hasBlock(sName)) {
			newBlock = false;
			block = blockMan->getBlock(sName);
			if (block->getSize() != dataSize)
				NAU_THROW("Block %s is already defined with a different size", name);
		}

		//	/*if (!spBlocks.count(name))*/ {
		//		// Get buffers size
		//		//block = spBlocks[name];
		//		
		//		//printf("DataSize:%d\n", dataSize);

		if (newBlock) {
			blockMan->addBlock(sName, dataSize);
			block = blockMan->getBlock(sName);
			block->setBindingIndex(blockMan->getCurrentBindingIndex());
			IBuffer *b = block->getBuffer();
			b->bind((unsigned int)GL_UNIFORM_BUFFER);
			glBufferData(GL_UNIFORM_BUFFER, dataSize, NULL, GL_DYNAMIC_DRAW);
			glUniformBlockBinding(m_P, i, blockMan->getCurrentBindingIndex());
			glBindBufferRange(GL_UNIFORM_BUFFER, blockMan->getCurrentBindingIndex(), 
								block->getBuffer()->getPropi(IBuffer::ID), 0, dataSize);
		}
		else {
			block = blockMan->getBlock(sName);
			IBuffer *b = block->getBuffer();
			b->bind((unsigned int)GL_UNIFORM_BUFFER);
			glUniformBlockBinding(m_P, i, block->getBindingIndex());
			glBindBufferRange(GL_UNIFORM_BUFFER, block->getBindingIndex(),
				block->getBuffer()->getPropi(IBuffer::ID), 0, dataSize);
		}
		m_Blocks[name] = i;
		glGetActiveUniformBlockiv(m_P, i, GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS, &activeUnif);

		//indices = (unsigned int *)malloc(sizeof(unsigned int) * activeUnif);
		glGetActiveUniformBlockiv(m_P, i, GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES, (int *)indices);

		glGetProgramiv(m_P, GL_ACTIVE_UNIFORM_MAX_LENGTH, &maxUniLength);
		//name2 = (char *)malloc(sizeof(char) * maxUniLength);

		for (int k = 0; k < activeUnif; ++k) {

			glGetActiveUniformName(m_P, indices[k], maxUniLength, &actualLen, name2);
			glGetActiveUniformsiv(m_P, 1, &indices[k], GL_UNIFORM_TYPE, &uniType);
			glGetActiveUniformsiv(m_P, 1, &indices[k], GL_UNIFORM_SIZE, &uniSize);
			glGetActiveUniformsiv(m_P, 1, &indices[k], GL_UNIFORM_OFFSET, &uniOffset);
			glGetActiveUniformsiv(m_P, 1, &indices[k], GL_UNIFORM_MATRIX_STRIDE, &uniMatStride);
			glGetActiveUniformsiv(m_P, 1, &indices[k], GL_UNIFORM_ARRAY_STRIDE, &uniArrayStride);

			if (uniArrayStride > 0)
				auxSize = uniArrayStride * uniSize;

			else if (uniMatStride > 0) {

				switch (uniType) {
				case (int)GL_FLOAT_MAT2:
				case (int)GL_FLOAT_MAT2x3:
				case (int)GL_FLOAT_MAT2x4:
				case (int)GL_DOUBLE_MAT2:
				case (int)GL_DOUBLE_MAT2x3:
				case (int)GL_DOUBLE_MAT2x4:
					auxSize = 2 * uniMatStride;
					break;
				case (int)GL_FLOAT_MAT3:
				case (int)GL_FLOAT_MAT3x2:
				case (int)GL_FLOAT_MAT3x4:
				case (int)GL_DOUBLE_MAT3:
				case (int)GL_DOUBLE_MAT3x2:
				case (int)GL_DOUBLE_MAT3x4:
					auxSize = 3 * uniMatStride;
					break;
				case (int)GL_FLOAT_MAT4:
				case (int)GL_FLOAT_MAT4x2:
				case (int)GL_FLOAT_MAT4x3:
				case (int)GL_DOUBLE_MAT4:
				case (int)GL_DOUBLE_MAT4x2:
				case (int)GL_DOUBLE_MAT4x3:
					auxSize = 4 * uniMatStride;
					break;
				}
			}
			else
				auxSize = Enums::getSize(GLUniform::spSimpleType[(GLenum)uniType]) * uniSize;;

			std::string uniName = name2;
			block->addUniform(uniName, GLUniform::spSimpleType[(GLenum)uniType],
				uniOffset, auxSize, uniArrayStride);
			
		}
	}
}


void 
GLProgram::setUniforms() {

	int i,index,len,size;
	unsigned int type;
	char *name = new char [m_MaxLength + 1]; 
	GLUniform uni;

	// set all types = NOT_USED
	std::vector<GLUniform>::iterator it;
	for(it = m_Uniforms.begin(); it != m_Uniforms.end(); it++) {
		it->setGLType(GLUniform::NOT_USED, 0);
	}
	// add new uniforms and reset types for previous uniforms
	
	for (i = 0; i < m_NumUniforms; i++) {

		glGetActiveUniform (m_P, i, m_MaxLength, &len, &size, (GLenum *)&type, name);
		int loc = glGetUniformLocation(m_P, name);
		if (loc != -1) {


			std::string n(name);
			index = findUniform (n);
			if (-1 != index) {
				m_Uniforms[index].setGLType(type,size);
				m_Uniforms[index].setLoc (loc);
			}
			else {
				uni.reset();
				std::string ProgName (name); 
				uni.setName (ProgName);
				uni.setGLType(type,size);
				uni.setLoc (loc);
				m_Uniforms.push_back (uni);
			}
		}
	}

	// delete all uniforms where type is NOT_USED
	for(it = m_Uniforms.begin(), i = 0; it != m_Uniforms.end(); i++ ) {
		if (it->getGLType() == GLUniform::NOT_USED) {
			it = m_Uniforms.erase(it);
		} 
		else {
			++it;
		}
	}
	m_NumUniforms = (int)m_Uniforms.size();
	delete name;
}


void 
GLProgram::updateUniforms() {

	for (int i = 0; i < m_NumUniforms; i++) { 
		glGetUniformfv (m_P, m_Uniforms[i].getLoc(), (float *)m_Uniforms[i].getValues());
	}
}


const GLUniform& 
GLProgram::getUniform (int i) {

	if (i < m_NumUniforms) {
		return (m_Uniforms[i]);
	} else {
		return (m_Uniforms[0]);
	}
}


std::string  
GLProgram::getShaderInfoLog(ShaderType type) {

//	GLuint shader;
    int infologLength = 0;
    int charsWritten  = 0;
	std::string res;
	char *infoLog;

	if (m_Shaders[type].id == 0)
		return "";

	res = IProgram::ShaderNames[type] + ": OK";

	glGetShaderiv (m_Shaders[type].id, GL_INFO_LOG_LENGTH, &infologLength);

    if (infologLength > 1) {
        infoLog = new char[infologLength]; 
        glGetShaderInfoLog (m_Shaders[type].id, infologLength, &charsWritten, infoLog);
		res.assign(infoLog);
		delete infoLog;
	}
    return (res);
}


const std::string &
GLProgram::getProgramInfoLog() {

    int infologLength = 0;
    int charsWritten  = 0;

	glGetProgramiv (m_P, GL_INFO_LOG_LENGTH, &infologLength);

    if (infologLength > 1) {
        char *infoLog = new char[infologLength]; 
        glGetProgramInfoLog (m_P, infologLength, &charsWritten, infoLog);
		m_ReturnString = infoLog;
		delete infoLog;

	} else {
		m_ReturnString = "Program: OK";
	}
	return(m_ReturnString);
}


int 
GLProgram::getNumberOfUserUniforms() {

	int count = 0;

	if (true == m_PLinked) {
		for (int i = 0; i < m_NumUniforms; i++) {
			if (m_Uniforms[i].getName().substr(0,3) != "gl_" ) {
				count++;
			}
		}
	}
	return (count);
}


bool 
GLProgram::getPropertyb(int query) {

	int res;
	glGetProgramiv(m_P, (GLenum)query, &res);
	return (res != 0);
}


int 
GLProgram::getPropertyi(int query) {

	int res;
	glGetProgramiv(m_P, (GLenum)query, &res);
	return (res);
}



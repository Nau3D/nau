#include "nau/slogger.h"
#include "nau/render/opengl/glprogram.h"
#include "nau/render/vertexdata.h"
#include "nau.h"

#include "nau/system/textfile.h"
#include "nau/config.h"

using namespace nau::render;
using namespace nau::system;

// STATIC METHOD

void GlProgram::FixedFunction() {

	glUseProgram(0);
}

#if NAU_OPENGL_VERSION >= 430
int GlProgram::ShaderGLId[IProgram::SHADER_COUNT] = 
	{GL_VERTEX_SHADER, GL_GEOMETRY_SHADER, GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER, GL_FRAGMENT_SHADER, GL_COMPUTE_SHADER};
#elif NAU_OPENGL_VERSION >= 400
int GlProgram::ShaderGLId[IProgram::SHADER_COUNT] = 
	{GL_VERTEX_SHADER, GL_GEOMETRY_SHADER, GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER, GL_FRAGMENT_SHADER};
#elif NAU_OPENGL_VERSION >= 320
int GlProgram::ShaderGLId[IProgram::SHADER_COUNT] = 
	{GL_VERTEX_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER};
#else
int GlProgram::ShaderGLId[IProgram::SHADER_COUNT] = 
	{GL_VERTEX_SHADER, GL_FRAGMENT_SHADER};
#endif

// CONSTRUCTORS

GlProgram::GlProgram() : 
	m_File(SHADER_COUNT,""),
	m_Source(SHADER_COUNT,""),
	m_ID(SHADER_COUNT,0),
	m_Compiled(SHADER_COUNT,false),
	m_NumUniforms (0), 
	m_MaxLength (0),
	m_PLinked (false),
	m_ShowGlobalUniforms (false),
	m_Name("default")
{
	m_P = glCreateProgram();
}




GlProgram::~GlProgram()
{
	if (m_P != 0)
		glDeleteProgram(m_P);

	for (int i = 0; i < SHADER_COUNT; ++i) {
	
		if (m_ID[i] != 0)
			glDeleteShader(m_ID[i]);
	}

}


bool
GlProgram::areCompiled(){
	
	bool res = true;

	for (int i = 0; i < SHADER_COUNT; ++i) {
		if (m_ID[i] != 0) 
			res = res && m_Compiled[i];
	}
	return(res);
}


bool 
GlProgram::isCompiled(IProgram::ShaderType type){
	
	return m_Compiled[type];
}


bool 
GlProgram::isLinked() {

	return(m_PLinked);
}


void 
GlProgram::setName(const std::string &name)
{
	m_Name = name;
}


const std::string &
GlProgram::getName() 
{
	return(m_Name);
}


bool
GlProgram::loadShader (IProgram::ShaderType type, const std::string &filename)
{
	if (true == setShaderFile(type,filename)) {
		m_Compiled[type] = compileShader(type);
		return m_Compiled[type];
	}
	else
		return false;
}


const std::string &
GlProgram::getShaderFile(ShaderType type) 
{
	return m_File[type];
}


bool 
GlProgram::setValueOfUniform (const std::string &name, void *values)
{
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
GlProgram::setValueOfUniform(int loc, void *values) {

	int i = findUniformByLocation(loc);
	if (-1 == i) {
		return false;
	}
	m_Uniforms[i].setValues(values);
	setValueOfUniform(i);
	return true;
}

int
GlProgram::getUniformLocation(std::string name) {

	int i = findUniform(name);
	if (-1 == i)
		return -1;
	else
		return m_Uniforms[i].getLoc();

}


bool
GlProgram::setShaderFile (IProgram::ShaderType type, const std::string &filename)
{
	// reset shader
	if (filename == "" && m_ID[type] != 0) {
		glDetachShader(m_P, ShaderGLId[type]);
		glDeleteShader(ShaderGLId[type]);
		m_File[type] = "";
		m_ID[type] = 0;
		m_Source[type] = "";
		return true;
	}

	// if first time
	if (m_ID[type] == 0) {
		m_ID[type] = glCreateShader(ShaderGLId[type]);
	}

	// init shader variables
	m_Compiled[type] = false;
	m_PLinked = false;
	m_File[type] = filename;
	m_Source[type] = nau::system::textFileRead(filename);
	
	// set shader source
	const char * vv = m_Source[type].c_str();
		
	glShaderSource (m_ID[type], 1, &vv, NULL);
	return true;
}


bool
GlProgram::reloadShaderFile (IProgram::ShaderType type)
{

	m_Compiled[type] = false;
	m_PLinked = false;
	m_Source[type] = textFileRead (m_File[type]);
	if (m_Source[type] != "") { // if read successfuly

		// set shader source
		const char * ff = (char *)m_Source[type].c_str();

		glShaderSource (m_ID[type], 1, &ff, NULL);
		return true;
	} 
	else 
		return false;
}


bool 
GlProgram::reload (void) 
{
	for (int i = 0; i < SHADER_COUNT; ++i) {
		reloadShaderFile((IProgram::ShaderType)i);
		m_Compiled[i] = compileShader((IProgram::ShaderType)i);
	}
	
	if (areCompiled()) {
		m_PLinked = linkProgram();
	}
	
	return m_PLinked;
}


int 
GlProgram::programValidate() {

	int v;

	glGetProgramiv(m_P,GL_VALIDATE_STATUS,&v);

	return(v);
}


bool
GlProgram::compileShader (IProgram::ShaderType type)
{
	int r;

	if (m_ID[type] != 0) {
		glCompileShader (m_ID[type]);
		
		glGetShaderiv (m_ID[type], GL_COMPILE_STATUS, &r);
		m_Compiled[type] = (1 == r);

		m_PLinked = false;

		return (m_Compiled[type]);
	}
	else
		return true;
}


bool 
GlProgram::linkProgram() 
{
	int r;

	if (!areCompiled()) {
		return false;
	}

	unsigned int index;
	for (index = 0; index < VertexData::MaxAttribs; index++) {
		glBindAttribLocation(m_P, index , VertexData::Syntax[index].c_str());
	}

	//for (int i = 0; i < SHADER_COUNT; ++i) {
	//	if (m_Compiled[i])
	//		glAttachShader(m_P, m_ID[i]);
	//	else
	//		glDetachShader(m_P,m_ID[i]);
	//}
	for (int i = 0; i < SHADER_COUNT; ++i) {
		if (m_ID[i] != 0)
			glAttachShader(m_P, m_ID[i]);
	}
	glLinkProgram (m_P);
	glUseProgram (m_P);

	glGetProgramiv (m_P, GL_LINK_STATUS, &r);
	m_PLinked = (1 == r);

	glGetProgramiv (m_P, GL_ACTIVE_UNIFORMS, &m_NumUniforms);
	glGetProgramiv (m_P, GL_ACTIVE_UNIFORM_MAX_LENGTH, &m_MaxLength);

	setUniforms();

	glUseProgram(0);

	return (m_PLinked);
}


int
GlProgram::getNumberOfUniforms() 
{
	if (true == m_PLinked) {
		return (m_NumUniforms);
	} else {
		return (-1);
	}
}


int
GlProgram::getAttributeLocation (const std::string &name)
{
	return glGetAttribLocation (m_P, name.c_str());
}


void 
GlProgram::useProgram (void) 
{

	if (true == m_PLinked) {
		glUseProgram (m_P);
	} else {
		glUseProgram (0);
	}
}


void 
GlProgram::showGlobalUniforms (void)
{
	m_ShowGlobalUniforms = !m_ShowGlobalUniforms;
}

bool 
GlProgram::prepare (void) 
{
	useProgram();
	return true;
}

bool
GlProgram::restore (void)
{
	glUseProgram (0);
	return true;
}

int 
GlProgram::findUniform (const std::string &name)
{
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
GlProgram::findUniformByLocation(int loc)
{
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
GlProgram::getUniform(const std::string &name) {

	int i = findUniform (name);
	if (-1 == i) {
		i = 0;
	}
	return (m_Uniforms[i]);
}


const IUniform&
GlProgram::getIUniform(int i) {

	assert((unsigned int) i < m_Uniforms.size());
	return (m_Uniforms[i]);
}


void 
GlProgram::setValueOfUniform (int i) {

	GLUniform uni;
	uni = m_Uniforms[i];
	uni.setValueInProgram();
}



void 
GlProgram::setUniforms() {

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

		glGetActiveUniform (m_P, i, m_MaxLength, &len, &size, &type, name);
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
//#if NAU_OPENGL_VERSION >= 400 
//			if (type == GL_UNSIGNED_INT_ATOMIC_COUNTER) {
//				GLenum prop = GL_OFFSET; int len, params;
//				glGetProgramResourceiv(m_P, GL_UNIFORM, i, 1, &prop, sizeof(int), &len, &params);
//				RENDERER->addAtomic(params/4, name);
//			}
//#endif
			if (size > 1) {

				for (int i = 0; i < size; i++) {
					std::stringstream s;
					s << n.c_str() << "[" << i << "]";
					std::string Location = s.str();
					index = findUniform(Location);

					int loc;
					loc = glGetUniformLocation(m_P, s.str().c_str());
					if (loc != -1) {
						if (-1 != index) {
							m_Uniforms[index].setGLType(type, 1);
							m_Uniforms[index].setLoc(loc);
						}
						else {
							uni.reset();
							std::string ProgName(s.str());
							uni.setName(ProgName);
							uni.setGLType(type, 1);
							uni.setLoc(loc);
							m_Uniforms.push_back(uni);
						}
					}
				}
			}
		}

	}

	// delete all uniforms where type is NOT_USED
	for(it = m_Uniforms.begin(), i = 0; it != m_Uniforms.end(); i++ ) {
		if (it->getGLType() == GLUniform::NOT_USED) {
			it = m_Uniforms.erase(it);
		} else {
			++it;
		}
	}
	m_NumUniforms = m_Uniforms.size();
	//for (int i = 0; i < m_NumUniforms; i++) {
	//	setValueOfUniform (i);
	//}
}

void 
GlProgram::updateUniforms() {

//	glUseProgram (m_P);
	for (int i = 0; i < m_NumUniforms; i++) { 
		glGetUniformfv (m_P, m_Uniforms[i].getLoc(), (float *)m_Uniforms[i].getValues());
	}

//	glUseProgram(0);
}

const GLUniform& 
GlProgram::getUniform (int i) {

	if (i < m_NumUniforms) {
		return (m_Uniforms[i]);
	} else {
		return (m_Uniforms[0]);
	}
}


std::string  
GlProgram::getShaderInfoLog(ShaderType type) 
{
//	GLuint shader;
    int infologLength = 0;
    int charsWritten  = 0;
	std::string res;
	char *infoLog;

	if (m_ID[type] == 0)
		return "";

	res = IProgram::ShaderNames[type] + ": OK";

	glGetShaderiv (m_ID[type], GL_INFO_LOG_LENGTH, &infologLength);

    if (infologLength > 1) {
        infoLog = new char[infologLength]; 
        glGetShaderInfoLog (m_ID[type], infologLength, &charsWritten, infoLog);	
		res.assign(infoLog);
		free(infoLog);
	} 
    return (res);
}


char* 
GlProgram::getProgramInfoLog() 
{

    int infologLength = 0;
    int charsWritten  = 0;
	std::string ok = "Program: OK";
    char *infoLog;

	glGetProgramiv (m_P, GL_INFO_LOG_LENGTH, &infologLength);

    if (infologLength > 1) {
        infoLog = new char[infologLength]; 
        glGetProgramInfoLog (m_P, infologLength, &charsWritten, infoLog);
		return (infoLog);
	} else {
		infoLog = new char[ok.size()];
		strcpy(infoLog,(char *)ok.c_str());
	}
	return(infoLog);
}


int 
GlProgram::getNumberOfUserUniforms()
{
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
GlProgram::getPropertyb(int query) {

	int res;
	glGetProgramiv(m_P, query, &res);
	return (res != 0);
}


int 
GlProgram::getPropertyi(int query) {

	int res;
	glGetProgramiv(m_P, query, &res);
	return (res);
}



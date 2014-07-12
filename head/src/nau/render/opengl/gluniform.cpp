#include <nau/slogger.h>
#include <nau/render/opengl/gluniform.h>
#include <GL/glew.h>

using namespace nau::render;

GlUniform::GlUniform()
{
	reset();
}

GlUniform::~GlUniform()
{

}


void 
GlUniform::reset()
{

	m_Loc = -1;

	m_Name = "";
	m_Semantics = NONE;
	m_UpperLimit = 1.0f;
	m_LowerLimit = 0.0f;
	for (int i = 0; i < 16; i++) {
		m_Values[i] = 0.0f;
	}
	m_Type = NONE;
	m_Cardinality = 0;
}

int 
GlUniform::getCardinality() 
{
	return (m_Cardinality);
}


void 
GlUniform::setName (std::string &name)
{
	m_Name = name;
}

std::string&
GlUniform::getName (void)
{
	return m_Name;
}

void 
GlUniform::setValues (float *v)
{

	for (int i = 0; i < m_Cardinality; i++) {
		m_Values[i] = v[i];
	}
}

void 
GlUniform::setValues (int *v) 
{

	for (int i = 0; i < m_Cardinality; i++) {
		m_Values[i] = (float)v[i];
	}
}

float*
GlUniform::getValues (void)
{
	return m_Values;
}

int
GlUniform::getLoc (void)
{
	return m_Loc;
}

void
GlUniform::setLoc (int loc)
{
	m_Loc = loc;
}

void 
GlUniform::setType (int type) 
{

	m_Type = type;

	switch (type) {

		case GL_FLOAT:
		case GL_SAMPLER_1D:
		case GL_SAMPLER_2D:
		case GL_SAMPLER_2D_SHADOW:
		case GL_SAMPLER_3D:
		case GL_SAMPLER_CUBE:
		case GL_INT:
		case GL_BOOL:
		case GL_SAMPLER_2D_MULTISAMPLE:
		case GL_IMAGE_2D:
		case GL_IMAGE_2D_MULTISAMPLE:
		case GL_SAMPLER_2D_ARRAY:
		case GL_SAMPLER_2D_ARRAY_SHADOW:

			m_Cardinality = 1;
			break;
		case GL_FLOAT_VEC2:
		case GL_INT_VEC2:
		case GL_BOOL_VEC2:
			m_Cardinality = 2;
			break;
		case GL_FLOAT_VEC3:
		case GL_INT_VEC3:
		case GL_BOOL_VEC3:
			m_Cardinality = 3;
			break;
		case GL_FLOAT_VEC4:
		case GL_INT_VEC4:
		case GL_BOOL_VEC4:
		case GL_FLOAT_MAT2:
			m_Cardinality = 4;
			break;
		case GL_FLOAT_MAT3:
			m_Cardinality = 9;
			break;
		case GL_FLOAT_MAT4:
			m_Cardinality = 16;
			break;
		case NOT_USED:
			m_Cardinality = 0;
			m_Type = NOT_USED;
			break;
		default:
			SLOG("%d - gluniform.cpp line 141 uniform type not supported in NAU", type);
	}
}
			
int 
GlUniform::getType()
{

	return (m_Type);
}

std::string
GlUniform::getProgramValueType() {

	switch (m_Type) {

		case GL_FLOAT:
			return("FLOAT");
		case GL_SAMPLER_1D:
		case GL_SAMPLER_2D:
		case GL_SAMPLER_2D_SHADOW:
		case GL_SAMPLER_3D:
		case GL_SAMPLER_CUBE:
		case GL_SAMPLER_2D_MULTISAMPLE:
		case GL_IMAGE_2D:
		case GL_IMAGE_2D_MULTISAMPLE:
		case GL_SAMPLER_2D_ARRAY:
		case GL_SAMPLER_2D_ARRAY_SHADOW:
			return("SAMPLER");
		case GL_INT:
			return("INT");
		case GL_BOOL:
			return("BOOL");
		case GL_FLOAT_VEC2:
			return("VEC2");
		case GL_INT_VEC2:
			return("IVEC2");
		case GL_BOOL_VEC2:
			return("BVEC2");
		case GL_FLOAT_VEC3:
			return("VEC3");
		case GL_INT_VEC3:
			return("IVEC3");
		case GL_BOOL_VEC3:
			return("BVEC3");
		case GL_FLOAT_VEC4:
			return("VEC4");
		case GL_INT_VEC4:
			return("IVEC4");
		case GL_BOOL_VEC4:
			return("BVEC4");
		case GL_FLOAT_MAT2:
			return("MAT2");
		case GL_FLOAT_MAT3:
			return("MAT3");
		case GL_FLOAT_MAT4:
			return("MAT4");
	}
	SLOG("%d - gluniform.cpp line 199 uniform type not supported in NAU", m_Type);
	return("FLOAT");
}



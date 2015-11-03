#include "nau.h"
#include "nau/slogger.h"
#include "nau/render/opengl/glUniform.h"


using namespace nau::render;

std::map<GLenum, std::string> GLUniform::spGLSLType;
//std::map<int, int> GLUniform::spGLSLTypeSize;
std::map<GLenum, Enums::DataType> GLUniform::spSimpleType;
bool GLUniform::Inited = Init();


GLUniform::GLUniform() : 
	m_Loc(-1), 
	m_Values(NULL), 
	m_Size(0),
	m_Cardinality(0)
{
}

GLUniform::~GLUniform() {

	//if (m_Values != NULL)
	//	free(m_Values);
}


void 
GLUniform::reset() {

	m_Loc = -1;
}


int 
GLUniform::getCardinality() {

	return (m_Cardinality);
}


void 
GLUniform::setValues(void *v) {

	memcpy(m_Values, v, m_Size);
}


//void
//GLUniform::setValues(nau::math::Data *v) {
//
//	m_Values = v;
//}


void
GLUniform::setArrayValue(int index, void *v) {

//	memcpy(m_Values + (Enums::getSize(m_SimpleType) * index), v, Enums::getSize(m_SimpleType));
}


//nau::math::Data *
//GLUniform::getValues(void)
//{
//	return m_Values;
//}


void*
GLUniform::getValues(void)
{
	return m_Values;
}



int
GLUniform::getLoc(void)
{
	return m_Loc;
}


void
GLUniform::setLoc(int loc)
{
	m_Loc = loc;
}


int
GLUniform::getArraySize(void)
{
	return m_ArraySize;
}


void 
GLUniform::setGLType(int type, int arraySize) {

	if (type != 0 && spSimpleType.count((GLenum)type) == 0)
		SLOG("%d - gluniform.cpp - uniform type not supported in NAU", type);

	m_GLType = type;
	m_SimpleType = spSimpleType[(GLenum)type];
	m_ArraySize = arraySize;
	m_Size = Enums::getSize(m_SimpleType) * m_ArraySize;
	m_Cardinality = Enums::getCardinality(m_SimpleType);
	m_Values = (void *)malloc(m_Size);
}
		

int 
GLUniform::getGLType() {

	return (m_GLType);
}


std::string
GLUniform::getStringGLType() {

	return spGLSLType[(GLenum)m_GLType];
}




void
GLUniform::setValueInProgram() {

	switch (m_SimpleType) {
		// inst, bools and samplers
		case Enums::INT:
		case Enums::BOOL:
		case Enums::SAMPLER:
			glUniform1iv(m_Loc, m_ArraySize, (GLint *)m_Values);
			break;
		case Enums::IVEC2:
		case Enums::BVEC2:
			glUniform2iv(m_Loc, m_ArraySize, (GLint *)m_Values);
			break;
		case Enums::IVEC3:
		case Enums::BVEC3:
			glUniform3iv(m_Loc, m_ArraySize, (GLint *)m_Values);
			break;
		case Enums::IVEC4:
		case Enums::BVEC4:
			glUniform4iv(m_Loc, m_ArraySize, (GLint *)m_Values);
			break;

		// unsigned ints
		case Enums::UINT:
			glUniform1uiv(m_Loc, m_ArraySize, (GLuint *)m_Values);
			break;
		case Enums::UIVEC2:
			glUniform2uiv(m_Loc, m_ArraySize, (GLuint *)m_Values);
			break;
		case Enums::UIVEC3:
			glUniform3uiv(m_Loc, m_ArraySize, (GLuint *)m_Values);
			break;
		case Enums::UIVEC4:
			glUniform4uiv(m_Loc, m_ArraySize, (GLuint *)m_Values);
			break;

		// floats
		case Enums::FLOAT:
			glUniform1fv(m_Loc, m_ArraySize, (GLfloat *)m_Values);
			break;
		case Enums::VEC2:
			glUniform2fv(m_Loc, m_ArraySize, (GLfloat *)m_Values);
			break;
		case Enums::VEC3:
			glUniform3fv(m_Loc, m_ArraySize, (GLfloat *)m_Values);
			break;
		case Enums::VEC4:
			glUniform4fv(m_Loc, m_ArraySize, (GLfloat *)m_Values);
			break;

		// doubles
		case Enums::DOUBLE:
			glUniform1dv(m_Loc, m_ArraySize, (GLdouble *)m_Values);
			break;
		case Enums::DVEC2:
			glUniform2dv(m_Loc, m_ArraySize, (GLdouble *)m_Values);
			break;
		case Enums::DVEC3:
			glUniform3dv(m_Loc, m_ArraySize, (GLdouble *)m_Values);
			break;
		case Enums::DVEC4:
			glUniform4dv(m_Loc, m_ArraySize, (GLdouble *)m_Values);
			break;

		// float matrices
		case Enums::MAT2:
			glUniformMatrix2fv(m_Loc, m_ArraySize, GL_FALSE, (GLfloat *)m_Values);
			break;

		case Enums::MAT3:
			glUniformMatrix3fv(m_Loc, m_ArraySize, GL_FALSE, (GLfloat *)m_Values);
			break;

		case Enums::MAT4:
			glUniformMatrix4fv(m_Loc, m_ArraySize, GL_FALSE, (GLfloat *)m_Values);
			break;

		case Enums::MAT2x3:
			glUniformMatrix2x3fv(m_Loc, m_ArraySize, GL_FALSE, (GLfloat *)m_Values);
			break;

		case Enums::MAT2x4:
			glUniformMatrix2x4fv(m_Loc, m_ArraySize, GL_FALSE, (GLfloat *)m_Values);
			break;

		case Enums::MAT3x2:
			glUniformMatrix3x2fv(m_Loc, m_ArraySize, GL_FALSE, (GLfloat *)m_Values);
			break;

		case Enums::MAT3x4:
			glUniformMatrix3x4fv(m_Loc, m_ArraySize, GL_FALSE, (GLfloat *)m_Values);
			break;

			// double matrices
		case Enums::DMAT2:
			glUniformMatrix2dv(m_Loc, m_ArraySize, GL_FALSE, (GLdouble *)m_Values);
			break;

		case Enums::DMAT3:
			glUniformMatrix3dv(m_Loc, m_ArraySize, GL_FALSE, (GLdouble *)m_Values);
			break;

		case Enums::DMAT4:
			glUniformMatrix4dv(m_Loc, m_ArraySize, GL_FALSE, (GLdouble *)m_Values);
			break;

		case Enums::DMAT2x3:
			glUniformMatrix2x3dv(m_Loc, m_ArraySize, GL_FALSE, (GLdouble *)m_Values);
			break;

		case Enums::DMAT2x4:
			glUniformMatrix2x4dv(m_Loc, m_ArraySize, GL_FALSE, (GLdouble *)m_Values);
			break;

		case Enums::DMAT3x2:
			glUniformMatrix3x2dv(m_Loc, m_ArraySize, GL_FALSE, (GLdouble *)m_Values);
			break;

		case Enums::DMAT3x4:
			glUniformMatrix3x4dv(m_Loc, m_ArraySize, GL_FALSE, (GLdouble *)m_Values);
			break;
		default:
			assert(false && "missing data types in switch statement");
	}
}


bool
GLUniform::Init() {

	spSimpleType[GL_SAMPLER_1D] = Enums::DataType::SAMPLER;
	spSimpleType[GL_SAMPLER_2D] = Enums::DataType::SAMPLER;
	spSimpleType[GL_SAMPLER_3D] = Enums::DataType::SAMPLER;
	spSimpleType[GL_SAMPLER_CUBE] = Enums::DataType::SAMPLER;
	spSimpleType[GL_SAMPLER_1D_SHADOW] = Enums::DataType::SAMPLER;
	spSimpleType[GL_SAMPLER_2D_SHADOW] = Enums::DataType::SAMPLER;
	spSimpleType[GL_SAMPLER_1D_ARRAY] = Enums::DataType::SAMPLER;
	spSimpleType[GL_SAMPLER_2D_ARRAY] = Enums::DataType::SAMPLER;
	spSimpleType[GL_SAMPLER_1D_ARRAY_SHADOW] = Enums::DataType::SAMPLER;
	spSimpleType[GL_SAMPLER_2D_ARRAY_SHADOW] = Enums::DataType::SAMPLER;
	spSimpleType[GL_SAMPLER_2D_MULTISAMPLE] = Enums::DataType::SAMPLER;
	spSimpleType[GL_SAMPLER_2D_MULTISAMPLE_ARRAY] = Enums::DataType::SAMPLER;
	spSimpleType[GL_SAMPLER_CUBE_SHADOW] = Enums::DataType::SAMPLER;
	spSimpleType[GL_SAMPLER_BUFFER] = Enums::DataType::SAMPLER;
	spSimpleType[GL_SAMPLER_2D_RECT] = Enums::DataType::SAMPLER;
	spSimpleType[GL_SAMPLER_2D_RECT_SHADOW] = Enums::DataType::SAMPLER;

	spSimpleType[GL_INT_SAMPLER_1D] = Enums::DataType::SAMPLER;
	spSimpleType[GL_INT_SAMPLER_2D] = Enums::DataType::SAMPLER;
	spSimpleType[GL_INT_SAMPLER_3D] = Enums::DataType::SAMPLER;
	spSimpleType[GL_INT_SAMPLER_CUBE] = Enums::DataType::SAMPLER;
	spSimpleType[GL_INT_SAMPLER_1D_ARRAY] = Enums::DataType::SAMPLER;
	spSimpleType[GL_INT_SAMPLER_2D_ARRAY] = Enums::DataType::SAMPLER;
	spSimpleType[GL_INT_SAMPLER_2D_MULTISAMPLE] = Enums::DataType::SAMPLER;
	spSimpleType[GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY] = Enums::DataType::SAMPLER;
	spSimpleType[GL_INT_SAMPLER_BUFFER] = Enums::DataType::SAMPLER;
	spSimpleType[GL_INT_SAMPLER_2D_RECT] = Enums::DataType::SAMPLER;

	spSimpleType[GL_UNSIGNED_INT_SAMPLER_1D] = Enums::DataType::SAMPLER;
	spSimpleType[GL_UNSIGNED_INT_SAMPLER_2D] = Enums::DataType::SAMPLER;
	spSimpleType[GL_UNSIGNED_INT_SAMPLER_3D] = Enums::DataType::SAMPLER;
	spSimpleType[GL_UNSIGNED_INT_SAMPLER_CUBE] = Enums::DataType::SAMPLER;
	spSimpleType[GL_UNSIGNED_INT_SAMPLER_1D_ARRAY] = Enums::DataType::SAMPLER;
	spSimpleType[GL_UNSIGNED_INT_SAMPLER_2D_ARRAY] = Enums::DataType::SAMPLER;
	spSimpleType[GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE] = Enums::DataType::SAMPLER;
	spSimpleType[GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY] = Enums::DataType::SAMPLER;
	spSimpleType[GL_UNSIGNED_INT_SAMPLER_BUFFER] = Enums::DataType::SAMPLER;
	spSimpleType[GL_UNSIGNED_INT_SAMPLER_2D_RECT] = Enums::DataType::SAMPLER;

	spSimpleType[GL_IMAGE_1D] = Enums::DataType::SAMPLER;
	spSimpleType[GL_IMAGE_2D] = Enums::DataType::SAMPLER;
	spSimpleType[GL_IMAGE_3D] = Enums::DataType::SAMPLER;
	spSimpleType[GL_IMAGE_2D_RECT] = Enums::DataType::SAMPLER;
	spSimpleType[GL_IMAGE_CUBE] = Enums::DataType::SAMPLER;
	spSimpleType[GL_IMAGE_BUFFER] = Enums::DataType::SAMPLER;
	spSimpleType[GL_IMAGE_1D_ARRAY] = Enums::DataType::SAMPLER;
	spSimpleType[GL_IMAGE_2D_ARRAY] = Enums::DataType::SAMPLER;
	spSimpleType[GL_IMAGE_2D_MULTISAMPLE] = Enums::DataType::SAMPLER;
	spSimpleType[GL_IMAGE_2D_MULTISAMPLE_ARRAY] = Enums::DataType::SAMPLER;

	spSimpleType[GL_INT_IMAGE_1D] = Enums::DataType::SAMPLER;
	spSimpleType[GL_INT_IMAGE_2D] = Enums::DataType::SAMPLER;
	spSimpleType[GL_INT_IMAGE_3D] = Enums::DataType::SAMPLER;
	spSimpleType[GL_INT_IMAGE_2D_RECT] = Enums::DataType::SAMPLER;
	spSimpleType[GL_INT_IMAGE_CUBE] = Enums::DataType::SAMPLER;
	spSimpleType[GL_INT_IMAGE_BUFFER] = Enums::DataType::SAMPLER;
	spSimpleType[GL_INT_IMAGE_1D_ARRAY] = Enums::DataType::SAMPLER;
	spSimpleType[GL_INT_IMAGE_2D_ARRAY] = Enums::DataType::SAMPLER;
	spSimpleType[GL_INT_IMAGE_2D_MULTISAMPLE] = Enums::DataType::SAMPLER;
	spSimpleType[GL_INT_IMAGE_2D_MULTISAMPLE_ARRAY] = Enums::DataType::SAMPLER;

	spSimpleType[GL_UNSIGNED_INT_IMAGE_1D] = Enums::DataType::SAMPLER;
	spSimpleType[GL_UNSIGNED_INT_IMAGE_2D] = Enums::DataType::SAMPLER;
	spSimpleType[GL_UNSIGNED_INT_IMAGE_3D] = Enums::DataType::SAMPLER;
	spSimpleType[GL_UNSIGNED_INT_IMAGE_2D_RECT] = Enums::DataType::SAMPLER;
	spSimpleType[GL_UNSIGNED_INT_IMAGE_CUBE] = Enums::DataType::SAMPLER;
	spSimpleType[GL_UNSIGNED_INT_IMAGE_BUFFER] = Enums::DataType::SAMPLER;
	spSimpleType[GL_UNSIGNED_INT_IMAGE_1D_ARRAY] = Enums::DataType::SAMPLER;
	spSimpleType[GL_UNSIGNED_INT_IMAGE_2D_ARRAY] = Enums::DataType::SAMPLER;
	spSimpleType[GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE] = Enums::DataType::SAMPLER;
	spSimpleType[GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_ARRAY] = Enums::DataType::SAMPLER;

	spSimpleType[GL_BOOL] = Enums::DataType::BOOL;
	spSimpleType[GL_BOOL_VEC2] = Enums::DataType::BVEC2;
	spSimpleType[GL_BOOL_VEC3] = Enums::DataType::BVEC3;
	spSimpleType[GL_BOOL_VEC4] = Enums::DataType::BVEC4;

	spSimpleType[GL_INT] = Enums::DataType::INT;
	spSimpleType[GL_INT_VEC2] = Enums::DataType::IVEC2;
	spSimpleType[GL_INT_VEC3] = Enums::DataType::IVEC3;
	spSimpleType[GL_INT_VEC4] = Enums::DataType::IVEC4;

	spSimpleType[GL_UNSIGNED_INT] = Enums::DataType::UINT;
	spSimpleType[GL_UNSIGNED_INT_VEC2] = Enums::DataType::UIVEC2;
	spSimpleType[GL_UNSIGNED_INT_VEC3] = Enums::DataType::UIVEC3;
	spSimpleType[GL_UNSIGNED_INT_VEC4] = Enums::DataType::UIVEC4;

	spSimpleType[GL_FLOAT] = Enums::DataType::FLOAT;
	spSimpleType[GL_FLOAT_VEC2] = Enums::DataType::VEC2;
	spSimpleType[GL_FLOAT_VEC3] = Enums::DataType::VEC3;
	spSimpleType[GL_FLOAT_VEC4] = Enums::DataType::VEC4;

	spSimpleType[GL_FLOAT_MAT2] = Enums::DataType::MAT2;
	spSimpleType[GL_FLOAT_MAT3] = Enums::DataType::MAT3;
	spSimpleType[GL_FLOAT_MAT4] = Enums::DataType::MAT4;
	spSimpleType[GL_FLOAT_MAT2x3] = Enums::DataType::MAT2x3;
	spSimpleType[GL_FLOAT_MAT2x4] = Enums::DataType::MAT2x4;
	spSimpleType[GL_FLOAT_MAT3x2] = Enums::DataType::MAT3x2;
	spSimpleType[GL_FLOAT_MAT3x4] = Enums::DataType::MAT3x4;
	spSimpleType[GL_FLOAT_MAT4x2] = Enums::DataType::MAT4x2;
	spSimpleType[GL_FLOAT_MAT4x3] = Enums::DataType::MAT4x3;

	spSimpleType[GL_DOUBLE] = Enums::DataType::DOUBLE;
	spSimpleType[GL_DOUBLE_VEC2] = Enums::DataType::DVEC2;
	spSimpleType[GL_DOUBLE_VEC3] = Enums::DataType::DVEC3;
	spSimpleType[GL_DOUBLE_VEC4] = Enums::DataType::DVEC4;

	spSimpleType[GL_DOUBLE_MAT2] = Enums::DataType::DMAT2;
	spSimpleType[GL_DOUBLE_MAT3] = Enums::DataType::DMAT3;
	spSimpleType[GL_DOUBLE_MAT4] = Enums::DataType::DMAT4;
	spSimpleType[GL_DOUBLE_MAT2x3] = Enums::DataType::DMAT2x3;
	spSimpleType[GL_DOUBLE_MAT2x4] = Enums::DataType::DMAT2x4;
	spSimpleType[GL_DOUBLE_MAT3x2] = Enums::DataType::DMAT3x2;
	spSimpleType[GL_DOUBLE_MAT3x4] = Enums::DataType::DMAT3x4;
	spSimpleType[GL_DOUBLE_MAT4x2] = Enums::DataType::DMAT4x2;
	spSimpleType[GL_DOUBLE_MAT4x3] = Enums::DataType::DMAT4x3;
	//
	spGLSLType[GL_FLOAT] = "GL_FLOAT";
	spGLSLType[GL_FLOAT_VEC2] = "GL_FLOAT_VEC2";
	spGLSLType[GL_FLOAT_VEC3] = "GL_FLOAT_VEC3";
	spGLSLType[GL_FLOAT_VEC4] = "GL_FLOAT_VEC4";

	spGLSLType[GL_DOUBLE] = "GL_DOUBLE";
	spGLSLType[GL_DOUBLE_VEC2] = "GL_DOUBLE_VEC2";
	spGLSLType[GL_DOUBLE_VEC3] = "GL_DOUBLE_VEC3";
	spGLSLType[GL_DOUBLE_VEC4] = "GL_DOUBLE_VEC4";

	spGLSLType[GL_BOOL] = "GL_BOOL";
	spGLSLType[GL_BOOL_VEC2] = "GL_BOOL_VEC2";
	spGLSLType[GL_BOOL_VEC3] = "GL_BOOL_VEC3";
	spGLSLType[GL_BOOL_VEC4] = "GL_BOOL_VEC4";

	spGLSLType[GL_INT] = "GL_INT";
	spGLSLType[GL_INT_VEC2] = "GL_INT_VEC2";
	spGLSLType[GL_INT_VEC3] = "GL_INT_VEC3";
	spGLSLType[GL_INT_VEC4] = "GL_INT_VEC4";

	spGLSLType[GL_UNSIGNED_INT] = "GL_UNSIGNED_INT";
	spGLSLType[GL_UNSIGNED_INT_VEC2] = "GL_UNSIGNED_INT_VEC2";
	spGLSLType[GL_UNSIGNED_INT_VEC3] = "GL_UNSIGNED_INT_VEC3";
	spGLSLType[GL_UNSIGNED_INT_VEC4] = "GL_UNSIGNED_INT_VEC4";

	spGLSLType[GL_FLOAT_MAT2] = "GL_FLOAT_MAT2";
	spGLSLType[GL_FLOAT_MAT3] = "GL_FLOAT_MAT3";
	spGLSLType[GL_FLOAT_MAT4] = "GL_FLOAT_MAT4";
	spGLSLType[GL_FLOAT_MAT2x3] = "GL_FLOAT_MAT2x3";
	spGLSLType[GL_FLOAT_MAT2x4] = "GL_FLOAT_MAT2x4";
	spGLSLType[GL_FLOAT_MAT3x2] = "GL_FLOAT_MAT3x2";
	spGLSLType[GL_FLOAT_MAT3x4] = "GL_FLOAT_MAT3x4";
	spGLSLType[GL_FLOAT_MAT4x2] = "GL_FLOAT_MAT4x2";
	spGLSLType[GL_FLOAT_MAT4x3] = "GL_FLOAT_MAT4x3";
	spGLSLType[GL_DOUBLE_MAT2] = "GL_DOUBLE_MAT2";
	spGLSLType[GL_DOUBLE_MAT3] = "GL_DOUBLE_MAT3";
	spGLSLType[GL_DOUBLE_MAT4] = "GL_DOUBLE_MAT4";
	spGLSLType[GL_DOUBLE_MAT2x3] = "GL_DOUBLE_MAT2x3";
	spGLSLType[GL_DOUBLE_MAT2x4] = "GL_DOUBLE_MAT2x4";
	spGLSLType[GL_DOUBLE_MAT3x2] = "GL_DOUBLE_MAT3x2";
	spGLSLType[GL_DOUBLE_MAT3x4] = "GL_DOUBLE_MAT3x4";
	spGLSLType[GL_DOUBLE_MAT4x2] = "GL_DOUBLE_MAT4x2";
	spGLSLType[GL_DOUBLE_MAT4x3] = "GL_DOUBLE_MAT4x3";

	spGLSLType[GL_SAMPLER_1D] = "GL_SAMPLER_1D";
	spGLSLType[GL_SAMPLER_2D] = "GL_SAMPLER_2D";
	spGLSLType[GL_SAMPLER_3D] = "GL_SAMPLER_3D";
	spGLSLType[GL_SAMPLER_CUBE] = "GL_SAMPLER_CUBE";
	spGLSLType[GL_SAMPLER_1D_SHADOW] = "GL_SAMPLER_1D_SHADOW";
	spGLSLType[GL_SAMPLER_2D_SHADOW] = "GL_SAMPLER_2D_SHADOW";
	spGLSLType[GL_SAMPLER_1D_ARRAY] = "GL_SAMPLER_1D_ARRAY";
	spGLSLType[GL_SAMPLER_2D_ARRAY] = "GL_SAMPLER_2D_ARRAY";
	spGLSLType[GL_SAMPLER_1D_ARRAY_SHADOW] = "GL_SAMPLER_1D_ARRAY_SHADOW";
	spGLSLType[GL_SAMPLER_2D_ARRAY_SHADOW] = "GL_SAMPLER_2D_ARRAY_SHADOW";
	spGLSLType[GL_SAMPLER_2D_MULTISAMPLE] = "GL_SAMPLER_2D_MULTISAMPLE";
	spGLSLType[GL_SAMPLER_2D_MULTISAMPLE_ARRAY] = "GL_SAMPLER_2D_MULTISAMPLE_ARRAY";
	spGLSLType[GL_SAMPLER_CUBE_SHADOW] = "GL_SAMPLER_CUBE_SHADOW";
	spGLSLType[GL_SAMPLER_BUFFER] = "GL_SAMPLER_BUFFER";
	spGLSLType[GL_SAMPLER_2D_RECT] = "GL_SAMPLER_2D_RECT";
	spGLSLType[GL_SAMPLER_2D_RECT_SHADOW] = "GL_SAMPLER_2D_RECT_SHADOW";
	spGLSLType[GL_INT_SAMPLER_1D] = "GL_INT_SAMPLER_1D";
	spGLSLType[GL_INT_SAMPLER_2D] = "GL_INT_SAMPLER_2D";
	spGLSLType[GL_INT_SAMPLER_3D] = "GL_INT_SAMPLER_3D";
	spGLSLType[GL_INT_SAMPLER_CUBE] = "GL_INT_SAMPLER_CUBE";
	spGLSLType[GL_INT_SAMPLER_1D_ARRAY] = "GL_INT_SAMPLER_1D_ARRAY";
	spGLSLType[GL_INT_SAMPLER_2D_ARRAY] = "GL_INT_SAMPLER_2D_ARRAY";
	spGLSLType[GL_INT_SAMPLER_2D_MULTISAMPLE] = "GL_INT_SAMPLER_2D_MULTISAMPLE";
	spGLSLType[GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY] = "GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY";
	spGLSLType[GL_INT_SAMPLER_BUFFER] = "GL_INT_SAMPLER_BUFFER";
	spGLSLType[GL_INT_SAMPLER_2D_RECT] = "GL_INT_SAMPLER_2D_RECT";
	spGLSLType[GL_UNSIGNED_INT_SAMPLER_1D] = "GL_UNSIGNED_INT_SAMPLER_1D";
	spGLSLType[GL_UNSIGNED_INT_SAMPLER_2D] = "GL_UNSIGNED_INT_SAMPLER_2D";
	spGLSLType[GL_UNSIGNED_INT_SAMPLER_3D] = "GL_UNSIGNED_INT_SAMPLER_3D";
	spGLSLType[GL_UNSIGNED_INT_SAMPLER_CUBE] = "GL_UNSIGNED_INT_SAMPLER_CUBE";
	spGLSLType[GL_UNSIGNED_INT_SAMPLER_1D_ARRAY] = "GL_UNSIGNED_INT_SAMPLER_1D_ARRAY";
	spGLSLType[GL_UNSIGNED_INT_SAMPLER_2D_ARRAY] = "GL_UNSIGNED_INT_SAMPLER_2D_ARRAY";
	spGLSLType[GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE] = "GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE";
	spGLSLType[GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY] = "GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY";
	spGLSLType[GL_UNSIGNED_INT_SAMPLER_BUFFER] = "GL_UNSIGNED_INT_SAMPLER_BUFFER";
	spGLSLType[GL_UNSIGNED_INT_SAMPLER_2D_RECT] = "GL_UNSIGNED_INT_SAMPLER_2D_RECT";

	spGLSLType[GL_IMAGE_1D] = "GL_IMAGE_1D";
	spGLSLType[GL_IMAGE_2D] = "GL_IMAGE_2D";
	spGLSLType[GL_IMAGE_3D] = "GL_IMAGE_3D";
	spGLSLType[GL_IMAGE_2D_RECT] = "GL_IMAGE_2D_RECT";
	spGLSLType[GL_IMAGE_CUBE] = "GL_IMAGE_CUBE";
	spGLSLType[GL_IMAGE_BUFFER] = "GL_IMAGE_BUFFER";
	spGLSLType[GL_IMAGE_1D_ARRAY] = "GL_IMAGE_1D_ARRAY";
	spGLSLType[GL_IMAGE_2D_ARRAY] = "GL_IMAGE_2D_ARRAY";
	spGLSLType[GL_IMAGE_2D_MULTISAMPLE] = "GL_IMAGE_2D_MULTISAMPLE";
	spGLSLType[GL_IMAGE_2D_MULTISAMPLE_ARRAY] = "GL_IMAGE_2D_MULTISAMPLE_ARRAY";

	spGLSLType[GL_INT_IMAGE_1D] = "GL_INT_IMAGE_1D";
	spGLSLType[GL_INT_IMAGE_2D] = "GL_INT_IMAGE_2D";
	spGLSLType[GL_INT_IMAGE_3D] = "GL_INT_IMAGE_3D";
	spGLSLType[GL_INT_IMAGE_2D_RECT] = "GL_INT_IMAGE_2D_RECT";
	spGLSLType[GL_INT_IMAGE_CUBE] = "GL_INT_IMAGE_CUBE";
	spGLSLType[GL_INT_IMAGE_BUFFER] = "GL_INT_IMAGE_BUFFER";
	spGLSLType[GL_INT_IMAGE_1D_ARRAY] = "GL_INT_IMAGE_1D_ARRAY";
	spGLSLType[GL_INT_IMAGE_2D_ARRAY] = "GL_INT_IMAGE_2D_ARRAY";
	spGLSLType[GL_INT_IMAGE_2D_MULTISAMPLE] = "GL_INT_IMAGE_2D_MULTISAMPLE";
	spGLSLType[GL_INT_IMAGE_2D_MULTISAMPLE_ARRAY] = "GL_INT_IMAGE_2D_MULTISAMPLE_ARRAY";

	spGLSLType[GL_UNSIGNED_INT_IMAGE_1D] = "GL_UNSIGNED_INT_IMAGE_1D";
	spGLSLType[GL_UNSIGNED_INT_IMAGE_2D] = "GL_UNSIGNED_INT_IMAGE_2D";
	spGLSLType[GL_UNSIGNED_INT_IMAGE_3D] = "GL_UNSIGNED_INT_IMAGE_3D";
	spGLSLType[GL_UNSIGNED_INT_IMAGE_2D_RECT] = "GL_UNSIGNED_INT_IMAGE_2D_RECT";
	spGLSLType[GL_UNSIGNED_INT_IMAGE_CUBE] = "GL_UNSIGNED_INT_IMAGE_CUBE";
	spGLSLType[GL_UNSIGNED_INT_IMAGE_BUFFER] = "GL_UNSIGNED_INT_IMAGE_BUFFER";
	spGLSLType[GL_UNSIGNED_INT_IMAGE_1D_ARRAY] = "GL_UNSIGNED_INT_IMAGE_1D_ARRAY";
	spGLSLType[GL_UNSIGNED_INT_IMAGE_2D_ARRAY] = "GL_UNSIGNED_INT_IMAGE_2D_ARRAY";
	spGLSLType[GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE] = "GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE";
	spGLSLType[GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_ARRAY] = "GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_ARRAY";

	spGLSLType[GL_UNSIGNED_INT_ATOMIC_COUNTER] = "GL_UNSIGNED_INT_ATOMIC_COUNTER";

	return true;
}
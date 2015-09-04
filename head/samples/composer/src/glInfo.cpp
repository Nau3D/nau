/** ----------------------------------------------------------
 * \class VSGLInfoLib
 *
 * Lighthouse3D
 *
 * VSGLInfoLib - Very Simple GL Information Library
 *
 *	
 * \version 0.1.0
 *  - Initial Release
 *
 * This class provides information about GL stuff
 *
 * Full documentation at 
 * http://www.lighthouse3d.com/very-simple-libs
 *
 ---------------------------------------------------------------*/

#include "glInfo.h"

#include <nau/config.h>

enum Types {
		DONT_KNOW, INT, UNSIGNED_INT, FLOAT, DOUBLE};
bool init();
Types getType(GLenum type);
int getRows(GLenum type);
int getColumns(GLenum type);
int getUniformByteSize(int size, int uniType, int arrayStride, int matStride);
void mapCurrentBufferNames();

//std::map<int, NauGlBufferInfo> buffermapping;



// local variables
std::map<int, std::string> spInternalF;
std::map<int, std::string> spDataF;
std::map<int, std::string> spTextureDataType;
std::map<int, std::string> spGLSLType;
std::map<int, int> spGLSLTypeSize;
std::map<int, std::string> spTextureFilter;
std::map<int, std::string> spTextureWrap;
std::map<int, std::string> spTextureCompFunc;
std::map<int, std::string> spTextureCompMode;
std::map<int, std::string> spTextureUnit;
std::map<int, int> spTextureBound;
std::map<int, std::string> spHint;
std::map<int, std::string> spTextureTarget;
std::map<int, std::string> spBufferAccess;
std::map<int, std::string> spBufferUsage;
std::map<int, std::string> spBufferBinding;
std::map<int, int> spBufferBound;
std::map<int, int> spBoundBuffer;
std::map<int, std::string> spShaderType;
std::map<int, std::string> spTransFeedBufferMode;
std::map<int, std::string> spGLSLPrimitives;
std::map<int, std::string> spTessGenSpacing;
std::map<int, std::string> spVertexOrder;
std::map<int, std::string> spShaderPrecision;

std::vector<unsigned int> spResult;
bool __spInit = init();
char spAux[256];





// check if an extension is supported
bool
isExtensionSupported(std::string extName) {

	int max, i = 0;
	char *s;

	glGetIntegerv(GL_NUM_EXTENSIONS, &max);
	do {
		s = (char *)glGetStringi(GL_EXTENSIONS, ++i);
	}
	while (i < max && strcmp(s,extName.c_str()) != 0);

	if (i == max)
		return false;
	else
		return true;
}



// gets all the names currently bound to buffers
std::vector<unsigned int> &
getBufferNames() {

	spResult.clear();
	for (unsigned int i = 0; i < 65535; ++i) {

		if (glIsBuffer(i)) {
			spResult.push_back(i);


		}
	}
	return spResult;
}



/* ------------------------------------------------------

		GLSL

-------------------------------------------------------- */


// gets all the names currently boundo to programs
std::vector<unsigned int> &
getProgramNames() {

	spResult.clear();
	for (unsigned int i = 0; i < 65535; ++i) {

		if (glIsProgram(i))
		 spResult.push_back(i);
	}
	return spResult;
}


//// gets all the names currently bound to Shaders
//std::vector<unsigned int> &
//getShaderNames() {
//
//	spResult.clear();
//	for (unsigned int i = 0; i < 65535; ++i) {
//
//		if (glIsShader(i))
//		 spResult.push_back(i);
//	}
//	return spResult;
//}


//// gets all the names currently bound to VAOs
//std::vector<unsigned int> &
//getVAONames() {
//
//	spResult.clear();
//	for (unsigned int i = 0; i < 65535; ++i) {
//
//		if (glIsVertexArray(i))
//		 spResult.push_back(i);
//	}
//	return spResult;
//}



/* ----------------------------------------------

		private auxiliary functions

----------------------------------------------- */


// init the library
// fills up the maps with enum to string
// to display human-readable messages
bool 
init() {

	spShaderPrecision[GL_LOW_FLOAT] = "GL_LOW_FLOAT";
	spShaderPrecision[GL_MEDIUM_FLOAT] = "GL_MEDIUM_FLOAT";
	spShaderPrecision[GL_HIGH_FLOAT] = "GL_HIGH_FLOAT";
	spShaderPrecision[GL_LOW_INT] = "GL_LOW_INT";
	spShaderPrecision[GL_MEDIUM_INT] = "GL_MEDIUM_INT";
	spShaderPrecision[GL_HIGH_INT] = "GL_HIGH_INT";

	spTessGenSpacing[GL_EQUAL] = "GL_EQUAL";
	spTessGenSpacing[GL_FRACTIONAL_EVEN] = "GL_FRACTIONAL_EVEN";
	spTessGenSpacing[GL_FRACTIONAL_ODD] = "GL_FRACTIONAL_ODD";

	spVertexOrder[GL_CCW] = "GL_CCW";
	spVertexOrder[GL_CW] = "GL_CW";

	spGLSLPrimitives[GL_QUADS] = "GL_QUADS";
	spGLSLPrimitives[GL_ISOLINES] = "GL_ISOLINES";
	spGLSLPrimitives[GL_POINTS] = "GL_POINTS";
	spGLSLPrimitives[GL_LINES] = "GL_LINES";
	spGLSLPrimitives[GL_LINES_ADJACENCY] = "GL_LINES_ADJACENCY";
	spGLSLPrimitives[GL_TRIANGLES] = "GL_TRIANGLES";
	spGLSLPrimitives[GL_LINE_STRIP] = "GL_LINE_STRIP";
	spGLSLPrimitives[GL_TRIANGLE_STRIP] = "GL_TRIANGLE_STRIP";
	spGLSLPrimitives[GL_TRIANGLES_ADJACENCY] = "GL_TRIANGLES_ADJACENCY";

	spTransFeedBufferMode[GL_SEPARATE_ATTRIBS] = "GL_SEPARATE_ATTRIBS";
	spTransFeedBufferMode[GL_INTERLEAVED_ATTRIBS] = "GL_INTERLEAVED_ATTRIBS";

	spShaderType[GL_VERTEX_SHADER] = "GL_VERTEX_SHADER";
	spShaderType[GL_GEOMETRY_SHADER] = "GL_GEOMETRY_SHADER";
	spShaderType[GL_TESS_CONTROL_SHADER] = "GL_TESS_CONTROL_SHADER";
	spShaderType[GL_TESS_EVALUATION_SHADER] = "GL_TESS_EVALUATION_SHADER";
	spShaderType[GL_FRAGMENT_SHADER] = "GL_FRAGMENT_SHADER";

	spHint[GL_FASTEST] = "GL_FASTEST";
	spHint[GL_NICEST] = "GL_NICEST";
	spHint[GL_DONT_CARE] = "GL_DONT_CARE";

	spBufferBinding[GL_ARRAY_BUFFER_BINDING] = "GL_ARRAY_BUFFER";
	spBufferBinding[GL_ELEMENT_ARRAY_BUFFER_BINDING] = "GL_ELEMENT_ARRAY_BUFFER";
	spBufferBinding[GL_PIXEL_PACK_BUFFER_BINDING] = "GL_PIXEL_PACK_BUFFER";
	spBufferBinding[GL_PIXEL_UNPACK_BUFFER_BINDING] = "GL_PIXEL_UNPACK_BUFFER";
	spBufferBinding[GL_TRANSFORM_FEEDBACK_BUFFER_BINDING] = "GL_TRANSFORM_FEEDBACK_BUFFER";
	spBufferBinding[GL_UNIFORM_BUFFER_BINDING] = "GL_UNIFORM_BUFFER";

//#if (NAU_OPENGL_VERSION >= 420)
	//spBufferBinding[GL_TEXTURE_BUFFER_BINDING] = "GL_TEXTURE_BUFFER";
	//spBufferBinding[GL_COPY_READ_BUFFER_BINDING] = "GL_COPY_READ_BUFFER";
	//spBufferBinding[GL_COPY_WRITE_BUFFER_BINDING] = "GL_COPY_WRITE_BUFFER";
	spBufferBinding[GL_DRAW_INDIRECT_BUFFER_BINDING] = "GL_DRAW_INDIRECT_BUFFER";
	spBufferBinding[GL_ATOMIC_COUNTER_BUFFER_BINDING] = "GL_ATOMIC_COUNTER_BUFFER";
//#endif

	spBufferBound[GL_ARRAY_BUFFER_BINDING] = GL_ARRAY_BUFFER;
	spBufferBound[GL_ELEMENT_ARRAY_BUFFER_BINDING] = GL_ELEMENT_ARRAY_BUFFER;
	spBufferBound[GL_PIXEL_PACK_BUFFER_BINDING] = GL_PIXEL_PACK_BUFFER;
	spBufferBound[GL_PIXEL_UNPACK_BUFFER_BINDING] = GL_PIXEL_UNPACK_BUFFER;
	spBufferBound[GL_TRANSFORM_FEEDBACK_BUFFER_BINDING] = GL_TRANSFORM_FEEDBACK_BUFFER;
	spBufferBound[GL_UNIFORM_BUFFER_BINDING] = GL_UNIFORM_BUFFER;

//#if (NAU_OPENGL_VERSION >= 420)
	//spBufferBound[GL_TEXTURE_BUFFER_BINDING] = GL_TEXTURE_BUFFER;
	//spBufferBound[GL_COPY_READ_BUFFER_BINDING] = GL_COPY_READ_BUFFER;
	//spBufferBound[GL_COPY_WRITE_BUFFER_BINDING] = GL_COPY_WRITE_BUFFER;
	spBufferBound[GL_DRAW_INDIRECT_BUFFER_BINDING] = GL_DRAW_INDIRECT_BUFFER;
	spBufferBound[GL_ATOMIC_COUNTER_BUFFER_BINDING] = GL_ATOMIC_COUNTER_BUFFER;
//#endif

	spBoundBuffer[GL_ARRAY_BUFFER] = GL_ARRAY_BUFFER_BINDING;
	spBoundBuffer[GL_ELEMENT_ARRAY_BUFFER] = GL_ELEMENT_ARRAY_BUFFER_BINDING;
	spBoundBuffer[GL_PIXEL_PACK_BUFFER] = GL_PIXEL_PACK_BUFFER_BINDING;
	spBoundBuffer[GL_PIXEL_UNPACK_BUFFER] = GL_PIXEL_UNPACK_BUFFER_BINDING;
	spBoundBuffer[GL_TRANSFORM_FEEDBACK_BUFFER] = GL_TRANSFORM_FEEDBACK_BUFFER_BINDING;
	spBoundBuffer[GL_UNIFORM_BUFFER] = GL_UNIFORM_BUFFER_BINDING;

//#if (NAU_OPENGL_VERSION >= 420)
	//spBoundBuffer[GL_TEXTURE_BUFFER] = GL_TEXTURE_BUFFER_BINDING;
	//spBoundBuffer[GL_COPY_READ_BUFFER] = GL_COPY_READ_BUFFER_BINDING;
	//spBoundBuffer[GL_COPY_WRITE_BUFFER] = GL_COPY_WRITE_BUFFER_BINDING;
	spBoundBuffer[GL_DRAW_INDIRECT_BUFFER] = GL_DRAW_INDIRECT_BUFFER;
	spBoundBuffer[GL_ATOMIC_COUNTER_BUFFER] = GL_ATOMIC_COUNTER_BUFFER;
//#endif

	spBufferUsage[GL_STREAM_DRAW] = "GL_STREAM_DRAW";
	spBufferUsage[GL_STREAM_READ] = "GL_STREAM_READ";
	spBufferUsage[GL_STREAM_COPY] = "GL_STREAM_COPY";
	spBufferUsage[GL_STATIC_DRAW] = "GL_STATIC_DRAW";
	spBufferUsage[GL_STATIC_READ] = "GL_STATIC_READ";
	spBufferUsage[GL_STATIC_COPY] = "GL_STATIC_COPY";
	spBufferUsage[GL_DYNAMIC_DRAW] = "GL_DYNAMIC_DRAW";
	spBufferUsage[GL_DYNAMIC_READ] = "GL_DYNAMIC_READ";
	spBufferUsage[GL_DYNAMIC_COPY] = "GL_DYNAMIC_COPY";

	spBufferAccess[GL_READ_ONLY] = "GL_READ_ONLY";
	spBufferAccess[GL_WRITE_ONLY] = "GL_WRITE_ONLY";
	spBufferAccess[GL_READ_WRITE] = "GL_READ_WRITE";

	spTextureTarget[GL_TEXTURE_1D] = "GL_TEXTURE_1D";
	spTextureTarget[GL_TEXTURE_1D_ARRAY] = "GL_TEXTURE_1D_ARRAY";
	spTextureTarget[GL_TEXTURE_2D] = "GL_TEXTURE_2D";
	spTextureTarget[GL_TEXTURE_2D_ARRAY] = "GL_TEXTURE_2D_ARRAY";
	spTextureTarget[GL_TEXTURE_2D_MULTISAMPLE] = "GL_TEXTURE_2D_MULTISAMPLE";
	spTextureTarget[GL_TEXTURE_2D_MULTISAMPLE_ARRAY] = "GL_TEXTURE_2D_MULTISAMPLE_ARRAY";
	spTextureTarget[GL_TEXTURE_3D] = "GL_TEXTURE_3D";
	spTextureTarget[GL_TEXTURE_BUFFER] = "GL_TEXTURE_BUFFER";
	spTextureTarget[GL_TEXTURE_CUBE_MAP] = "GL_TEXTURE_CUBE_MAP";
	spTextureTarget[GL_TEXTURE_RECTANGLE] = "GL_TEXTURE_RECTANGLE";

	spTextureBound[GL_TEXTURE_1D] = GL_TEXTURE_BINDING_1D;
	spTextureBound[GL_TEXTURE_1D_ARRAY] = GL_TEXTURE_BINDING_1D_ARRAY;
	spTextureBound[GL_TEXTURE_2D] = GL_TEXTURE_BINDING_2D;
	spTextureBound[GL_TEXTURE_2D_ARRAY] = GL_TEXTURE_BINDING_2D_ARRAY;
	spTextureBound[GL_TEXTURE_2D_MULTISAMPLE] = GL_TEXTURE_BINDING_2D_MULTISAMPLE;
	spTextureBound[GL_TEXTURE_2D_MULTISAMPLE_ARRAY] = GL_TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY;
	spTextureBound[GL_TEXTURE_3D] = GL_TEXTURE_BINDING_3D;
	spTextureBound[GL_TEXTURE_BUFFER] = GL_TEXTURE_BINDING_BUFFER;
	spTextureBound[GL_TEXTURE_CUBE_MAP] = GL_TEXTURE_BINDING_CUBE_MAP;
	spTextureBound[GL_TEXTURE_RECTANGLE] = GL_TEXTURE_BINDING_RECTANGLE;

	spTextureUnit[GL_TEXTURE0] = "GL_TEXTURE0";
	spTextureUnit[GL_TEXTURE1] = "GL_TEXTURE1";
	spTextureUnit[GL_TEXTURE2] = "GL_TEXTURE2";
	spTextureUnit[GL_TEXTURE3] = "GL_TEXTURE3";
	spTextureUnit[GL_TEXTURE4] = "GL_TEXTURE4";
	spTextureUnit[GL_TEXTURE5] = "GL_TEXTURE5";
	spTextureUnit[GL_TEXTURE6] = "GL_TEXTURE6";
	spTextureUnit[GL_TEXTURE7] = "GL_TEXTURE7";

	spTextureCompMode[GL_NONE] = "GL_NONE";	
	spTextureCompFunc[GL_COMPARE_REF_TO_TEXTURE] = "GL_COMPARE_REF_TO_TEXTURE";	

	spTextureCompFunc[GL_LEQUAL] = "GL_LEQUAL";	
	spTextureCompFunc[GL_GEQUAL] = "GL_GEQUAL";	
	spTextureCompFunc[GL_LESS] = "GL_LESS";	
	spTextureCompFunc[GL_GREATER] = "GL_GREATER";	
	spTextureCompFunc[GL_EQUAL] = "GL_EQUAL";	
	spTextureCompFunc[GL_NOTEQUAL] = "GL_NOTEQUAL";	
	spTextureCompFunc[GL_ALWAYS] = "GL_ALWAYS";	
	spTextureCompFunc[GL_NEVER] = "GL_NEVER";

	spTextureWrap[GL_CLAMP_TO_EDGE] = "GL_CLAMP_TO_EDGE";
	spTextureWrap[GL_CLAMP_TO_BORDER] = "GL_CLAMP_TO_BORDER";
	spTextureWrap[GL_MIRRORED_REPEAT] = "GL_MIRRORED_REPEAT"; 
	spTextureWrap[GL_REPEAT] = "GL_REPEAT";

	spTextureFilter[GL_NEAREST] = "GL_NEAREST";
	spTextureFilter[GL_LINEAR] = "GL_LINEAR";
	spTextureFilter[GL_NEAREST_MIPMAP_NEAREST] = "GL_NEAREST_MIPMAP_NEAREST";
	spTextureFilter[GL_LINEAR_MIPMAP_NEAREST] = "GL_LINEAR_MIPMAP_NEAREST";
	spTextureFilter[GL_NEAREST_MIPMAP_LINEAR] = "GL_NEAREST_MIPMAP_LINEAR";
	spTextureFilter[GL_LINEAR_MIPMAP_LINEAR] = "GL_LINEAR_MIPMAP_LINEAR";

	spGLSLTypeSize[GL_FLOAT] = sizeof(float); 
	spGLSLTypeSize[GL_FLOAT_VEC2] = sizeof(float)*2; 
	spGLSLTypeSize[GL_FLOAT_VEC3] = sizeof(float)*3; 
	spGLSLTypeSize[GL_FLOAT_VEC4] = sizeof(float)*4; 

	spGLSLTypeSize[GL_DOUBLE] = sizeof(double); 
	spGLSLTypeSize[GL_DOUBLE_VEC2] = sizeof(double)*2; 
	spGLSLTypeSize[GL_DOUBLE_VEC3] = sizeof(double)*3; 
	spGLSLTypeSize[GL_DOUBLE_VEC4] = sizeof(double)*4; 

	spGLSLTypeSize[GL_SAMPLER_1D] = sizeof(int); 
	spGLSLTypeSize[GL_SAMPLER_2D] = sizeof(int); 
	spGLSLTypeSize[GL_SAMPLER_3D] = sizeof(int); 
	spGLSLTypeSize[GL_SAMPLER_CUBE] = sizeof(int); 
	spGLSLTypeSize[GL_SAMPLER_1D_SHADOW] = sizeof(int); 
	spGLSLTypeSize[GL_SAMPLER_2D_SHADOW] = sizeof(int); 
	spGLSLTypeSize[GL_SAMPLER_1D_ARRAY] = sizeof(int); 
	spGLSLTypeSize[GL_SAMPLER_2D_ARRAY] = sizeof(int); 
	spGLSLTypeSize[GL_SAMPLER_1D_ARRAY_SHADOW] = sizeof(int); 
	spGLSLTypeSize[GL_SAMPLER_2D_ARRAY_SHADOW] = sizeof(int); 
	spGLSLTypeSize[GL_SAMPLER_2D_MULTISAMPLE] = sizeof(int); 
	spGLSLTypeSize[GL_SAMPLER_2D_MULTISAMPLE_ARRAY] = sizeof(int); 
	spGLSLTypeSize[GL_SAMPLER_CUBE_SHADOW] = sizeof(int); 
	spGLSLTypeSize[GL_SAMPLER_BUFFER] = sizeof(int); 
	spGLSLTypeSize[GL_SAMPLER_2D_RECT] = sizeof(int); 
	spGLSLTypeSize[GL_SAMPLER_2D_RECT_SHADOW] = sizeof(int); 
	spGLSLTypeSize[GL_INT_SAMPLER_1D] = sizeof(int); 
	spGLSLTypeSize[GL_INT_SAMPLER_2D] = sizeof(int); 
	spGLSLTypeSize[GL_INT_SAMPLER_3D] = sizeof(int); 
	spGLSLTypeSize[GL_INT_SAMPLER_CUBE] = sizeof(int); 
	spGLSLTypeSize[GL_INT_SAMPLER_1D_ARRAY] = sizeof(int); 
	spGLSLTypeSize[GL_INT_SAMPLER_2D_ARRAY] = sizeof(int); 
	spGLSLTypeSize[GL_INT_SAMPLER_2D_MULTISAMPLE] = sizeof(int); 
	spGLSLTypeSize[GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY] = sizeof(int); 
	spGLSLTypeSize[GL_INT_SAMPLER_BUFFER] = sizeof(int); 
	spGLSLTypeSize[GL_INT_SAMPLER_2D_RECT] = sizeof(int); 
	spGLSLTypeSize[GL_UNSIGNED_INT_SAMPLER_1D] = sizeof(int); 
	spGLSLTypeSize[GL_UNSIGNED_INT_SAMPLER_2D] = sizeof(int); 
	spGLSLTypeSize[GL_UNSIGNED_INT_SAMPLER_3D] = sizeof(int); 
	spGLSLTypeSize[GL_UNSIGNED_INT_SAMPLER_CUBE] = sizeof(int); 
	spGLSLTypeSize[GL_UNSIGNED_INT_SAMPLER_1D_ARRAY] = sizeof(int); 
	spGLSLTypeSize[GL_UNSIGNED_INT_SAMPLER_2D_ARRAY] = sizeof(int); 
	spGLSLTypeSize[GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE] = sizeof(int); 
	spGLSLTypeSize[GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY] = sizeof(int); 
	spGLSLTypeSize[GL_UNSIGNED_INT_SAMPLER_BUFFER] = sizeof(int); 
	spGLSLTypeSize[GL_UNSIGNED_INT_SAMPLER_2D_RECT] = sizeof(int); 
	spGLSLTypeSize[GL_BOOL] = sizeof(int); 
	spGLSLTypeSize[GL_INT] = sizeof(int); 
	spGLSLTypeSize[GL_BOOL_VEC2] = sizeof(int)*2; 
	spGLSLTypeSize[GL_INT_VEC2] = sizeof(int)*2; 
	spGLSLTypeSize[GL_BOOL_VEC3] = sizeof(int)*3; 
	spGLSLTypeSize[GL_INT_VEC3] = sizeof(int)*3;  
	spGLSLTypeSize[GL_BOOL_VEC4] = sizeof(int)*4; 
	spGLSLTypeSize[GL_INT_VEC4] = sizeof(int)*4; 

	spGLSLTypeSize[GL_UNSIGNED_INT] = sizeof(int); 
	spGLSLTypeSize[GL_UNSIGNED_INT_VEC2] = sizeof(int)*2; 
	spGLSLTypeSize[GL_UNSIGNED_INT_VEC3] = sizeof(int)*2; 
	spGLSLTypeSize[GL_UNSIGNED_INT_VEC4] = sizeof(int)*2; 

	spGLSLTypeSize[GL_FLOAT_MAT2] = sizeof(float)*4; 
	spGLSLTypeSize[GL_FLOAT_MAT3] = sizeof(float)*9; 
	spGLSLTypeSize[GL_FLOAT_MAT4] = sizeof(float)*16; 
	spGLSLTypeSize[GL_FLOAT_MAT2x3] = sizeof(float)*6; 
	spGLSLTypeSize[GL_FLOAT_MAT2x4] = sizeof(float)*8; 
	spGLSLTypeSize[GL_FLOAT_MAT3x2] = sizeof(float)*6; 
	spGLSLTypeSize[GL_FLOAT_MAT3x4] = sizeof(float)*12; 
	spGLSLTypeSize[GL_FLOAT_MAT4x2] = sizeof(float)*8; 
	spGLSLTypeSize[GL_FLOAT_MAT4x3] = sizeof(float)*12; 
	spGLSLTypeSize[GL_DOUBLE_MAT2] = sizeof(double)*4; 
	spGLSLTypeSize[GL_DOUBLE_MAT3] = sizeof(double)*9; 
	spGLSLTypeSize[GL_DOUBLE_MAT4] = sizeof(double)*16; 
	spGLSLTypeSize[GL_DOUBLE_MAT2x3] = sizeof(double)*6; 
	spGLSLTypeSize[GL_DOUBLE_MAT2x4] = sizeof(double)*8; 
	spGLSLTypeSize[GL_DOUBLE_MAT3x2] = sizeof(double)*6; 
	spGLSLTypeSize[GL_DOUBLE_MAT3x4] = sizeof(double)*12; 
	spGLSLTypeSize[GL_DOUBLE_MAT4x2] = sizeof(double)*8; 
	spGLSLTypeSize[GL_DOUBLE_MAT4x3] = sizeof(double)*12; 



	spGLSLType[GL_FLOAT] = "GL_FLOAT"; 
	spGLSLType[GL_FLOAT_VEC2] = "GL_FLOAT_VEC2";  
	spGLSLType[GL_FLOAT_VEC3] = "GL_FLOAT_VEC3";  
	spGLSLType[GL_FLOAT_VEC4] = "GL_FLOAT_VEC4";  
	spGLSLType[GL_DOUBLE] = "GL_DOUBLE"; 
	spGLSLType[GL_DOUBLE_VEC2] = "GL_DOUBLE_VEC2";  
	spGLSLType[GL_DOUBLE_VEC3] = "GL_DOUBLE_VEC3";  
	spGLSLType[GL_DOUBLE_VEC4] = "GL_DOUBLE_VEC4";  
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
	spGLSLType[GL_BOOL] = "GL_BOOL";  
	spGLSLType[GL_INT] = "GL_INT";  
	spGLSLType[GL_BOOL_VEC2] = "GL_BOOL_VEC2";
	spGLSLType[GL_INT_VEC2] = "GL_INT_VEC2";  
	spGLSLType[GL_BOOL_VEC3] = "GL_BOOL_VEC3";
	spGLSLType[GL_INT_VEC3] = "GL_INT_VEC3";  
	spGLSLType[GL_BOOL_VEC4] = "GL_BOOL_VEC4";
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



	spTextureDataType[GL_NONE] = "GL_NONE";
	spTextureDataType[GL_SIGNED_NORMALIZED] = "GL_SIGNED_NORMALIZED";
	spTextureDataType[GL_UNSIGNED_NORMALIZED] = "GL_UNSIGNED_NORMALIZED";
	spTextureDataType[GL_FLOAT] = "GL_FLOAT";
	spTextureDataType[GL_INT] = "GL_INT";
	spTextureDataType[GL_UNSIGNED_INT] = "GL_UNSIGNED_INT";

	spDataF[GL_UNSIGNED_BYTE] = "GL_UNSIGNED_BYTE";
	spDataF[GL_BYTE] = "GL_BYTE";
	spDataF[GL_UNSIGNED_SHORT] = "GL_UNSIGNED_SHORT";
	spDataF[GL_SHORT] = "GL_SHORT";
	spDataF[GL_UNSIGNED_INT] = "GL_UNSIGNED_INT";
	spDataF[GL_INT] = "GL_INT";
	spDataF[GL_HALF_FLOAT] = "GL_HALF_FLOAT";
	spDataF[GL_FLOAT] = "GL_FLOAT";

	spDataF[GL_UNSIGNED_BYTE_3_3_2] = "GL_UNSIGNED_BYTE_3_3_2";
	spDataF[GL_UNSIGNED_BYTE_2_3_3_REV] = "GL_UNSIGNED_BYTE_2_3_3_REV";
	spDataF[GL_UNSIGNED_SHORT_5_6_5] = "GL_UNSIGNED_SHORT_5_6_5";
	spDataF[GL_UNSIGNED_SHORT_5_6_5_REV] = "GL_UNSIGNED_SHORT_5_6_5_REV";
	spDataF[GL_UNSIGNED_SHORT_4_4_4_4] = "GL_UNSIGNED_SHORT_4_4_4_4";
	spDataF[GL_UNSIGNED_SHORT_4_4_4_4_REV] = "GL_UNSIGNED_SHORT_4_4_4_4_REV";
	spDataF[GL_UNSIGNED_SHORT_5_5_5_1] = "GL_UNSIGNED_SHORT_5_5_5_1";
	spDataF[GL_UNSIGNED_SHORT_1_5_5_5_REV] = "GL_UNSIGNED_SHORT_1_5_5_5_REV";
	spDataF[GL_UNSIGNED_INT_8_8_8_8] = "GL_UNSIGNED_INT_8_8_8_8";
	spDataF[GL_UNSIGNED_INT_8_8_8_8_REV] = "GL_UNSIGNED_INT_8_8_8_8_REV";
	spDataF[GL_UNSIGNED_INT_10_10_10_2] = "GL_UNSIGNED_INT_10_10_10_2";
	spDataF[GL_UNSIGNED_INT_2_10_10_10_REV] = "GL_UNSIGNED_INT_2_10_10_10_REV";

	spInternalF[GL_STENCIL_INDEX] = "GL_STENCIL_INDEX";
	spInternalF[GL_DEPTH_COMPONENT] = "GL_DEPTH_COMPONENT";
	spInternalF[GL_DEPTH_STENCIL] = "GL_DEPTH_STENCIL";
	spInternalF[GL_DEPTH_COMPONENT16] = "GL_DEPTH_COMPONENT16";
	spInternalF[GL_DEPTH_COMPONENT24] = "GL_DEPTH_COMPONENT24";
	spInternalF[GL_DEPTH_COMPONENT32] = "GL_DEPTH_COMPONENT32";
	spInternalF[GL_DEPTH_COMPONENT32F] = "GL_DEPTH_COMPONENT32F";
	spInternalF[GL_DEPTH24_STENCIL8] = "GL_DEPTH24_STENCIL8";
	spInternalF[GL_DEPTH32F_STENCIL8] = "GL_DEPTH32F_STENCIL8";
	spInternalF[GL_RED_INTEGER] = "GL_RED_INTEGER";
	spInternalF[GL_GREEN_INTEGER] = "GL_GREEN_INTEGER";
	spInternalF[GL_BLUE_INTEGER] = "GL_BLUE_INTEGER";

	spInternalF[GL_RG_INTEGER] = "GL_RG_INTEGER";
	spInternalF[GL_RGB_INTEGER] = "GL_RGB_INTEGER";
	spInternalF[GL_RGBA_INTEGER] = "GL_RGBA_INTEGER";
	spInternalF[GL_BGR_INTEGER] = "GL_BGR_INTEGER";
	spInternalF[GL_BGRA_INTEGER] = "GL_BGRA_INTEGER";

	spInternalF[GL_RED] = "GL_RED";
	spInternalF[GL_RG] = "GL_RG";
	spInternalF[GL_RGB] = "GL_RGB";
	spInternalF[GL_RGBA] = "GL_RGBA";
	spInternalF[GL_R3_G3_B2] = "GL_R3_G3_B2";
	spInternalF[GL_RGB2_EXT] = "GL_RGB2_EXT";
	spInternalF[GL_COMPRESSED_RED] = "GL_COMPRESSED_RED";
	spInternalF[GL_COMPRESSED_RG] = "GL_COMPRESSED_RG";
	spInternalF[GL_COMPRESSED_RGB] = "GL_COMPRESSED_RGB";
	spInternalF[GL_COMPRESSED_RGBA] = "GL_COMPRESSED_RGBA";
	spInternalF[GL_COMPRESSED_SRGB] = "GL_COMPRESSED_SRGB";
	spInternalF[GL_COMPRESSED_SRGB_ALPHA] = "GL_COMPRESSED_SRGB_ALPHA";
	spInternalF[GL_COMPRESSED_RED_RGTC1] = "GL_COMPRESSED_RED_RGTC1";
	spInternalF[GL_COMPRESSED_SIGNED_RED_RGTC1] = "GL_COMPRESSED_SIGNED_RED_RGTC1";
	spInternalF[GL_COMPRESSED_RG_RGTC2] = "GL_COMPRESSED_RG_RGTC2";
	spInternalF[GL_COMPRESSED_SIGNED_RG_RGTC2] = "GL_COMPRESSED_SIGNED_RG_RGTC2";

	spInternalF[GL_COMPRESSED_RGBA_BPTC_UNORM] = "GL_COMPRESSED_RGBA_BPTC_UNORM";
	spInternalF[GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM] = "GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM";
	spInternalF[GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT] = "GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT";
	spInternalF[GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT] = "GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT";

	spInternalF[GL_R8] = "GL_R8";
	spInternalF[GL_R16] = "GL_R16";
	spInternalF[GL_RG8] = "GL_RG8";
	spInternalF[GL_RG16] = "GL_RG16";
	spInternalF[GL_R16F] = "GL_R16F";
	spInternalF[GL_R32F] = "GL_R32F";
	spInternalF[GL_RG16F] = "GL_RG16F";
	spInternalF[GL_RG32F] = "GL_RG32F";
	spInternalF[GL_R8I] = "GL_R8I";
	spInternalF[GL_R8UI] = "GL_R8UI";
	spInternalF[GL_R16I] = "GL_R16I";
	spInternalF[GL_R16UI] = "GL_R16UI";
	spInternalF[GL_R32I] = "GL_R32I";
	spInternalF[GL_R32UI] = "GL_R32UI";
	spInternalF[GL_RG8I] = "GL_RG8I";
	spInternalF[GL_RG8UI] = "GL_RG8UI";
	spInternalF[GL_RG16I] = "GL_RG16I";
	spInternalF[GL_RG16UI] = "GL_RG16UI";
	spInternalF[GL_RG32I] = "GL_RG32I";
	spInternalF[GL_RG32UI] = "GL_RG32UI";
	spInternalF[GL_RGB_S3TC] = "GL_RGB_S3TC";
	spInternalF[GL_RGB4_S3TC] = "GL_RGB4_S3TC";
	spInternalF[GL_RGBA_S3TC] = "GL_RGBA_S3TC";
	spInternalF[GL_RGBA4_S3TC] = "GL_RGBA4_S3TC";
	spInternalF[GL_RGBA_DXT5_S3TC] = "GL_RGBA_DXT5_S3TC";
	spInternalF[GL_RGBA4_DXT5_S3TC] = "GL_RGBA4_DXT5_S3TC";
	spInternalF[GL_COMPRESSED_RGB_S3TC_DXT1_EXT] = "GL_COMPRESSED_RGB_S3TC_DXT1_EXT";
	spInternalF[GL_COMPRESSED_RGBA_S3TC_DXT1_EXT] = "GL_COMPRESSED_RGBA_S3TC_DXT1_EXT";
	spInternalF[GL_COMPRESSED_RGBA_S3TC_DXT3_EXT] = "GL_COMPRESSED_RGBA_S3TC_DXT3_EXT";
	spInternalF[GL_COMPRESSED_RGBA_S3TC_DXT5_EXT] = "GL_COMPRESSED_RGBA_S3TC_DXT5_EXT";
	spInternalF[GL_R1UI_V3F_SUN] = "GL_R1UI_V3F_SUN";
	spInternalF[GL_R1UI_C4UB_V3F_SUN] = "GL_R1UI_C4UB_V3F_SUN";
	spInternalF[GL_R1UI_C3F_V3F_SUN] = "GL_R1UI_C3F_V3F_SUN";
	spInternalF[GL_R1UI_N3F_V3F_SUN] = "GL_R1UI_N3F_V3F_SUN";
	spInternalF[GL_R1UI_C4F_N3F_V3F_SUN] = "GL_R1UI_C4F_N3F_V3F_SUN";
	spInternalF[GL_R1UI_T2F_V3F_SUN] = "GL_R1UI_T2F_V3F_SUN";
	spInternalF[GL_R1UI_T2F_N3F_V3F_SUN] = "GL_R1UI_T2F_N3F_V3F_SUN";
	spInternalF[GL_R1UI_T2F_C4F_N3F_V3F_SUN] = "GL_R1UI_T2F_C4F_N3F_V3F_SUN";
	spInternalF[GL_RGB_SIGNED_SGIX] = "GL_RGB_SIGNED_SGIX";
	spInternalF[GL_RGBA_SIGNED_SGIX] = "GL_RGBA_SIGNED_SGIX";
	spInternalF[GL_RGB16_SIGNED_SGIX] = "GL_RGB16_SIGNED_SGIX";
	spInternalF[GL_RGBA16_SIGNED_SGIX] = "GL_RGBA16_SIGNED_SGIX";
	spInternalF[GL_RGB_EXTENDED_RANGE_SGIX] = "GL_RGB_EXTENDED_RANGE_SGIX";
	spInternalF[GL_RGBA_EXTENDED_RANGE_SGIX] = "GL_RGBA_EXTENDED_RANGE_SGIX";
	spInternalF[GL_RGB16_EXTENDED_RANGE_SGIX] = "GL_RGB16_EXTENDED_RANGE_SGIX";
	spInternalF[GL_RGBA16_EXTENDED_RANGE_SGIX] = "GL_RGBA16_EXTENDED_RANGE_SGIX";
	spInternalF[GL_COMPRESSED_RGB_FXT1_3DFX] = "GL_COMPRESSED_RGB_FXT1_3DFX";
	spInternalF[GL_COMPRESSED_RGBA_FXT1_3DFX] = "GL_COMPRESSED_RGBA_FXT1_3DFX";
	spInternalF[GL_RGBA_UNSIGNED_DOT_PRODUCT_MAPPING_NV] = "GL_RGBA_UNSIGNED_DOT_PRODUCT_MAPPING_NV";
	spInternalF[GL_RGBA_FLOAT_MODE_ARB] = "GL_RGBA_FLOAT_MODE_ARB";
	spInternalF[GL_COMPRESSED_LUMINANCE_ALPHA_3DC_ATI] = "GL_COMPRESSED_LUMINANCE_ALPHA_3DC_ATI";
	spInternalF[GL_RGB_422_APPLE] = "GL_RGB_422_APPLE";
	spInternalF[GL_RGBA_SIGNED_COMPONENTS_EXT] = "GL_RGBA_SIGNED_COMPONENTS_EXT";
	spInternalF[GL_COMPRESSED_SRGB_S3TC_DXT1_EXT] = "GL_COMPRESSED_SRGB_S3TC_DXT1_EXT";
	spInternalF[GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT] = "GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT";
	spInternalF[GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT] = "GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT";
	spInternalF[GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT] = "GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT";
	spInternalF[GL_COMPRESSED_LUMINANCE_LATC1_EXT] = "GL_COMPRESSED_LUMINANCE_LATC1_EXT";
	spInternalF[GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT] = "GL_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT";
	spInternalF[GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT] = "GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT";
	spInternalF[GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT] = "GL_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT";
	spInternalF[GL_RGBA_INTEGER_MODE_EXT] = "GL_RGBA_INTEGER_MODE_EXT";
	spInternalF[GL_COMPRESSED_RGBA_BPTC_UNORM_ARB] = "GL_COMPRESSED_RGBA_BPTC_UNORM_ARB";
	spInternalF[GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM_ARB] = "GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM_ARB";
	spInternalF[GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT_ARB] = "GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT_ARB";
	spInternalF[GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_ARB] = "GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_ARB";
	spInternalF[GL_RG_SNORM] = "GL_RG_SNORM";
	spInternalF[GL_RGB_SNORM] = "GL_RGB_SNORM";
	spInternalF[GL_RGBA_SNORM] = "GL_RGBA_SNORM";
	spInternalF[GL_R8_SNORM] = "GL_R8_SNORM";
	spInternalF[GL_RG8_SNORM] = "GL_RG8_SNORM";
	spInternalF[GL_RGB8_SNORM] = "GL_RGB8_SNORM";
	spInternalF[GL_RGBA8_SNORM] = "GL_RGBA8_SNORM";
	spInternalF[GL_R16_SNORM] = "GL_R16_SNORM";
	spInternalF[GL_RG16_SNORM] = "GL_RG16_SNORM";
	spInternalF[GL_RGB16_SNORM] = "GL_RGB16_SNORM";
	spInternalF[GL_RGBA16_SNORM] = "GL_RGBA16_SNORM";
	spInternalF[GL_RGB10_A2UI] = "GL_RGB10_A2UI";

	return true;
}




// gets the atomic data type
Types 
getType(GLenum type) {

	switch (type) {
		case GL_DOUBLE:
		case GL_DOUBLE_MAT2:
		case GL_DOUBLE_MAT2x3:
		case GL_DOUBLE_MAT2x4:
		case GL_DOUBLE_MAT3:
		case GL_DOUBLE_MAT3x2:
		case GL_DOUBLE_MAT3x4:
		case GL_DOUBLE_MAT4:
		case GL_DOUBLE_MAT4x2:
		case GL_DOUBLE_MAT4x3:
		case GL_DOUBLE_VEC2:
		case GL_DOUBLE_VEC3:
		case GL_DOUBLE_VEC4:
			return DOUBLE;
		case GL_FLOAT:
		case GL_FLOAT_MAT2:
		case GL_FLOAT_MAT2x3:
		case GL_FLOAT_MAT2x4:
		case GL_FLOAT_MAT3:
		case GL_FLOAT_MAT3x2:
		case GL_FLOAT_MAT3x4:
		case GL_FLOAT_MAT4:
		case GL_FLOAT_MAT4x2:
		case GL_FLOAT_MAT4x3:
		case GL_FLOAT_VEC2:
		case GL_FLOAT_VEC3:
		case GL_FLOAT_VEC4:
			return FLOAT;
		case GL_BOOL:
		case GL_BOOL_VEC2:
		case GL_BOOL_VEC3:
		case GL_BOOL_VEC4:
		case GL_INT:
		case GL_INT_SAMPLER_1D:
		case GL_INT_SAMPLER_1D_ARRAY:
		case GL_INT_SAMPLER_2D:
		case GL_INT_SAMPLER_2D_ARRAY:
		case GL_INT_SAMPLER_2D_MULTISAMPLE:
		case GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
		case GL_INT_SAMPLER_2D_RECT:
		case GL_INT_SAMPLER_3D:
		case GL_INT_SAMPLER_BUFFER:
		case GL_INT_SAMPLER_CUBE:
		case GL_INT_VEC2:
		case GL_INT_VEC3:
		case GL_INT_VEC4:
		case GL_SAMPLER_1D:
		case GL_SAMPLER_1D_ARRAY:
		case GL_SAMPLER_1D_ARRAY_SHADOW:
		case GL_SAMPLER_1D_SHADOW:
		case GL_SAMPLER_2D:
		case GL_SAMPLER_2D_ARRAY:
		case GL_SAMPLER_2D_ARRAY_SHADOW:
		case GL_SAMPLER_2D_MULTISAMPLE:
		case GL_SAMPLER_2D_MULTISAMPLE_ARRAY:
		case GL_SAMPLER_2D_RECT:
		case GL_SAMPLER_2D_RECT_SHADOW:
		case GL_SAMPLER_2D_SHADOW:
		case GL_SAMPLER_3D:
		case GL_SAMPLER_BUFFER:
		case GL_SAMPLER_CUBE:
		case GL_SAMPLER_CUBE_SHADOW:
			return INT;
		case GL_UNSIGNED_INT:
		case GL_UNSIGNED_INT_SAMPLER_1D:
		case GL_UNSIGNED_INT_SAMPLER_1D_ARRAY:
		case GL_UNSIGNED_INT_SAMPLER_2D:
		case GL_UNSIGNED_INT_SAMPLER_2D_ARRAY:
		case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE:
		case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY:
		case GL_UNSIGNED_INT_SAMPLER_2D_RECT:
		case GL_UNSIGNED_INT_SAMPLER_3D:
		case GL_UNSIGNED_INT_SAMPLER_BUFFER:
		case GL_UNSIGNED_INT_SAMPLER_CUBE:
		case GL_UNSIGNED_INT_VEC2:
		case GL_UNSIGNED_INT_VEC3:
		case GL_UNSIGNED_INT_VEC4:
			return UNSIGNED_INT;

		default:
			return DONT_KNOW;

	}
}


// gets the number of rows for a GLSL type
int 
getRows(GLenum type) {

	switch(type) {
		case GL_DOUBLE_MAT2:
		case GL_DOUBLE_MAT2x3:
		case GL_DOUBLE_MAT2x4:
		case GL_FLOAT_MAT2:
		case GL_FLOAT_MAT2x3:
		case GL_FLOAT_MAT2x4:
			return 2;

		case GL_DOUBLE_MAT3:
		case GL_DOUBLE_MAT3x2:
		case GL_DOUBLE_MAT3x4:
		case GL_FLOAT_MAT3:
		case GL_FLOAT_MAT3x2:
		case GL_FLOAT_MAT3x4:
			return 3;

		case GL_DOUBLE_MAT4:
		case GL_DOUBLE_MAT4x2:
		case GL_DOUBLE_MAT4x3:
		case GL_FLOAT_MAT4:
		case GL_FLOAT_MAT4x2:
		case GL_FLOAT_MAT4x3:
			return 4;

		default: return 1;
	}
}


// gets the number of columns for a GLSL type
int 
getColumns(GLenum type) {

	switch(type) {
		case GL_DOUBLE_MAT2:
		case GL_FLOAT_MAT2:
		case GL_DOUBLE_MAT3x2:
		case GL_FLOAT_MAT3x2:
		case GL_DOUBLE_MAT4x2:
		case GL_FLOAT_MAT4x2:
		case GL_UNSIGNED_INT_VEC2:
		case GL_INT_VEC2:
		case GL_BOOL_VEC2:
		case GL_FLOAT_VEC2:
		case GL_DOUBLE_VEC2:
			return 2;

		case GL_DOUBLE_MAT2x3:
		case GL_FLOAT_MAT2x3:
		case GL_DOUBLE_MAT3:
		case GL_FLOAT_MAT3:
		case GL_DOUBLE_MAT4x3:
		case GL_FLOAT_MAT4x3:
		case GL_UNSIGNED_INT_VEC3:
		case GL_INT_VEC3:
		case GL_BOOL_VEC3:
		case GL_FLOAT_VEC3:
		case GL_DOUBLE_VEC3:
			return 3;

		case GL_DOUBLE_MAT2x4:
		case GL_FLOAT_MAT2x4:
		case GL_DOUBLE_MAT3x4:
		case GL_FLOAT_MAT3x4:
		case GL_DOUBLE_MAT4:
		case GL_FLOAT_MAT4:
		case GL_UNSIGNED_INT_VEC4:
		case GL_INT_VEC4:
		case GL_BOOL_VEC4:
		case GL_FLOAT_VEC4:
		case GL_DOUBLE_VEC4:
			return 4;

		default: return 1;
	}
}


// aux function to get the size in bytes of a uniform
// it takes the strides into account
int 
getUniformByteSize(int uniSize, 
				int uniType, 
				int uniArrayStride, 
				int uniMatStride) {

	int auxSize;
	if (uniArrayStride > 0)
		auxSize = uniArrayStride * uniSize;

	else if (uniMatStride > 0) {

		switch(uniType) {
			case GL_FLOAT_MAT2:
			case GL_FLOAT_MAT2x3:
			case GL_FLOAT_MAT2x4:
			case GL_DOUBLE_MAT2:
			case GL_DOUBLE_MAT2x3:
			case GL_DOUBLE_MAT2x4:
				auxSize = 2 * uniMatStride;
				break;
			case GL_FLOAT_MAT3:
			case GL_FLOAT_MAT3x2:
			case GL_FLOAT_MAT3x4:
			case GL_DOUBLE_MAT3:
			case GL_DOUBLE_MAT3x2:
			case GL_DOUBLE_MAT3x4:
				auxSize = 3 * uniMatStride;
				break;
			case GL_FLOAT_MAT4:
			case GL_FLOAT_MAT4x2:
			case GL_FLOAT_MAT4x3:
			case GL_DOUBLE_MAT4:
			case GL_DOUBLE_MAT4x2:
			case GL_DOUBLE_MAT4x3:
				auxSize = 4 * uniMatStride;
				break;
		}
	}
	else
		auxSize = spGLSLTypeSize[uniType];

	return auxSize;
}




//Additional Custom Functions
void getUniformNames(unsigned int program, std::vector<std::string> &namelist){
	int activeUnif, actualLen, index;
	char name[256];

	namelist.clear();

	// is it a program ?
	if (glIsProgram(program)) {

		// Get uniforms info (not in named blocks)
		glGetProgramiv(program, GL_ACTIVE_UNIFORMS, &activeUnif);

		for (unsigned int i = 0; i < (unsigned int)activeUnif; ++i) {
			glGetActiveUniformsiv(program, 1, &i, GL_UNIFORM_BLOCK_INDEX, &index);
			if (index == -1) {
				glGetActiveUniformName(program, i, 256, &actualLen, name);	
				namelist.push_back(name);
			}
		}
	}
}

//Additional Custom Functions
std::string getDatatypeString(int datatype){
	return spDataF[datatype];
}

void getBlockNames(unsigned int program, std::vector<std::string> &namelist){
	int count;
	char name[256];

	namelist.clear();

	// is it a program ?
	if (glIsProgram(program)) {
		
	glGetProgramiv(program, GL_ACTIVE_UNIFORM_BLOCKS, &count);

		for (int i = 0; i < count; ++i) {
			// Get buffers name
			glGetActiveUniformBlockName(program, i, 256, NULL, name);
			namelist.push_back(name);
		}
	}
}

//untested
void getBlockData(unsigned int program, std::string blockName, int &datasize, int &blockbindingpoint, int &bufferbindingpoint, std::vector<std::string> &uniformnamelist){
	int index, activeUnif, actualLen;
	char name[256];

	uniformnamelist.clear();

	// is it a program ?
	if (glIsProgram(program)) {
		
		index = glGetUniformBlockIndex(program, blockName.c_str());
		glGetActiveUniformBlockiv(program, index, GL_UNIFORM_BLOCK_DATA_SIZE, &datasize);
		glGetActiveUniformBlockiv(program, index,  GL_UNIFORM_BLOCK_BINDING, &blockbindingpoint);
		glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, index, &bufferbindingpoint);
		glGetProgramiv(program, GL_ACTIVE_UNIFORMS, &activeUnif);
		
		int *indices;
		indices = (int *)malloc(sizeof(unsigned int) * activeUnif);
		glGetActiveUniformBlockiv(program, index, GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES, indices);
		
		for (int i = 0; i < activeUnif; ++i) {
			// Get buffers name
			glGetActiveUniformName(program, indices[i], 256, &actualLen, name);
			uniformnamelist.push_back(name);
		}

		free(indices);
	}
}


void getUniformData(unsigned int program, std::string uniformName, std::string &uniformType, int &uniformSize, int &uniformArrayStride, std::vector<std::string> &values){

	// is it a program ?
	if (!glIsProgram(program)) {
		return;
	}
	
	int uniType, 
		uniSize, uniArrayStride;
	GLenum type;
	GLsizei l;
	GLint s;
	char c[50];
	unsigned int loc = glGetUniformLocation((int)program, uniformName.c_str());
	glGetActiveUniform(program, loc, 0, &l, &s, &type, c);
	
	if (loc != -1) {
		int auxSize;
		int rows = getRows(type), columns = getColumns(type);

		glGetActiveUniformsiv(program, 1, &loc, GL_UNIFORM_TYPE, &uniType);
		uniformType = spGLSLType[uniType].c_str();

		glGetActiveUniformsiv(program, 1, &loc, GL_UNIFORM_SIZE, &uniSize);
		glGetActiveUniformsiv(program, 1, &loc, GL_UNIFORM_ARRAY_STRIDE, &uniArrayStride);
		
		if (uniArrayStride > 0)
			auxSize = uniArrayStride * uniSize;
		else
			auxSize = spGLSLTypeSize[uniType];

		uniformSize = auxSize;
		uniformArrayStride = uniArrayStride;

		if (getType(type) == FLOAT) {
			float f[16];
			glGetUniformfv(program, loc, f);
			getUniformValuef(f,rows,columns,values);
		}
		else if (getType(type) == INT) {
			int f[16];
			glGetUniformiv(program, loc, f);
			getUniformValuei(f,rows,columns,values);
		}
		else if (getType(type) == UNSIGNED_INT) {
			unsigned int f[16];
			glGetUniformuiv(program, loc, f);
			getUniformValueui(f,rows,columns,values);
		}
		else if (getType(type) == DOUBLE) {
			double f[16];
			glGetUniformdv(program, loc, f);
			getUniformValued(f,rows,columns,values);
		}
	}
}

void getUniformData(unsigned int program, std::string blockName, std::string uniformName, std::string &uniformType, int &uniformSize, int &uniformArrayStride, std::vector<std::string> &values){

	// is it a program ?
	if (!glIsProgram(program)) {
		return;
	}
	
	int bindex = glGetUniformBlockIndex(program, blockName.c_str());

	
	if (bindex == GL_INVALID_INDEX) {
		return;
	}
	
	int bindIndex,bufferIndex;
	glGetActiveUniformBlockiv(program, bindex,  GL_UNIFORM_BLOCK_BINDING, &bindIndex);
	glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, bindIndex, &bufferIndex);
	int prevBuffer;
	glGetIntegerv(spBoundBuffer[GL_UNIFORM_BUFFER], &prevBuffer);

	int uniType, 
		uniSize,uniArrayStride;
	GLint type;
	const char *c = uniformName.c_str();
	unsigned int loc;
	glGetUniformIndices(program, 1, &c, &loc);

	glGetActiveUniformsiv(program, 1, &loc, GL_UNIFORM_TYPE, &type);
	
	if (loc != -1) {
		int auxSize;
		int rows = getRows(type), columns = getColumns(type);

		glGetActiveUniformsiv(program, 1, &loc, GL_UNIFORM_TYPE, &uniType);
		uniformType = spGLSLType[uniType].c_str();

		glGetActiveUniformsiv(program, 1, &loc, GL_UNIFORM_SIZE, &uniSize);
		glGetActiveUniformsiv(program, 1, &loc, GL_UNIFORM_ARRAY_STRIDE, &uniArrayStride);
		
		if (uniArrayStride > 0)
			auxSize = uniArrayStride * uniSize;
		else
			auxSize = spGLSLTypeSize[uniType];

		uniformSize = auxSize;
		uniformArrayStride = uniArrayStride;

		if (getType(type) == FLOAT) {
			float f[16];
			glGetUniformfv(program, loc, f);
			getUniformValuef(f,rows,columns,values);
		}
		else if (getType(type) == INT) {
			int f[16];
			glGetUniformiv(program, loc, f);
			getUniformValuei(f,rows,columns,values);
		}
		else if (getType(type) == UNSIGNED_INT) {
			unsigned int f[16];
			glGetUniformuiv(program, loc, f);
			getUniformValueui(f,rows,columns,values);
		}
		else if (getType(type) == DOUBLE) {
			double f[16];
			glGetUniformdv(program, loc, f);
			getUniformValued(f,rows,columns,values);
		}
	}

	glBindBuffer(GL_UNIFORM_BUFFER, prevBuffer);
}



// aux function to display float based uniforms
void 
getUniformValuef(float *f, int rows, int columns, std::vector<std::string> &values) {

	for (int i = 0; i < rows; ++i) {
		if (columns == 1)
			values.push_back(std::to_string(f[i*columns]));
		else if (columns == 2)
			values.push_back(std::to_string(f[i*columns]) + " " + std::to_string(f[i*columns+1]));
		else if (columns == 3)
			values.push_back(std::to_string(f[i*columns]) + " " + std::to_string(f[i*columns+1]) + " " + std::to_string(f[i*columns+2]));
		else if (columns == 4)
			values.push_back(std::to_string(f[i*columns]) + " " + std::to_string(f[i*columns+1]) + " " + std::to_string(f[i*columns+2]) + " " + std::to_string(f[i*columns+3]));
	}
}


// aux function to display int based uniforms
void 
getUniformValuei(int *f, int rows, int columns, std::vector<std::string> &values) {

	for (int i = 0; i < rows; ++i) {
		if (columns == 1)
			values.push_back(std::to_string(f[i*columns]));
		else if (columns == 2)
			values.push_back(std::to_string(f[i*columns]) + " " + std::to_string(f[i*columns+1]));
		else if (columns == 3)
			values.push_back(std::to_string(f[i*columns]) + " " + std::to_string(f[i*columns+1]) + " " + std::to_string(f[i*columns+2]));
		else if (columns == 4)
			values.push_back(std::to_string(f[i*columns]) + " " + std::to_string(f[i*columns+1]) + " " + std::to_string(f[i*columns+2]) + " " + std::to_string(f[i*columns+3]));
	}
}


// aux function to display unsigned int based uniforms
void 
getUniformValueui(unsigned int *f, int rows, int columns, std::vector<std::string> &values) {

	for (int i = 0; i < rows; ++i) {
		if (columns == 1)
			values.push_back(std::to_string(f[i*columns]));
		else if (columns == 2)
			values.push_back(std::to_string(f[i*columns]) + " " + std::to_string(f[i*columns+1]));
		else if (columns == 3)
			values.push_back(std::to_string(f[i*columns]) + " " + std::to_string(f[i*columns+1]) + " " + std::to_string(f[i*columns+2]));
		else if (columns == 4)
			values.push_back(std::to_string(f[i*columns]) + " " + std::to_string(f[i*columns+1]) + " " + std::to_string(f[i*columns+2]) + " " + std::to_string(f[i*columns+3]));
	}
}


// aux function to display double based uniforms
void 
getUniformValued(double *f, int rows, int columns, std::vector<std::string> &values) {

	for (int i = 0; i < rows; ++i) {
		if (columns == 1)
			values.push_back(std::to_string(f[i*columns]));
		else if (columns == 2)
			values.push_back(std::to_string(f[i*columns]) + " " + std::to_string(f[i*columns+1]));
		else if (columns == 3)
			values.push_back(std::to_string(f[i*columns]) + " " + std::to_string(f[i*columns+1]) + " " + std::to_string(f[i*columns+2]));
		else if (columns == 4)
			values.push_back(std::to_string(f[i*columns]) + " " + std::to_string(f[i*columns+1]) + " " + std::to_string(f[i*columns+2]) + " " + std::to_string(f[i*columns+3]));
	}

}

void 
getProgramInfoData(unsigned int program, std::vector<std::pair<std::string,char>> &shadersInfo, std::vector<std::string> &stdInfo,  std::vector<std::string> &geomInfo,  std::vector<std::string> &tessInfo) {


	// check if name is really a program
	if (!glIsProgram(program)) {
		return;
	}

	unsigned int shaders[5];
	int count, info, linked;
	bool geom= false, tess = false;

	shadersInfo.clear();
	stdInfo.clear();
	geomInfo.clear();
	tessInfo.clear();

	// Get the shader's name
	glGetProgramiv(program, GL_ATTACHED_SHADERS,&count);
	glGetAttachedShaders(program, count, NULL, shaders);
	for (int i = 0;  i < count; ++i) {
		glGetShaderiv(shaders[i], GL_SHADER_TYPE, &info);
		if (info == GL_GEOMETRY_SHADER){
			geom = true;
			shadersInfo.push_back(std::pair<std::string,char>(spShaderType[info] + ": " + std::to_string(shaders[i]),'g'));
		}
		if (info == GL_TESS_EVALUATION_SHADER || info == GL_TESS_CONTROL_SHADER){
			tess = true;
			shadersInfo.push_back(std::pair<std::string,char>(spShaderType[info] + ": " + std::to_string(shaders[i]),'t'));
		}
		else {
			shadersInfo.push_back(std::pair<std::string,char>(spShaderType[info] + ": " + std::to_string(shaders[i]),0));
		}
	}

	// Get program info
	glGetProgramiv(program, GL_PROGRAM_SEPARABLE, &info);
	stdInfo.push_back("Program Separable: " + std::to_string(info));

	glGetProgramiv(program, GL_PROGRAM_BINARY_RETRIEVABLE_HINT, &info);
	stdInfo.push_back("Program Binary Retrievable Hint: " + std::to_string(info));

	glGetProgramiv(program, GL_LINK_STATUS, &linked);
	stdInfo.push_back("Link Status: " + std::to_string(linked));

	glGetProgramiv(program, GL_VALIDATE_STATUS, &info);
	stdInfo.push_back("Validate_Status: " + std::to_string(info));

	glGetProgramiv(program, GL_DELETE_STATUS, &info);
	stdInfo.push_back("Delete_Status: " + std::to_string(info));

	glGetProgramiv(program, GL_ACTIVE_ATTRIBUTES, &info);
	stdInfo.push_back("Active_Attributes: " + std::to_string(info));

	glGetProgramiv(program, GL_ACTIVE_UNIFORMS, &info);
	stdInfo.push_back("Active_Uniforms: " + std::to_string(info));

	glGetProgramiv(program, GL_ACTIVE_UNIFORM_BLOCKS, &info);
	stdInfo.push_back("Active_Uniform_Blocks: " + std::to_string(info));

#if (NAU_OPENGL_VERSION >= 420)
	glGetProgramiv(program, GL_ACTIVE_ATOMIC_COUNTER_BUFFERS, &info);
	stdInfo.push_back("Active_Atomic Counters: " + std::to_string(info));
#endif
	// check if trans feed is active
	glGetProgramiv(program, GL_TRANSFORM_FEEDBACK_BUFFER_MODE, &info);
	stdInfo.push_back("Transform Feedback Buffer Mode: " + spTransFeedBufferMode[info]);

	glGetProgramiv(program, GL_TRANSFORM_FEEDBACK_VARYINGS, &info);
	stdInfo.push_back("Transform Feedback Varying: " + std::to_string(info));

	// Geometry shader info, if present
	if (geom && linked) {
		glGetProgramiv(program, GL_GEOMETRY_VERTICES_OUT, &info);
	    geomInfo.push_back("Geometry Vertices Out: " + std::to_string(info));

		glGetProgramiv(program, GL_GEOMETRY_INPUT_TYPE, &info);
	    geomInfo.push_back("Geometry Input Type: " + spGLSLPrimitives[info]);

		glGetProgramiv(program, GL_GEOMETRY_OUTPUT_TYPE, &info);
	    geomInfo.push_back("Geometry Output Type: " + spGLSLPrimitives[info]);

		glGetProgramiv(program, GL_GEOMETRY_SHADER_INVOCATIONS, &info);
	    geomInfo.push_back("Geometry Shader Invocations: " + std::to_string(info));
	}
	// tessellation shaders info, if present
	if (tess && linked) {
		glGetProgramiv(program, GL_TESS_CONTROL_OUTPUT_VERTICES, &info);
	    tessInfo.push_back("Tess Control Output Vertices: " + std::to_string(info));

		glGetProgramiv(program, GL_TESS_GEN_MODE, &info);
	    tessInfo.push_back("Tess Gen Mode: " + spGLSLPrimitives[info]);

		glGetProgramiv(program, GL_TESS_GEN_SPACING, &info);
	    tessInfo.push_back("Tess Spacing: " + spTessGenSpacing[info]);

		glGetProgramiv(program, GL_TESS_GEN_VERTEX_ORDER, &info);
	    tessInfo.push_back("Tess Vertex Order: " + spVertexOrder[info]);

		glGetProgramiv(program, GL_TESS_GEN_POINT_MODE, &info);
	    tessInfo.push_back("Tess Gen Point Mode: " + std::to_string(info));
	}
}

void
getAttributesData(unsigned int program, std::vector<std::pair<std::string, std::pair<int,std::string>>> &attributeList) {

	int activeAttr, size, loc;
	GLsizei length;
	GLenum type;
	char name[256];


	// check if it is a program
	if (!glIsProgram(program)) {
		return;
	}

	attributeList.clear();

	// how many attribs?
	glGetProgramiv(program, GL_ACTIVE_ATTRIBUTES, &activeAttr);
	// get location and type for each attrib
	for (unsigned int i = 0; i < (unsigned int)activeAttr; ++i) {

		glGetActiveAttrib(program,	i, 256, &length, &size, &type, name);
		loc = glGetAttribLocation(program, name);
		attributeList.push_back(std::pair<std::string, std::pair<int,std::string>>(name, std::pair<int,std::string>(loc, spGLSLType[type])));
	}
}


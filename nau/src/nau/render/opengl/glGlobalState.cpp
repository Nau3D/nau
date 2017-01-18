#include "nau/render/opengl/glGlobalState.h" 

#include <glbinding/gl/gl.h>
using namespace gl;


GLGlobalState::GLGlobalState() {

}


std::string 
GLGlobalState::getStateValue(unsigned int enumValue, FunctionType type) {

	GLenum ev = (GLenum)enumValue;
	switch (type){
		case BOOLEANV:{
			GLboolean value;
			glGetBooleanv(ev, &value);
			return value == GL_TRUE ? "GL_TRUE" : "GL_FALSE";
		}
		case DOUBLEV:{
			GLdouble value;
			glGetDoublev(ev, &value);
			return std::to_string(value);
		}
		case FLOATV:{
			GLfloat value;
			glGetFloatv(ev, &value);
			return std::to_string(value);
		}
		case INTEGERV:{
			GLint value;
			glGetIntegerv(ev, &value);
			return std::to_string(value);
		}
		case INTEGER64V:{
			GLint64 value;
			glGetInteger64v(ev, &value);
			return std::to_string(value);
		}
	}
	return "";
}


std::string 
GLGlobalState::getStateValue(unsigned int enumValue, FunctionType type, unsigned int length) {

	GLenum ev = (GLenum)enumValue;
	std::string finalString = "[";
	switch (type){
		case BOOLEANV:{
			GLboolean value[256];
			glGetBooleanv(ev, value);
			for (unsigned int i = 0; i < length; i++){
				if (i != 0){
					finalString += ", ";
				}
				if (value[i] == GL_TRUE){
					finalString += "GL_TRUE";
				}
				else{
					finalString += "GL_FALSE";
				}
			}
			break;
		}
		case DOUBLEV:{
			GLdouble value[256];
			glGetDoublev(ev, value);
			for (unsigned int i = 0; i < length; i++){
				if (i != 0){
					finalString += ", ";
				}
				finalString += std::to_string(value[i]);
			}
			break;
		}
		case FLOATV:{
			GLfloat value[256];
			glGetFloatv(ev, value);
			for (unsigned int i = 0; i < length; i++){
				if (i != 0){
					finalString += ", ";
				}
				finalString += std::to_string(value[i]);
			}
			break;
		}
		case INTEGERV:{
			GLint value[256];
			glGetIntegerv(ev, value);
			for (unsigned int i = 0; i < length; i++){
				if (i != 0){
					finalString += ", ";
				}
				finalString += std::to_string(value[i]);
			}
			break;
		}
		case INTEGERI_V:{
			GLint value[256];
			for (unsigned int i = 0; i < length; i++){
				glGetIntegeri_v(ev, i, &value[i]);
				if (i != 0){
					finalString += ", ";
				}
				finalString += std::to_string(value[i]);
			}
			break;
		}
		case INTEGER64V: {
			GLint64 value[256];
			glGetInteger64v(ev, value);
			for (unsigned int i = 0; i < length; i++){
				if (i != 0){
					finalString += ", ";
				}
				finalString += std::to_string(value[i]);
			}
			break;
		}
	}
	finalString += "]";
	return finalString;
}


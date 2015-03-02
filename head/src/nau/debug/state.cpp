#ifndef NAU_GL_STATE_READER_CPP
#define NAU_GL_STATE_READER_CPP

#include "state.h"
#include <GL/glew.h>
#include "nau.h"
#include "nau/system/fileutil.h"

using namespace nau::system;

std::map<std::string, State::StateVariable> State::stateVariablesMap;
//std::map<std::string, State::StateFunction> State::functionMap;
std::map<std::string, State::FunctionType> State::functionMap;
std::string State::s_Path = "";
std::string State::s_File = "";

//enum FunctionType{
//	GL_GLGETBOOLEANV,
//	GL_GLGETDOUBLEV,
//	GL_GLGETFLOATV,
//	GL_GLGETINTEGERV,
//	GL_GLGETINTEGER64V
//	GL_ENUM_I_BOOLEANPTR,
//	GL_ENUM_I_DOUBLEPTR,
//	GL_ENUM_I_FLOATPTR,
//	GL_ENUM_I_INTEGERPTR,
//	GL_ENUM_I_INTEGER64PTR
//};

void State::init(){
	functionMap.clear();
	functionMap["glGetBooleanv"] = GL_GLGETBOOLEANV;
	functionMap["glGetDoublev"] = GL_GLGETDOUBLEV;
	functionMap["glGetFloatv"] = GL_GLGETFLOATV;
	functionMap["glGetIntegerv"] = GL_GLGETINTEGERV;
	functionMap["glGetInteger64v"] = GL_GLGETINTEGER64V;
}
State::StateVariable::StateVariable() :
functionName(""),
enumValue(0),
length(0)
{
}
State::StateVariable::StateVariable(std::string pfunctionName, int penumValue, int plength) :
functionName(pfunctionName),
enumValue(penumValue)
{
	length = plength;
	if (length <= 0){
		length = 1;
	}
}

//State::StateFunction::StateFunction()
//{
//}
//State::StateFunction::StateFunction(FunctionType ptype){
//	type = ptype;
//}


std::vector<std::string> State::getStateEnumNames(){
	std::vector<std::string> enumNames;
	std::map<std::string, State::StateVariable>::iterator iter;

	for (iter = stateVariablesMap.begin(); iter != stateVariablesMap.end(); iter++){
		enumNames.push_back(iter->first);
	}

	return enumNames;
}

std::string State::getState(std::string enumName){
	if (stateVariablesMap.find(enumName) != stateVariablesMap.end()){
		StateVariable enumVariable = stateVariablesMap[enumName];
		if (functionMap.find(enumVariable.functionName) != functionMap.end()){
			FunctionType type = functionMap[enumVariable.functionName];
			if (enumVariable.length == 1){ //single value
				return getStateValue(enumVariable.enumValue, type);
			}
			else{
				return getStateValue(enumVariable.enumValue, type, enumVariable.length);
			}
		}
	}
	return "";
}

std::string State::getStateValue(int enumValue, FunctionType type){
	switch (type){
	case GL_GLGETBOOLEANV:{
		GLboolean value;
		glGetBooleanv(enumValue, &value);
		if (value == GL_TRUE){
			return "GL_TRUE";
		}
		else{
			return "GL_FALSE";
		}
	}
	case GL_GLGETDOUBLEV:{
		GLdouble value;
		glGetDoublev(enumValue, &value);
		return std::to_string(value);
	}
	case GL_GLGETFLOATV:{
		GLfloat value;
		glGetFloatv(enumValue, &value);
		return std::to_string(value);
	}
	case GL_GLGETINTEGERV:{
		GLint value;
		glGetIntegerv(enumValue, &value);
		return std::to_string(value);
	}
	case GL_GLGETINTEGER64V:{
		GLint64 value;
		glGetInteger64v(enumValue, &value);
		return std::to_string(value);
	}
	}
	return "";
}

std::string State::getStateValue(int enumValue, FunctionType type, int length){
	std::string finalString = "[";
	switch (type){
	case GL_GLGETBOOLEANV:{
		GLboolean value[256];
		glGetBooleanv(enumValue, value);
		for (int i = 0; i < length; i++){
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
	case GL_GLGETDOUBLEV:{
		GLdouble value[256];
		glGetDoublev(enumValue, value);
		for (int i = 0; i < length; i++){
			if (i != 0){
				finalString += ", ";
			}
			finalString += std::to_string(value[i]);
		}
		break;
	}
	case GL_GLGETFLOATV:{
		GLfloat value[256];
		glGetFloatv(enumValue, value);
		for (int i = 0; i < length; i++){
			if (i != 0){
				finalString += ", ";
			}
			finalString += std::to_string(value[i]);
		}
		break;
	}
	case GL_GLGETINTEGERV:{
		GLint value[256];
		glGetIntegerv(enumValue, value);
		for (int i = 0; i < length; i++){
			if (i != 0){
				finalString += ", ";
			}
			finalString += std::to_string(value[i]);
		}
		break;
	}
	case GL_GLGETINTEGER64V:{
		GLint64 value[256];
		glGetInteger64v(enumValue, value);
		for (int i = 0; i < length; i++){
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




void State::loadStateXMLFile(std::string file){
	State::s_Path = FileUtil::GetPath(file);
	State::s_File = file;

	TiXmlDocument doc(file.c_str());
	bool loadOkay = doc.LoadFile();

	if (!loadOkay) {

		NAU_THROW("Parsing Error -%s- Line(%d) Column(%d) in file: %s", doc.ErrorDesc(), doc.ErrorRow(), doc.ErrorCol(), file.c_str());
	}
	TiXmlHandle hDoc(&doc);
	TiXmlHandle hRoot(0);
	TiXmlElement *pElem;

	//Methods
	pElem = hDoc.FirstChildElement().Element();
	if (0 == pElem) {
		NAU_THROW("Parsing Error in file: %s", file.c_str());
	}
	hRoot = TiXmlHandle(pElem);


	try {
		//Load here
		loadMethods(hRoot);
	}
	catch (std::string &s) {
		throw(s);
	}
}

void State::loadMethods(TiXmlHandle &handle)
{
	TiXmlElement *pElem;

	string name;
	string dllpath;
	string data = "";

	pElem = handle.FirstChild().Element();
	for (; 0 != pElem; pElem = pElem->NextSiblingElement()) {
		TiXmlHandle methodHandle(pElem);
		std::string functionName = loadFunction(methodHandle);
		loadEnums(methodHandle, functionName);
		//pElem->QueryStringAttribute("name", &name);
		//pElem->QueryStringAttribute("dll", &dllpath);
		if (pElem->GetText()){
			data = pElem->GetText();
		}

	}
}

std::string State::loadFunction(TiXmlHandle &hRoot)
{
	std::string name = "";
	TiXmlElement *pElem;
	TiXmlHandle handle(hRoot.FirstChild("function").Element());
	if (handle.Element()){
		pElem = handle.Element();
		pElem->QueryStringAttribute("name", &name);
	}
	return name; 
}

void State::loadEnums(TiXmlHandle &hRoot, std::string functionName)
{
	std::string name = "";
	std::string valueString = "";
	TiXmlElement *pElem;
	TiXmlHandle handle(hRoot.FirstChild("enums").Element());
	pElem = handle.FirstChild().Element();
	for (; 0 != pElem; pElem = pElem->NextSiblingElement()) {
		int value, length = 1;
		pElem->QueryStringAttribute("name", &name);
		pElem->QueryStringAttribute("value", &valueString);
		if (pElem->QueryIntAttribute("length", &length) != TIXML_SUCCESS){
			length = 1;
		}
		value = (int)strtol(valueString.c_str(), NULL, 0);
		stateVariablesMap[name] = StateVariable(functionName, value, length);
	}
}
#endif
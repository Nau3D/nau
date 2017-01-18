#include "nau/render/iGlobalState.h"

#include "nau/config.h"
#include "nau/render/opengl/glGlobalState.h"

#include <assert.h>

using namespace nau::render;


std::map<std::string, IGlobalState::FunctionType> IGlobalState::FunctionMap = {
	{"Booleanv", BOOLEANV},
	{"Doublev", DOUBLEV},
	{"Floatv", FLOATV},
	{"Integerv", INTEGERV},
	{"Integeri_v", INTEGERI_V},
	{"Integer64v", INTEGER64V}
};


IGlobalState::StateVariable::StateVariable() :
	functionName(""),
	enumValue(0),
	length(0) {

}


IGlobalState::StateVariable::StateVariable(std::string pfunctionName, unsigned int penumValue, 
					unsigned int plength) :functionName(pfunctionName), enumValue(penumValue)
{
	length = plength;
	if (length <= 0){
		length = 1;
	}
}


IGlobalState *
IGlobalState::Create() {

#if NAU_OPENGL == 1
	return new GLGlobalState();
#endif
	return NULL;

}

IGlobalState::IGlobalState() {

}


void
IGlobalState::addStateEnum(std::string &name, std::string &functionName,
				unsigned int penumValue, unsigned int pLength) {

	stateVariablesMap[name] = StateVariable(functionName, penumValue, pLength);
}

void
IGlobalState::getStateEnumNames(std::vector<std::string> *enumNames) {

	for (auto anEnum:stateVariablesMap){
		enumNames->push_back(anEnum.first);
	}
}


std::string 
IGlobalState::getState(std::string enumName) {

	if (stateVariablesMap.find(enumName) != stateVariablesMap.end()) {

		StateVariable enumVariable = stateVariablesMap[enumName];
		assert(FunctionMap.find(enumVariable.functionName) != FunctionMap.end());

		FunctionType type = FunctionMap[enumVariable.functionName];
		if (enumVariable.length == 1) { //single value
			return getStateValue(enumVariable.enumValue, type);
		}
		else {
			return getStateValue(enumVariable.enumValue, type, enumVariable.length);
		}
	}
	else
		return "";
}

#ifndef NAU_GL_STATE_READER_H
#define NAU_GL_STATE_READER_H
#include <tinyxml.h>
#include <string>
#include <vector>
#include <map>

class State{

	enum FunctionType{
		GL_GLGETBOOLEANV,
		GL_GLGETDOUBLEV,
		GL_GLGETFLOATV,
		GL_GLGETINTEGERV,
		GL_GLGETINTEGER64V
		//GL_ENUM_I_BOOLEANPTR,
		//GL_ENUM_I_DOUBLEPTR,
		//GL_ENUM_I_FLOATPTR,
		//GL_ENUM_I_INTEGERPTR,
		//GL_ENUM_I_INTEGER64PTR
	};

	struct StateVariable{
		StateVariable();
		StateVariable(std::string pfunctionName, int penumValue, int plength = 1);
		std::string functionName;
		int length;
		int enumValue;
	};

	//struct StateFunction{
	//	StateFunction();
	//	StateFunction(FunctionType ptype);
	//	FunctionType type;
	//};

	static std::map<std::string, StateVariable> stateVariablesMap;
	//static std::map<std::string, StateFunction> functionMap;
	static std::map<std::string, FunctionType> functionMap;

	static void loadMethods(TiXmlHandle &handle);
	static std::string loadFunction(TiXmlHandle &hRoot);
	static void loadEnums(TiXmlHandle &hRoot, std::string functionName);


	static std::string getStateValue(int enumValue, FunctionType type);
	static std::string getStateValue(int enumValue, FunctionType type, int length);
public:
	static void init();
	static void loadStateXMLFile(std::string file);
	static std::vector<std::string> getStateEnumNames();
	static std::string getState(std::string enumName);
	//static std::string getState(std::string enumName,int index);

	static std::string s_Path;
	static std::string s_File;
};

#endif
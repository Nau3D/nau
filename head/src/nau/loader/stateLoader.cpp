#include <nau/loader/stateLoader.h>

#include <nau/render/iGlobalState.h>

#include <nau/errors.h>

using namespace nau::loader;

void 
StateLoader::LoadStateXMLFile(std::string file, IGlobalState *state) {

	TiXmlDocument doc(file.c_str());
	bool loadOkay = doc.LoadFile();

	if (!loadOkay) {
		NAU_THROW("Parsing Error -%s- Line(%d) Column(%d) in file: %s", doc.ErrorDesc(), doc.ErrorRow(), doc.ErrorCol(), file.c_str());
	}

	TiXmlHandle hDoc(&doc);
	TiXmlHandle hRoot(0);
	TiXmlElement *pElem;

	pElem = hDoc.FirstChildElement().Element();
	if (0 == pElem) {
		NAU_THROW("Parsing Error in file: %s", file.c_str());
	}
	hRoot = TiXmlHandle(pElem);

	try {
		LoadFunctionEnums(hRoot, state);
	}
	catch (std::string &s) {
		throw(s);
	}
}


void 
StateLoader::LoadFunctionEnums(TiXmlHandle &handle, IGlobalState *state) {

	TiXmlElement *pElem;

	std::string name;
	std::string data = "";
	std::string functionName;

	pElem = handle.FirstChild().Element();
	for (; 0 != pElem; pElem = pElem->NextSiblingElement()) {

		TiXmlHandle methodHandle(pElem);
		pElem->QueryStringAttribute("function", &functionName);
		
		LoadEnums(methodHandle, functionName, state);
	}
}


void 
StateLoader::LoadEnums(TiXmlHandle &hRoot, std::string functionName, IGlobalState *state)
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

		state->addStateEnum(name, functionName, value, length);
	}
}
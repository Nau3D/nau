#include "nau/interface/interface.h"

#include "nau.h"
#include "nau/config.h"
#include "nau/slogger.h"
#include "nau/math/vec2.h"
#include "nau/math/vec3.h"
#include "nau/math/vec4.h"
#include "nau/math/matrix.h"
#include "nau/system/textutil.h"


#include <algorithm>
#include <vector>

using namespace nau::inter;
using namespace nau::math;



// SINGLETON STUFF

ToolBar *ToolBar::Instance = NULL;


ToolBar*
ToolBar::GetInstance() {

	if (Instance == NULL) {

		Instance = new ToolBar();
	}
	return Instance;
}

ToolBar::~ToolBar() {

	clear();
}


ToolBar::ToolBar() {

}

// ----------------------------------------------------------------
// INSTANCE METHODS

void
ToolBar::clear() {

	m_Windows.clear();
}


void 
ToolBar::render() {
}


bool
ToolBar::createWindow(const std::string &label) {

	std::string name = label;
	name.erase(remove_if(name.begin(), name.end(), [](char c) { return !isalpha(c); }), name.end());

	if (m_Windows.count(label) != 0)
		return false;

	m_Windows[label] = Items(0);
	return true;
}


const std::map<std::string, ToolBar::Items>&
ToolBar::getWindows() {
	return m_Windows;
}



bool
ToolBar::addColor(const std::string &windowName, const std::string &varLabel,
	const std::string &varType, const std::string &varContext,
	const std::string &component, int id, const std::string &luaScript, const std::string &luaScriptFile) {

	// window does not exist
	if (m_Windows.count(windowName) == 0)
		return false;

	AttribSet *attrSet = NAU->getAttribs(varType);
	// type is not valid
	if (attrSet == NULL)
		return false;

	Enums::DataType dt;
	int attr;

	attrSet->getPropTypeAndId(component, &dt, &attr);
	// component is invalid
	if (attr == -1)
		return false;

	if (dt != Enums::VEC4)
		return false;

	NauVar var;
	var.dt = dt;
	var.label = varLabel;
	var.type = varType;
	var.context = varContext;
	var.component = component;
	var.luaScript = luaScript;
	var.luaScriptFile = luaScriptFile;
	var.semantics = Attribute::Semantics::COLOR;
	var.aClass = NAU_VAR;

	m_Windows[windowName].push_back(var);

	return true;
}


void 
ToolBar::parse(const std::string& s, float& min, float& max, float& step) {

	char* token = strtok((char*)s.c_str(), " ");

	min = 0.0f; max = 0.0f; step = 0.0f;
	while (token != NULL)
	{
		std::string ts = token;
		size_t pos = ts.find("=");
		std::string ff = ts.substr(pos + 1);
		std::string tt = ts.substr(0, pos);

		if (tt == "min") min = (float)atof(ff.c_str());
		if (tt == "max") max = (float)atof(ff.c_str());
		if (tt == "step") step = (float)atof(ff.c_str());

		token = strtok(NULL, " ");
	}
}


bool
ToolBar::addVar(const std::string &windowName, const std::string &varLabel,
	const std::string &varType, const std::string &varContext,
	const std::string &component, int id, const std::string def, const std::string &luaScript, const std::string &luaScriptFile) {

	// window does not exist
	if (m_Windows.count(windowName) == 0)
		return false;

	AttribSet *attrSet = NAU->getAttribs(varType);
	// type is not valid
	if (attrSet == NULL)
		return false;

	Enums::DataType dt;
	int attr;

	attrSet->getPropTypeAndId(component, &dt, &attr);
	// component is not valid
	if (attr == -1)
		return false;

	NauVar var;

	parse(def, var.min, var.max, var.step);


	var.dt = dt;
	var.aClass = NAU_VAR;
	var.label = varLabel;
	var.type = varType;
	var.context = varContext;
	var.component = component;
	var.luaScript = luaScript;
	var.luaScriptFile = luaScriptFile;
	var.semantics = Attribute::Semantics::NONE;
	var.id = id;

	m_Windows[windowName].push_back(var);

	return true;
}


bool
ToolBar::addEnum(const std::string& windowName, const std::string& varLabel,
	const std::string& varType, const std::string& varContext,
	const std::string& component, const std::string& enums, int id,
	const std::string def, const std::string& luaScript, const std::string& luaScriptFile) {

	if (m_Windows.count(windowName) == 0)
		return false;


	NauVar var;
	var.aClass = CUSTOM_ENUM;
	var.label = varLabel;
	var.type = varType;
	var.context = varContext;
	var.component = component;
	var.luaScript = luaScript;
	var.luaScriptFile = luaScriptFile;
	var.semantics = Attribute::Semantics::NONE;

	size_t k = enums.length() + 2;
	var.options.resize(k);
	int l = 0;
	for (int i = 0; i < enums.length(); ++i) {
		if (enums[i] == ',')
			var.options[l++] = '\0';
		else
			var.options[l++] = enums[i];
	}
	
	var.options[l] = '\0';
	m_Windows[windowName].push_back(var);

	return true;
}




bool 
ToolBar::addPipelineList(const std::string &windowName, const std::string &label, const std::string &luaScript, const std::string &luaScriptFile) {

	// window does not exist
	if (m_Windows.count(windowName) == 0)
		return false;

	std::vector<std::string> vs;
	RENDERMANAGER->getPipelineNames(&vs);

	size_t k = 0;
	for (auto& str : vs) {
		k += str.length()+1;
	}
	k++;

	NauVar var;
	var.aClass = PIPELINE_LIST;
	var.label =label;
	var.luaScript = luaScript;
	var.luaScriptFile = luaScriptFile;
	var.semantics = Attribute::Semantics::NONE;

	var.options.resize(k);
	int l = 0;
	for (auto& str : vs) {
		for (int i = 0; i < str.length(); ++i) {
			var.options[l++] = str[i];
		}
		var.options[l++] = '\0';
	}
	var.options[l] = '\0';
	m_Windows[windowName].push_back(var);

	return true;
}


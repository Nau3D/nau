#include "nau/interface/interface.h"

#include "nau.h"
#include "nau/config.h"
#include "nau/slogger.h"
#include "nau/math/data.h"

#include <algorithm>

using namespace nau::inter;
using namespace nau::math;

// VECs
TwStructMember ToolBar::Vec2Members[2] = {
	{ "X", TW_TYPE_FLOAT, offsetof(vec2, x), " " },
	{ "Y", TW_TYPE_FLOAT, offsetof(vec2, y), " " } };
TwType ToolBar::Vec2;

TwStructMember ToolBar::Vec3Members[3] = {
	{ "X", TW_TYPE_FLOAT, offsetof(vec3, x), " " },
	{ "Y", TW_TYPE_FLOAT, offsetof(vec3, y), " " },
	{ "Z", TW_TYPE_FLOAT, offsetof(vec3, z), " " } };
TwType ToolBar::Vec3;

TwStructMember ToolBar::Vec4Members[4] = {
	{ "X", TW_TYPE_FLOAT, offsetof(vec4, x), " " },
	{ "Y", TW_TYPE_FLOAT, offsetof(vec4, y), " " },
	{ "Z", TW_TYPE_FLOAT, offsetof(vec4, z), " " },
	{ "W", TW_TYPE_FLOAT, offsetof(vec4, w), " " } };
TwType ToolBar::Vec4;


// UIVECs
TwStructMember ToolBar::UIVec2Members[2] = {
	{ "X", TW_TYPE_UINT32, offsetof(uivec2, x), " " },
	{ "Y", TW_TYPE_UINT32, offsetof(uivec2, y), " " } };
TwType ToolBar::UIVec2;

TwStructMember ToolBar::UIVec3Members[3] = {
	{ "X", TW_TYPE_UINT32, offsetof(uivec3, x), " " },
	{ "Y", TW_TYPE_UINT32, offsetof(uivec3, y), " " },
	{ "Z", TW_TYPE_UINT32, offsetof(uivec3, z), " " } };
TwType ToolBar::UIVec3;


// IVEC2
TwStructMember ToolBar::IVec2Members[2] = {
	{ "X", TW_TYPE_INT32, offsetof(ivec2, x), " " },
	{ "Y", TW_TYPE_INT32, offsetof(ivec2, y), " " } };
TwType ToolBar::IVec2;

TwStructMember ToolBar::IVec3Members[3] = {
	{ "X", TW_TYPE_INT32, offsetof(ivec3, x), " " },
	{ "Y", TW_TYPE_INT32, offsetof(ivec3, y), " " },
	{ "Z", TW_TYPE_INT32, offsetof(ivec3, z), " " } };
TwType ToolBar::IVec3;


// BVECs
TwStructMember ToolBar::BVec4Members[4] = {
	{ "X", TW_TYPE_BOOLCPP, offsetof(bvec4, x), " " },
	{ "Y", TW_TYPE_BOOLCPP, offsetof(bvec4, y), " " },
	{ "Z", TW_TYPE_BOOLCPP, offsetof(bvec4, z), " " },
	{ "W", TW_TYPE_BOOLCPP, offsetof(bvec4, w), " " } };
TwType ToolBar::BVec4;

// MATRICES
TwStructMember ToolBar::Mat3Members[9] = {
	{ "R0C0", TW_TYPE_FLOAT, offsetof(mat3, m_Matrix), " " },
	{ "R0C1", TW_TYPE_FLOAT, offsetof(mat3, m_Matrix) + sizeof(float), " " },
	{ "R0C2", TW_TYPE_FLOAT, offsetof(mat3, m_Matrix) + sizeof(float) * 2, " " },
	{ "R1C0", TW_TYPE_FLOAT, offsetof(mat3, m_Matrix) + sizeof(float) * 3, " " },
	{ "R1C1", TW_TYPE_FLOAT, offsetof(mat3, m_Matrix) + sizeof(float) * 4, " " },
	{ "R1C2", TW_TYPE_FLOAT, offsetof(mat3, m_Matrix) + sizeof(float) * 5, " " },
	{ "R2C0", TW_TYPE_FLOAT, offsetof(mat3, m_Matrix) + sizeof(float) * 6, " " },
	{ "R2C1", TW_TYPE_FLOAT, offsetof(mat3, m_Matrix) + sizeof(float) * 7, " " },
	{ "R2C2", TW_TYPE_FLOAT, offsetof(mat3, m_Matrix) + sizeof(float) * 8, " " } };
TwType ToolBar::Mat3;

TwStructMember ToolBar::Mat4Members[16] = {
	{ "R0C0", TW_TYPE_FLOAT, offsetof(mat4, m_Matrix), " " },
	{ "R0C1", TW_TYPE_FLOAT, offsetof(mat4, m_Matrix) + sizeof(float), " " },
	{ "R0C2", TW_TYPE_FLOAT, offsetof(mat4, m_Matrix) + sizeof(float) * 2, " " },
	{ "R0C3", TW_TYPE_FLOAT, offsetof(mat4, m_Matrix) + sizeof(float) * 3, " " }, 
	{ "R1C0", TW_TYPE_FLOAT, offsetof(mat4, m_Matrix) + sizeof(float) * 4, " " },
	{ "R1C1", TW_TYPE_FLOAT, offsetof(mat4, m_Matrix) + sizeof(float) * 5, " " },
	{ "R1C2", TW_TYPE_FLOAT, offsetof(mat4, m_Matrix) + sizeof(float) * 6, " " },
	{ "R1C3", TW_TYPE_FLOAT, offsetof(mat4, m_Matrix) + sizeof(float) * 7, " " },
	{ "R2C0", TW_TYPE_FLOAT, offsetof(mat4, m_Matrix) + sizeof(float) * 8, " " },
	{ "R2C1", TW_TYPE_FLOAT, offsetof(mat4, m_Matrix) + sizeof(float) * 9, " " },
	{ "R2C2", TW_TYPE_FLOAT, offsetof(mat4, m_Matrix) + sizeof(float) * 10, " " },
	{ "R2C3", TW_TYPE_FLOAT, offsetof(mat4, m_Matrix) + sizeof(float) * 11, " " },
	{ "R3C0", TW_TYPE_FLOAT, offsetof(mat4, m_Matrix) + sizeof(float) * 12, " " },
	{ "R3C1", TW_TYPE_FLOAT, offsetof(mat4, m_Matrix) + sizeof(float) * 13, " " },
	{ "R3C2", TW_TYPE_FLOAT, offsetof(mat4, m_Matrix) + sizeof(float) * 14, " " },
	{ "R3C3", TW_TYPE_FLOAT, offsetof(mat4, m_Matrix) + sizeof(float) * 15, " " },
};
TwType ToolBar::Mat4;


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

#ifdef NAU_OPENGL
	TwInit(TW_OPENGL_CORE, NULL);
#endif
	Vec2 = TwDefineStruct("VEC2", Vec2Members, 2, sizeof(vec2), NULL, NULL);
	Vec3 = TwDefineStruct("VEC3", Vec3Members, 3, sizeof(vec3), NULL, NULL);
	Vec4 = TwDefineStruct("VEC4", Vec4Members, 4, sizeof(vec4), NULL, NULL);

	UIVec2 = TwDefineStruct("UIVEC2", UIVec2Members, 2, sizeof(uivec2), NULL, NULL);
	UIVec3 = TwDefineStruct("UIVEC3", UIVec3Members, 3, sizeof(uivec3), NULL, NULL);

	IVec2 = TwDefineStruct("IVEC2", IVec2Members, 2, sizeof(ivec2), NULL, NULL);
	IVec3 = TwDefineStruct("IVEC3", IVec3Members, 3, sizeof(ivec3), NULL, NULL);

	BVec4 = TwDefineStruct("BVEC4", BVec4Members, 4, sizeof(bvec4), NULL, NULL);

	Mat3 = TwDefineStruct("MAT3", Mat3Members, 9, sizeof(mat3), NULL, NULL);
	Mat4 = TwDefineStruct("MAT4", Mat4Members, 16, sizeof(mat4), NULL, NULL);

	TwDefine(" Testing iconifiable=true ");

	TwHandleErrors(ErrorHandler);
}


// INSTANCE METHODS


void
ToolBar::clear() {

	m_Windows.clear();
	TwDeleteAllBars();
}


void 
ToolBar::init() {


}

void 
ToolBar::render() {

	if (m_Windows.size() != 0)
		TwDraw();
}


void 
TW_CALL ToolBar::ErrorHandler(const char *errorMessage) {

	SLOG("AntTweakBar Error: %s", errorMessage);
}


bool
ToolBar::createWindow(const std::string &label) {

	std::string name = label;
	name.erase(remove_if(name.begin(), name.end(), [](char c) { return !isalpha(c); }), name.end());

	if (m_Windows.count(label) != 0)
		return false;

	TwBar *t = TwNewBar(name.c_str());
	char s[256];
	sprintf(s, " %s color='25 25 25' alpha=128 text=light label='%s'", name.c_str(), label.c_str());
	TwDefine(s);
	if (!t)
		return false;

	m_Windows[label] = std::pair<std::string, TwBar *>(name,t);
	return true;
}


bool
ToolBar::addColor(const std::string &windowName, const std::string &varLabel,
	const std::string &varType, const std::string &varContext,
	const std::string &component, int id) {

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

	NauVar *clientData = new NauVar();
	clientData->type = varType;
	clientData->context = varContext;
	clientData->component = component;

	std::string name = varLabel;
	name.erase(remove_if(name.begin(), name.end(), [](char c) { return !isalpha(c); }), name.end());
	char s[256];
	if (name != varLabel) {
		sprintf(s, " label='%s' ", varLabel.c_str());
	}
	else
		s[0] = '\0';

	if (attrSet->get(attr, dt)->getReadOnlyFlag() == false)
		return TwAddVarCB(m_Windows[windowName].second, varLabel.c_str(), TW_TYPE_COLOR4F, SetColorCallBack, GetColorCallBack, clientData, NULL) == 1;
	else
		return TwAddVarCB(m_Windows[windowName].second, varLabel.c_str(), TW_TYPE_COLOR4F, NULL, GetColorCallBack, clientData, NULL) == 1;
}


bool
ToolBar::addDir(const std::string &windowName, const std::string &varLabel,
	const std::string &varType, const std::string &varContext,
	const std::string &component, int id) {

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

	if (dt != Enums::VEC4)
		return false;

	NauVar *clientData = new NauVar();
	clientData->type = varType;
	clientData->context = varContext;
	clientData->component = component;

	std::string name = varLabel;
	name.erase(remove_if(name.begin(), name.end(), [](char c) { return !isalpha(c); }), name.end());
	char s[256];
	if (name != varLabel) {
		sprintf(s, " label='%s' ", varLabel.c_str());
	}
	else
		s[0] = '\0';

	if (attrSet->get(attr, dt)->getReadOnlyFlag() == false)
		return (TwAddVarCB(m_Windows[windowName].second, varLabel.c_str(), TW_TYPE_DIR3F, SetDirCallBack, GetDirCallBack, clientData, s) == 1);
	else
		return (TwAddVarCB(m_Windows[windowName].second, varLabel.c_str(), TW_TYPE_DIR3F, NULL, GetDirCallBack, clientData, s) == 1);

}


bool 
ToolBar::addVar(const std::string &windowName, const std::string &varLabel,
	const std::string &varType, const std::string &varContext,
	const std::string &component, int id) {

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

	NauVar *clientData = new NauVar();
	clientData->type = varType;
	clientData->context = varContext;
	clientData->component = component;

	TwBar *t = m_Windows[windowName].second;

	std::string name = varLabel;
	name.erase(remove_if(name.begin(), name.end(), [](char c) { return !isalpha(c); }), name.end());
	char s[256];
	if (name != varLabel) {
		sprintf(s, " label='%s' ", varLabel.c_str());
	}
	else
		s[0] = '\0';

	switch (dt)
	{
	case nau::Enums::INT:
		if (attrSet->get(attr, dt)->getReadOnlyFlag() == false)
			return (TwAddVarCB(t, varLabel.c_str(), TW_TYPE_INT32, SetIntCallBack, GetIntCallBack, clientData, s) == 1);
		else
			return (TwAddVarCB(t, varLabel.c_str(), TW_TYPE_INT32, NULL, GetIntCallBack, clientData, s) == 1);
		break;
	case nau::Enums::UINT:
		if (attrSet->get(attr, dt)->getReadOnlyFlag() == false)
			return (TwAddVarCB(t, varLabel.c_str(), TW_TYPE_UINT32, SetUIntCallBack, GetUIntCallBack, clientData, s) == 1);
		else
			return (TwAddVarCB(t, varLabel.c_str(), TW_TYPE_UINT32, NULL, GetUIntCallBack, clientData, s) == 1);
		break;
	case nau::Enums::UIVEC2:
		if (attrSet->get(attr, dt)->getReadOnlyFlag() == false)
			return (TwAddVarCB(t, varLabel.c_str(), UIVec2, SetCallBack, GetUIVec2CallBack, clientData, s) == 1);
		else
			return (TwAddVarCB(t, varLabel.c_str(), UIVec2, NULL, GetUIVec2CallBack, clientData, s) == 1);
		break;
	case nau::Enums::UIVEC3:
		if (attrSet->get(attr, dt)->getReadOnlyFlag() == false)
			return (TwAddVarCB(t, varLabel.c_str(), UIVec3, SetCallBack, GetUIVec3CallBack, clientData, s) == 1);
		else
			return (TwAddVarCB(t, varLabel.c_str(), UIVec3, NULL, GetUIVec3CallBack, clientData, s) == 1);
		break;
	case nau::Enums::BOOL:
		if (attrSet->get(attr, dt)->getReadOnlyFlag() == false)
			return (TwAddVarCB(t, varLabel.c_str(), TW_TYPE_BOOLCPP, SetBoolCallBack, GetBoolCallBack, clientData, s) == 1);
		else
			return (TwAddVarCB(t, varLabel.c_str(), TW_TYPE_BOOLCPP, NULL, GetBoolCallBack, clientData, s) == 1);
		break;
	case nau::Enums::BVEC4:
		if (attrSet->get(attr, dt)->getReadOnlyFlag() == false)
			return (TwAddVarCB(t, varLabel.c_str(), BVec4, SetCallBack, GetBVec4CallBack, clientData, s) == 1);
		else
			return (TwAddVarCB(t, varLabel.c_str(), BVec4, NULL, GetBVec4CallBack, clientData, s) == 1);
		break;
	case nau::Enums::FLOAT:
		if (attrSet->get(attr, dt)->getReadOnlyFlag() == false)
			return (TwAddVarCB(t, varLabel.c_str(), TW_TYPE_FLOAT, SetFloatCallBack, GetFloatCallBack, clientData, s) == 1);
		else
			return (TwAddVarCB(t, varLabel.c_str(), TW_TYPE_FLOAT, NULL, GetFloatCallBack, clientData, s) == 1);
		break;
	case nau::Enums::VEC2:
		if (attrSet->get(attr, dt)->getReadOnlyFlag() == false)
			return (TwAddVarCB(t, varLabel.c_str(), Vec2, SetCallBack, GetVec2CallBack, clientData, s) == 1);
		else
			return (TwAddVarCB(t, varLabel.c_str(), Vec2, NULL, GetVec2CallBack, clientData, s) == 1);
		break;
	case nau::Enums::VEC3:
		if (attrSet->get(attr, dt)->getReadOnlyFlag() == false)
			return (TwAddVarCB(t, varLabel.c_str(), Vec3, SetCallBack, GetVec3CallBack, clientData, s) == 1);
		else
			return (TwAddVarCB(t, varLabel.c_str(), Vec3, NULL, GetVec3CallBack, clientData, s) == 1);
		break;
	case nau::Enums::VEC4:
		if (attrSet->get(attr, dt)->getReadOnlyFlag() == false)
			return (TwAddVarCB(t, varLabel.c_str(), Vec4, SetCallBack, GetVec4CallBack, clientData, s) == 1);
		else
			return (TwAddVarCB(t, varLabel.c_str(), Vec4, NULL, GetVec4CallBack, clientData, s) == 1);
		break;
	case nau::Enums::MAT3:
		if (attrSet->get(attr, dt)->getReadOnlyFlag() == false)
			return (TwAddVarCB(t, varLabel.c_str(), Mat3, SetCallBack, GetMat3CallBack, clientData, s) == 1);
		else
			return (TwAddVarCB(t, varLabel.c_str(), Mat3, NULL, GetMat3CallBack, clientData, s) == 1);
		break;
	case nau::Enums::MAT4:
		if (attrSet->get(attr, dt)->getReadOnlyFlag() == false)
			return (TwAddVarCB(t, varLabel.c_str(), Mat4, SetCallBack, GetMat4CallBack, clientData, s) == 1);
		else
			return (TwAddVarCB(t, varLabel.c_str(), Mat4, NULL, GetMat4CallBack, clientData, s) == 1);
		break;
	case nau::Enums::ENUM: 
	{
		const std::vector<std::string> &vs = NAU->getAttribs(varType)->getListString(id);
		const std::vector<int> &vi = NAU->getAttribs(varType)->getListValues(id);
		TwEnumVal *enums;
		enums = (TwEnumVal *)malloc(vi.size() * sizeof(TwEnumVal));
		for (int i = 0; i < vi.size(); ++i) {
			enums[i].Label = vs[i].c_str();
			enums[i].Value = vi[i];
		}
		TwType options = TwDefineEnum(varLabel.c_str(), enums, (unsigned int)vi.size());
		if (attrSet->get(id, dt)->getReadOnlyFlag() == false)
			return (TwAddVarCB(t, varLabel.c_str(), options, SetIntCallBack, GetIntCallBack, clientData, s) == 1);
		else
			return (TwAddVarCB(t, varLabel.c_str(), options, NULL, GetIntCallBack, clientData, s) == 1);
	}
		break;
	default:
		assert(false && "Missing type in ToolBar::addVar");
		return false;
	}
	return true;
}


bool 
ToolBar::addPipelineList(const std::string &windowName, const std::string &label) {

	// window does not exist
	if (m_Windows.count(windowName) == 0)
		return false;

	std::vector<std::string> vs;
	RENDERMANAGER->getPipelineNames(&vs);

	TwEnumVal *enums;
	enums = (TwEnumVal *)malloc(vs.size() * sizeof(TwEnumVal));
	for (int i = 0; i < vs.size(); ++i) {
		enums[i].Label = vs[i].c_str();
		enums[i].Value = i;
	}
	TwType options = TwDefineEnum("Pipelines", enums, (unsigned int)vs.size());

	std::string name = label;
	name.erase(remove_if(name.begin(), name.end(), [](char c) { return !isalpha(c); }), name.end());
	char s[256];
	if (name != label) {
		sprintf(s, " label='%s' ", label.c_str());
	}
	else
		s[0] = '\0';
	return (TwAddVarCB(m_Windows[windowName].second, name.c_str(), options, SetPipelineCallBack, GetPipelineCallBack, NULL, s) == 1);
}

// STATIC METHODS


void
TW_CALL ToolBar::SetPipelineCallBack(const void *value, void *clientData) {

	RENDERMANAGER->setActivePipeline(*(int *)value);
}


void
TW_CALL ToolBar::GetPipelineCallBack(void *value, void *clientData) {

	std::string pipName = RENDERMANAGER->getActivePipeline()->getName();
	*(int *)value = RENDERMANAGER->getPipelineIndex(pipName);
}


void
TW_CALL ToolBar::SetColorCallBack(const void *value, void *clientData) {

	NauVar *v = static_cast<NauVar *>(clientData);
	vec4 v2;
	v2.set(*(float *)value, *((float *)value + 1), *((float *)value + 2), *((float *)value + 3));
	NAU->setAttribute(v->type, v->context, v->component, v->id, &v2);
}


void 
TW_CALL ToolBar::GetColorCallBack(void * value, void * clientData) {

	NauVar *v = static_cast<NauVar *>(clientData);
	vec4 *v2 = (vec4 *)NAU->getAttribute(v->type, v->context, v->component, v->id);
	memcpy(value, &(v2->x), sizeof(float) *4);
}


void 
TW_CALL ToolBar::SetDirCallBack(const void * value, void * clientData) {

	NauVar *v = static_cast<NauVar *>(clientData);
	vec4 v2;
	v2.set(*(float *)value, *((float *)value + 1), *((float *)value + 2), 0);
	NAU->setAttribute(v->type, v->context, v->component, v->id, &v2);
}


void 
TW_CALL ToolBar::GetDirCallBack(void * value, void * clientData) {

	NauVar *v = static_cast<NauVar *>(clientData);
	vec4 *v2 = (vec4 *)NAU->getAttribute(v->type, v->context, v->component, v->id);
	memcpy(value, &(v2->x), sizeof(float) * 3);
}


void
TW_CALL ToolBar::SetCallBack(const void *value, void *clientData) {

	NauVar *v = static_cast<NauVar *>(clientData);
	Data *v2 = (Data *)value;
	NAU->setAttribute(v->type, v->context, v->component, v->id, v2);
}


void 
TW_CALL ToolBar::SetIntCallBack(const void *value, void *clientData) {

	NauVar *v = static_cast<NauVar *>(clientData);
	std::shared_ptr<NauInt> p = std::shared_ptr<NauInt>(new NauInt(*(int *)value));
	NAU->setAttribute(v->type, v->context, v->component, v->id, p.get());
}


void
TW_CALL ToolBar::SetBoolCallBack(const void *value, void *clientData) {

	NauVar *v = static_cast<NauVar *>(clientData);
	std::shared_ptr<NauInt> p = std::shared_ptr<NauInt>(new NauInt(*(int *)value));
	NAU->setAttribute(v->type, v->context, v->component, v->id, p.get());
}


void
TW_CALL ToolBar::ToolBar::SetFloatCallBack(const void *value, void *clientData) {

	NauVar *v = static_cast<NauVar *>(clientData);
	std::shared_ptr<NauFloat> p = std::shared_ptr<NauFloat>(new NauFloat(*(float *)value));
	NAU->setAttribute(v->type, v->context, v->component, v->id, p.get());
}


void 
TW_CALL ToolBar::SetUIntCallBack(const void *value, void *clientData) {

	NauVar *v = static_cast<NauVar *>(clientData);
	std::shared_ptr<NauUInt> p = std::shared_ptr<NauUInt>(new NauUInt(*(unsigned int *)value));
	NAU->setAttribute(v->type, v->context, v->component, v->id, p.get());
}




void 
TW_CALL ToolBar::GetIntCallBack(void *value, void *clientData) {

	NauVar *v = static_cast<NauVar *>(clientData);
	int *v2 = (int *)NAU->getAttribute(v->type, v->context, v->component, v->id);
	*(int *)value = *v2;
}


void 
TW_CALL ToolBar::GetFloatCallBack(void *value, void *clientData) {

	NauVar *v = static_cast<NauVar *>(clientData);
	float *v2 = (float *)NAU->getAttribute(v->type, v->context, v->component, v->id);
	*(float *)value = *v2;
}


void 
TW_CALL ToolBar::GetVec2CallBack(void *value, void *clientData) {

	NauVar *v = static_cast<NauVar *>(clientData);
	Data *v2 = (vec2 *)NAU->getAttribute(v->type, v->context, v->component, v->id);
	memcpy(value, v2, sizeof(vec2));
}


void 
TW_CALL ToolBar::GetVec3CallBack(void *value, void *clientData) {

	NauVar *v = static_cast<NauVar *>(clientData);
	Data *v2 = (vec3 *)NAU->getAttribute(v->type, v->context, v->component, v->id);
	memcpy(value, v2, sizeof(vec3));
}


void 
TW_CALL ToolBar::GetVec4CallBack(void *value, void *clientData) {

	NauVar *v = static_cast<NauVar *>(clientData);
	vec4 *v2 = (vec4 *)NAU->getAttribute(v->type, v->context, v->component, v->id);
	memcpy(value, v2, sizeof(vec4));
}


void 
TW_CALL ToolBar::GetUIntCallBack(void *value, void *clientData) {

	NauVar *v = static_cast<NauVar *>(clientData);
	unsigned int *v2 = (unsigned int *)NAU->getAttribute(v->type, v->context, v->component, v->id);
	*(unsigned int *)value = *v2;
}


void 
TW_CALL ToolBar::GetUIVec2CallBack(void *value, void *clientData) {

	NauVar *v = static_cast<NauVar *>(clientData);
	Data *v2 = (uivec2 *)NAU->getAttribute(v->type, v->context, v->component, v->id);
	memcpy(value, v2, sizeof(uivec2));
}


void 
TW_CALL ToolBar::GetUIVec3CallBack(void *value, void *clientData) {

	NauVar *v = static_cast<NauVar *>(clientData);
	Data *v2 = (uivec3 *)NAU->getAttribute(v->type, v->context, v->component, v->id);
	memcpy(value, v2, sizeof(uivec3));
}


void TW_CALL nau::inter::ToolBar::GetBoolCallBack(void * value, void * clientData)
{
	NauVar *v = static_cast<NauVar *>(clientData);
	int *v2 = (int *)NAU->getAttribute(v->type, v->context, v->component, v->id);
	*(bool *)value = (*v2 == 1);
}

void
TW_CALL ToolBar::GetBVec4CallBack(void *value, void *clientData) {

	NauVar *v = static_cast<NauVar *>(clientData);
	Data *v2 = (bvec4 *)NAU->getAttribute(v->type, v->context, v->component, v->id);
	memcpy(value, v2, sizeof(bvec4));
}


void 
TW_CALL ToolBar::GetMat3CallBack(void * value, void * clientData) {
	NauVar *v = static_cast<NauVar *>(clientData);
	Data *v2 = (mat3 *)NAU->getAttribute(v->type, v->context, v->component, v->id);
	memcpy(value, v2, sizeof(mat3));
}


void 
TW_CALL ToolBar::GetMat4CallBack(void * value, void * clientData) {
	NauVar *v = static_cast<NauVar *>(clientData);
	Data *v2 = (mat4 *)NAU->getAttribute(v->type, v->context, v->component, v->id);
	memcpy(value, v2, sizeof(mat4));
}



#include "nau/scene/iScene.h"

#include "nau.h"
#include "nau/math/matrix.h"

using namespace nau::math;
using namespace nau::scene;

bool 
IScene::Init() {

	// VEC4
	Attribs.add(Attribute(SCALE, "SCALE", Enums::DataType::VEC4, false, new vec4(1.0f, 1.0f, 1.0f, 1.0f)));
	Attribs.add(Attribute(TRANSLATE, "TRANSLATE", Enums::DataType::VEC4, false, new vec4(0.0f, 0.0f, 0.0f, 0.0f)));
	Attribs.add(Attribute(ROTATE, "ROTATE", Enums::DataType::VEC4, false, new vec4(0.0f, 1.0f, 0.0f, 0.0f)));

	// VEC3
	Attribs.add(Attribute(BB_MAX, "BB_MAX", Enums::DataType::VEC3, true, new vec3(-1.0f, -1.0f, -1.0f)));
	Attribs.add(Attribute(BB_MIN, "BB_MIN", Enums::DataType::VEC3, true, new vec3(1.0f, 1.0f, 1.0f)));
	Attribs.add(Attribute(BB_CENTER, "BB_CENTER", Enums::DataType::VEC3, true, new vec3(0.0f, 0.0f, 0.0f)));

	// ENUM
	Attribs.add(Attribute(TRANSFORM_ORDER, "TRANSFORM_ORDER", Enums::ENUM, false, new NauInt(T_R_S)));
	Attribs.listAdd("TRANSFORM_ORDER", "T_R_S", T_R_S);
	Attribs.listAdd("TRANSFORM_ORDER", "T_S_R", T_S_R);
	Attribs.listAdd("TRANSFORM_ORDER", "R_T_S", R_T_S);
	Attribs.listAdd("TRANSFORM_ORDER", "R_S_T", R_S_T);
	Attribs.listAdd("TRANSFORM_ORDER", "S_R_T", S_R_T);
	Attribs.listAdd("TRANSFORM_ORDER", "S_T_R", S_T_R);

	// UINT
	Attribs.add(Attribute(TRIANGLE_COUNT, "TRIANGLE_COUNT", Enums::DataType::UINT, true, new NauUInt(0)));


	//#ifndef _WINDLL
	NAU->registerAttributes("SCENE", &Attribs);
	//#endif

	return true;
}


AttribSet &
IScene::GetAttribs() {
	return Attribs; 
}

AttribSet IScene::Attribs;
bool IScene::Inited = Init();


void 
IScene::updateTransform() {

	mat4 tis; 

	switch (m_EnumProps[TRANSFORM_ORDER]) {

	case T_R_S:
		tis.setIdentity();
		tis.translate(m_Float4Props[TRANSLATE].x, m_Float4Props[TRANSLATE].y, m_Float4Props[TRANSLATE].z);
		tis.rotate(m_Float4Props[ROTATE].w, m_Float4Props[ROTATE].x, m_Float4Props[ROTATE].y, m_Float4Props[ROTATE].z);
		tis.scale(m_Float4Props[SCALE].x, m_Float4Props[SCALE].y, m_Float4Props[SCALE].z);
		break;
	case T_S_R:
		tis.setIdentity();
		tis.translate(m_Float4Props[TRANSLATE].x, m_Float4Props[TRANSLATE].y, m_Float4Props[TRANSLATE].z);
		tis.scale(m_Float4Props[SCALE].x, m_Float4Props[SCALE].y, m_Float4Props[SCALE].z);
		tis.rotate(m_Float4Props[ROTATE].w, m_Float4Props[ROTATE].x, m_Float4Props[ROTATE].y, m_Float4Props[ROTATE].z);
		break;
	case R_T_S:
		tis.setIdentity();
		tis.rotate(m_Float4Props[ROTATE].w, m_Float4Props[ROTATE].x, m_Float4Props[ROTATE].y, m_Float4Props[ROTATE].z);
		tis.translate(m_Float4Props[TRANSLATE].x, m_Float4Props[TRANSLATE].y, m_Float4Props[TRANSLATE].z);
		tis.scale(m_Float4Props[SCALE].x, m_Float4Props[SCALE].y, m_Float4Props[SCALE].z);
		break;
	case R_S_T:
		tis.setIdentity();
		tis.rotate(m_Float4Props[ROTATE].w, m_Float4Props[ROTATE].x, m_Float4Props[ROTATE].y, m_Float4Props[ROTATE].z);
		tis.scale(m_Float4Props[SCALE].x, m_Float4Props[SCALE].y, m_Float4Props[SCALE].z);
		tis.translate(m_Float4Props[TRANSLATE].x, m_Float4Props[TRANSLATE].y, m_Float4Props[TRANSLATE].z);
		break;
	case S_R_T:
		tis.setIdentity();
		tis.scale(m_Float4Props[SCALE].x, m_Float4Props[SCALE].y, m_Float4Props[SCALE].z);
		tis.rotate(m_Float4Props[ROTATE].w, m_Float4Props[ROTATE].x, m_Float4Props[ROTATE].y, m_Float4Props[ROTATE].z);
		tis.translate(m_Float4Props[TRANSLATE].x, m_Float4Props[TRANSLATE].y, m_Float4Props[TRANSLATE].z);
		break;
	case S_T_R:
		tis.setIdentity();
		tis.scale(m_Float4Props[SCALE].x, m_Float4Props[SCALE].y, m_Float4Props[SCALE].z);
		tis.translate(m_Float4Props[TRANSLATE].x, m_Float4Props[TRANSLATE].y, m_Float4Props[TRANSLATE].z);
		tis.rotate(m_Float4Props[ROTATE].w, m_Float4Props[ROTATE].x, m_Float4Props[ROTATE].y, m_Float4Props[ROTATE].z);
		break;
	}
	setTransform(tis);
	std::shared_ptr<nau::event_::IEventData> e3 = nau::event_::EventFactory::Create("String");
	std::string * name = new std::string(m_Name);
	e3->setData(name);
	delete name;
	EVENTMANAGER->notifyEvent("SCENE_TRANSFORM", "SCENE", "", e3);
}


void
IScene::setPropf4(Float4Property prop, vec4& aVec) {

	switch (prop) {
	case SCALE:
	case ROTATE:
	case TRANSLATE:
		m_Float4Props[prop] = aVec;
		updateTransform();
		break;
	default:
		AttributeValues::setPropf4(prop, aVec);
	}
}


void 
IScene::setPrope(EnumProperty prop, int v) {

	switch (prop) {
	case TRANSFORM_ORDER:
		m_EnumProps[prop] = v;
		updateTransform();
		break;
	default:
		AttributeValues::setPrope(prop, v);
	}
}


vec3 &
IScene::getPropf3(Float3Property prop) {

	switch (prop) {
	case BB_MIN: m_Float3Props[BB_MIN] = getBoundingVolume().getMin();
		return m_Float3Props[BB_MIN];
	case BB_MAX: m_Float3Props[BB_MAX] = getBoundingVolume().getMax();
		return m_Float3Props[BB_MAX];
	case BB_CENTER: m_Float3Props[BB_CENTER] = getBoundingVolume().getCenter();
		return m_Float3Props[BB_CENTER];
	default: return AttributeValues::getPropf3(prop);
	}
}


const std::string &
IScene::getType() {

	return m_Type;
}


void nau::scene::IScene::recompile() {

	m_Compiled = false;

	compile();
}

#include <nau/render/istate.h>
#include <nau/config.h>

#ifdef NAU_OPENGL
#include <nau/render/opengl/glstate.h>	
#elif NAU_DIRECTX
#include <nau/render/dx/dxstate.h>
#endif

#include <nau.h>

using namespace nau::render;
using namespace nau;

bool
IState::Init() {

	// BOOL
	Attribs.add(Attribute(BLEND, "BLEND", Enums::DataType::BOOL, false, new bool(false)));
	//Attribs.add(Attribute(FOG, "FOG", Enums::DataType::BOOL, false, new bool(false)));
	//Attribs.add(Attribute(ALPHA_TEST, "ALPHA_TEST", Enums::DataType::BOOL, false, new bool(false)));
	Attribs.add(Attribute(DEPTH_TEST, "DEPTH_TEST", Enums::DataType::BOOL, false, new bool(true)));
	Attribs.add(Attribute(CULL_FACE, "CULL_FACE", Enums::DataType::BOOL, false, new bool(true)));
	Attribs.add(Attribute(COLOR_MASK, "COLOR_MASK", Enums::DataType::BOOL, false, new bool(false)));
	Attribs.add(Attribute(DEPTH_MASK, "DEPTH_MASK", Enums::DataType::BOOL, false, new bool(true)));

	// ENUM
	//Attribs.add(Attribute(FOG_MODE, "FOG_MODE", Enums::DataType::ENUM, false));
	//Attribs.add(Attribute(FOG_COORD_SRC, "FOG_COORD_SRC", Enums::DataType::ENUM, false));
	Attribs.add(Attribute(DEPTH_FUNC, "DEPTH_FUNC", Enums::DataType::ENUM, false));
	Attribs.add(Attribute(CULL_TYPE, "CULL_TYPE", Enums::DataType::ENUM, false));
	Attribs.add(Attribute(ORDER_TYPE, "ORDER_TYPE", Enums::DataType::ENUM, false));
	Attribs.add(Attribute(BLEND_SRC, "BLEND_SRC", Enums::DataType::ENUM, false));
	Attribs.add(Attribute(BLEND_DST, "BLEND_DST", Enums::DataType::ENUM, false));
	Attribs.add(Attribute(BLEND_EQUATION, "BLEND_EQUATION", Enums::DataType::ENUM, false));
	//Attribs.add(Attribute(ALPHA_FUNC, "ALPHA_FUNC", Enums::DataType::ENUM, false));

	// FLOAT
	//Attribs.add(Attribute(FOG_START, "FOG_START", Enums::DataType::FLOAT, false, new float(0)));
	//Attribs.add(Attribute(FOG_END, "FOG_END", Enums::DataType::FLOAT, false, new float(1)));
	//Attribs.add(Attribute(FOG_DENSITY, "FOG_DENSITY", Enums::DataType::FLOAT, false, new float(1)));
	//Attribs.add(Attribute(ALPHA_VALUE, "ALPHA_VALUE", Enums::DataType::FLOAT, false, new float(0)));

	// INT
	Attribs.add(Attribute(ORDER, "ORDER", Enums::DataType::INT, false, new int(0)));

	// FLOAT4
	Attribs.add(Attribute(BLEND_COLOR, "BLEND_COLOR", Enums::DataType::VEC4, false, new vec4(0.0f, 0.0f, 0.0f, 0.0f)));
	//Attribs.add(Attribute(FOG_COLOR, "FOG_COLOR", Enums::DataType::VEC4, false, new vec4(0.0f, 0.0f, 0.0f, 0.0f)));

	// BOOL4
	Attribs.add(Attribute(COLOR_MASK_B4, "COLOR_MASK_B4", Enums::DataType::BVEC4, false, new bvec4(true, true, true, true)));

	NAU->registerAttributes("STATE", &Attribs);

	return true;
}


AttribSet IState::Attribs;
bool IState::Inited = Init();


IState*
IState::create() {

#ifdef NAU_OPENGL
	return new GlState ;
#elif NAU_DIRECTX
	return new DXState;
#endif
}


IState::IState():
		m_defColor(0.0f, 0.0f, 0.0f, 1.0f)
{
	registerAndInitArrays("STATE", Attribs);
	m_Name = "default";
}

			
IState* 
IState::clone() {

	IState *res = IState::create();

	copy(res);

	//std::map< int, int>::iterator iterInt;
	//iterInt = m_IntProps.begin();
	//for ( ; iterInt != m_IntProps.end(); ++iterInt) {
	//
	//	res->m_IntProps[iterInt->first] = iterInt->second;
	//}

	//std::map< int, bool>::iterator iterBool;
	//iterBool = m_EnableProps.begin();
	//for ( ; iterBool != m_EnableProps.end(); ++iterBool) {
	//
	//	res->m_EnableProps[iterBool->first] = iterBool->second;
	//}

	//std::map< int, float>::iterator iterFloat;
	//iterFloat = m_FloatProps.begin();
	//for ( ; iterFloat != m_FloatProps.end(); ++iterFloat) {
	//
	//	res->m_FloatProps[iterFloat->first] = iterFloat->second;
	//}

	//std::map< int, vec4>::iterator iterVec4;
	//iterVec4 = m_Float4Props.begin();
	//for ( ; iterVec4 != m_Float4Props.end(); ++iterVec4) {
	//
	//	res->m_Float4Props[iterVec4->first] = iterVec4->second;
	//}

	//std::map< int, bvec4>::iterator iterBool4;
	//iterBool4 = m_Bool4Props.begin();
	//for ( ; iterBool4 != m_Bool4Props.end(); ++iterBool4) {
	//
	//	res->m_Bool4Props[iterBool4->first] = iterBool4->second;
	//}

	//iterInt = m_EnumProps.begin();
	//for ( ; iterInt != m_EnumProps.end(); ++iterInt) {
	//
	//	res->m_EnumProps[iterInt->first] = iterInt->second;
	//}
	return res;
}


//void 
//IState::clear() {
//
//	m_IntProps.clear();
//	m_EnableProps.clear();
//	m_FloatProps.clear();
//	m_Float4Props.clear();
//	m_Bool4Props.clear();
//	m_EnumProps.clear();
//}


void 
IState::setDefault() {

	initArrays();
	//Attribs.initAttribInstanceBoolArray(m_EnableProps);
	//Attribs.initAttribInstanceEnumArray(m_EnumProps);
	//Attribs.initAttribInstanceFloatArray(m_FloatProps);
	//Attribs.initAttribInstanceVec4Array(m_Float4Props);
	//Attribs.initAttribInstanceIntArray(m_IntProps);
	//Attribs.initAttribInstanceBvec4Array(m_Bool4Props);
	m_Name = "default";


}

//
// GET methods
//


std::string
IState::getName() {

	return(m_Name);
}


void 
IState::setName(std::string aName) {

	m_Name = aName;
}


//void *
//IState::getProp(int prop, Enums::DataType type) {
//
//	switch (type) {
//
//	case Enums::FLOAT:
//		assert(m_FloatProps.count(prop) > 0);
//		return(&(m_FloatProps[prop]));
//		break;
//	case Enums::VEC4:
//		assert(m_Float4Props.count(prop) > 0);
//		return(&(m_Float4Props[prop]));
//		break;
//	case Enums::INT:
//		assert(m_IntProps.count(prop) > 0);
//		return(&(m_IntProps[prop]));
//		break;
//	case Enums::ENUM:
//		assert(m_EnumProps.count(prop) > 0);
//		return(&(m_EnumProps[prop]));
//		break;
//	case Enums::BOOL:
//		assert(m_EnableProps.count(prop) > 0);
//		return(&(m_EnableProps[prop]));
//		break;
//		
//	}
//	return NULL;
//}


//void 
//IState::setProp(int prop, Enums::DataType type, void *value) {
//
//	switch (type) {
//
//		case Enums::FLOAT:
//			if (prop < COUNT_FLOATPROPERTY)
//				setProp((FloatProperty)prop, *(float *)value);
//			else
//				m_FloatProps[prop] = *(float *)value;
//			break;
//		case Enums::VEC4:
//			if (prop < COUNT_FLOAT4PROPERTY)
//				setProp((Float4Property)prop, *(vec4 *)value);
//			else
//				m_Float4Props[prop].set((vec4 *)value);
//			break;
//		case Enums::INT:
//			if (prop < COUNT_INTPROPERTY)
//				setProp((IntProperty)prop, *(int *)value);
//			else
//				m_IntProps[prop] = *(int *)value;
//			break;
//		case Enums::ENUM:
//			assert(prop < COUNT_ENUMPROPERTY);
//			setProp((EnumProperty)prop,*(int *)value);
//			break;
//		case Enums::BOOL:
//			assert(prop < COUNT_BOOLPROPERTY);
//			setProp((BoolProperty)prop,*(bool *)value);
//			break;
//		case Enums::BVEC4:
//			assert(prop < COUNT_BOOL4PROPERTY);
//			setProp((Bool4Property)prop,*(bvec4 *)value);
//			break;
//
//	}
//}	


//int 
//IState::getPropi(IntProperty aProp) {
//
//	//return(m_IntProps[aProp]);
//
//	if (m_IntProps.count(aProp) != 0)
//		return (m_IntProps[aProp]);
//
//	//else return default value
//	switch(aProp) {
//		case ORDER: return 0;
//		default: return 0;
//	}
//}
//
//
//int 
//IState::getPrope(EnumProperty aProp){
//
//	//return(m_EnumProps[aProp]);
//
//	if (m_EnumProps.count(aProp) != 0)
//		return (m_EnumProps[aProp]);
//	//else return default value
//	else {
//		void *v = Attribs.getDefault(aProp, Enums::ENUM);
//		if (v != NULL)
//			return *(int *)v;
//		else 
//			return 0;
//	}
//}
//
//
//float 
//IState::getPropf(FloatProperty aProp) {
//	
//	//return(m_FloatProps[aProp]);
//
//	if (m_FloatProps.count(aProp) != 0)
//		return (m_FloatProps[aProp]);
//	else {
//		void *v = Attribs.getDefault(aProp, Enums::FLOAT);
//		if (v != NULL)
//			return *(float *)v;
//		else 
//			return 0.0f;
//	}
//}
//
//
//const vec4&
//IState::getProp4f(Float4Property aProp) {
//
//	//return m_Float4Props[aProp];
//
//	if (m_Float4Props.count(aProp) != 0)
//		return (m_Float4Props[aProp]);
//	else {
//		void *v = Attribs.getDefault(aProp, Enums::VEC4);
//		if (v != NULL)
//			return *(vec4 *)v;
//		else 
//			return m_defColor;
//	}
//}
//
//
//const bvec4&
//IState::getProp4b(Bool4Property aProp) {
//
//
//	if (m_Bool4Props.count(aProp) != 0) {
//
//		return m_Bool4Props[aProp];
//	}
//	else {
//		void *v = Attribs.getDefault(aProp, Enums::BVEC4);
//		if (v != NULL)
//			return *(bvec4 *)v;
//		else 
//			return m_defBoolV;
//	}
//}
//
//
//bool
//IState::getPropb(BoolProperty aProp) {
//
//	//return(m_EnableProps[s]);
//
//	if (m_EnableProps.count(aProp) != 0)
//		return (m_EnableProps[aProp]);
//	else {
//		void *v = Attribs.getDefault(aProp, Enums::BOOL);
//		if (v != NULL)
//			return *(bool *)v;
//		else 
//			return 0.0f;
//	}
//}


//
// SET methods
//



//void 
//IState::setProp(BoolProperty s, bool value) {
//
//	m_EnableProps[s] = value;
//}
//
//
//void
//IState::setProp(IntProperty prop, int value) {
//
//	m_IntProps[prop] = value;
//}
//
//
//void
//IState::setProp(FloatProperty prop, float value) {
//
//	m_FloatProps[prop] = value;
//}
//
//
//void
//IState::setProp(Float4Property prop, float r, float g, float b, float a){
//
//	m_Float4Props[prop].set(r,g,b,a);
//}
//
//
//void
//IState::setProp(Float4Property prop, const vec4& values){
//
//	m_Float4Props[prop].set(values);
//
//}
//
//
//void
//IState::setProp(Bool4Property prop, bool r, bool g, bool b, bool a){
//
//	m_Bool4Props[prop].set(r,g,b,a);
//}
//
//
//void
//IState::setProp(Bool4Property prop, bool* values) {
//
//	m_Bool4Props[prop].set(values[0], values [1], values [2], values[3]);
//}
//
//
//void
//IState::setProp(Bool4Property prop, bvec4& values) {
//
//	m_Bool4Props[prop].set(values);
//}
//
//void
//IState::setProp(EnumProperty prop, int value){
//
//	m_EnumProps[prop] = value;
//}



/* Is a property defined? */
//
//bool 
//IState::isSetProp(IntProperty prop) {
//
//	return (m_IntProps.count(prop) != 0);
//}
//
//
//bool 
//IState::isSetProp(FloatProperty prop) {
//
//	return (m_FloatProps.count(prop) != 0);
//}
//
//
//bool 
//IState::isSetProp(Float4Property prop) {
//
//	return (m_Float4Props.count(prop) != 0);
//}
//
//
//bool 
//IState::isSetProp(Bool4Property prop) {
//
//	return (m_Bool4Props.count(prop) != 0);
//}
//
//
//bool 
//IState::isSetProp(EnumProperty prop) {
//
//	return (m_EnumProps.count(prop) != 0);
//}



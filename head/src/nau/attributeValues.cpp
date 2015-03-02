#include "nau/attributeValues.h"

#include "nau/slogger.h"


int AttributeValues::NextAttrib = 0;
// ----------------------------------------------
//		ENUM
// ----------------------------------------------

int 
AttributeValues::getPrope(EnumProperty prop) {

	return m_EnumProps[prop];
}


bool 
AttributeValues::isValide(EnumProperty prop, int value) {

	Attribute attr = m_Attribs->get(prop, Enums::ENUM);
	if (attr.getName() == "NO_ATTR")
		return false;
	if (attr.isValid(value)) 
		return true;
	else
		return false;
}


void 
AttributeValues::setPrope(EnumProperty prop, int value) {
		
	assert(isValide(prop,value));
	m_EnumProps[prop] = value;
}


// ----------------------------------------------
//		INT
// ----------------------------------------------


int 
AttributeValues::getPropi(IntProperty prop) {

	return m_IntProps[prop];
}


bool
AttributeValues::isValidi(IntProperty prop, int value) {

	Attribute attr = m_Attribs->get(prop, Enums::INT);
	if (attr.getName() == "NO_ATTR")
		return false;
	int *max, *min;
	if (attr.getRangeDefined()) {
		max = (int *)attr.getMax();
		min = (int *)attr.getMin();

		if (max != NULL && value > *max)
			return false;
		if (min != NULL && value < *min)
			return false;
	}
	return true;
}


void 
AttributeValues::setPropi(IntProperty prop, int value) {

	assert(isValidi(prop, value));
	m_IntProps[prop] = value;
}


// ----------------------------------------------
//		INT2
// ----------------------------------------------


ivec2 &
AttributeValues::getPropi2(Int2Property prop) {

	return m_Int2Props[prop];
}


bool
AttributeValues::isValidi2(Int2Property prop, ivec2 &value) {

	Attribute attr = m_Attribs->get(prop, Enums::IVEC2);
	if (attr.getName() == "NO_ATTR")
		return false;
	ivec2 *max, *min;
	if (attr.getRangeDefined()) {
		max = (ivec2 *)attr.getMax();
		min = (ivec2 *)attr.getMin();

		if (max != NULL && value > *max)
			return false;
		if (min != NULL && value < *min)
			return false;
	}
	return true;
}


void
AttributeValues::setPropi2(Int2Property prop, ivec2 &value) {

	assert(isValidi2(prop, value));
	m_Int2Props[prop] = value;
}

// ----------------------------------------------
//		UINT
// ----------------------------------------------


unsigned int 
AttributeValues::getPropui(UIntProperty prop) {

	return m_UIntProps[prop];
}


bool 
AttributeValues::isValidui(UIntProperty prop, unsigned int value) {

	Attribute attr = m_Attribs->get(prop, Enums::UINT);
	if (attr.getName() == "NO_ATTR")
		return false;

	unsigned int *max, *min;
	if (attr.getRangeDefined()) {
		max = (unsigned int *)attr.getMax();
		min = (unsigned int *)attr.getMin();

		if (max != NULL && value > *max)
			return false;
		if (min != NULL && value < *min)
			return false;
	}
	return true;
}


void 
AttributeValues::setPropui(UIntProperty prop, int unsigned value) {
	assert(isValidui(prop, value));
	m_UIntProps[prop] = value;
}


// ----------------------------------------------
//		UINT3
// ----------------------------------------------


uivec3 &
AttributeValues::getPropui3(UInt3Property prop) {

	return m_UInt3Props[prop];
}


bool
AttributeValues::isValidui3(UInt3Property prop, uivec3 &value) {

	Attribute attr = m_Attribs->get(prop, Enums::UIVEC3);
	if (attr.getName() == "NO_ATTR")
		return false;

	uivec3 *max, *min;
	if (attr.getRangeDefined()) {
		max = (uivec3 *)attr.getMax();
		min = (uivec3 *)attr.getMin();

		if (max != NULL && value > *max)
			return false;
		if (min != NULL && value < *min)
			return false;
	}
	return true;
}


void
AttributeValues::setPropui3(UInt3Property prop, uivec3 &value) {
	assert(isValidui3(prop, value));
	m_UInt3Props[prop] = value;
}


// ----------------------------------------------
//		BOOL
// ----------------------------------------------


bool 
AttributeValues::getPropb(BoolProperty prop) {

	return m_BoolProps[prop];
}


bool 
AttributeValues::isValidb(BoolProperty prop, bool value) {

	Attribute attr = m_Attribs->get(prop, Enums::BOOL);
	if (attr.getName() == "NO_ATTR")
		return false;
	else
		return true;
}


void 
AttributeValues::setPropb(BoolProperty prop, bool value) {
	assert(isValidb(prop, value));
	m_BoolProps[prop] = value;
}


// ----------------------------------------------
//		BOOL4
// ----------------------------------------------


bvec4 &
AttributeValues::getPropb4(Bool4Property prop) {

	return m_Bool4Props[prop];
}


bool 
AttributeValues::isValidb4(Bool4Property prop, bvec4 &value) {

	Attribute attr = m_Attribs->get(prop, Enums::BVEC4);
	if (attr.getName() == "NO_ATTR")
		return false;
	else
		return true;
}


void 
AttributeValues::setPropb4(Bool4Property prop, bvec4 &value) {
	assert(isValidb4(prop, value));
	m_Bool4Props[prop] = value;
}


// ----------------------------------------------
//		FLOAT
// ----------------------------------------------


float 
AttributeValues::getPropf(FloatProperty prop) {

	return m_FloatProps[prop];
}


bool 
AttributeValues::isValidf(FloatProperty prop, float f) {

	Attribute attr = m_Attribs->get(prop, Enums::FLOAT);
	if (attr.getName() == "NO_ATTR")
		return false;
	float *max, *min;
	if (attr.getRangeDefined()) {
		max = (float *)attr.getMax();
		min = (float *)attr.getMin();

		if (max != NULL && f > *max)
			return false;
		if (min != NULL && f < *min)
			return false;
	}
	return true;
}


void 
AttributeValues::setPropf(FloatProperty prop, float value) {
	assert(isValidf(prop, value));
	m_FloatProps[prop] = value;
}


// ----------------------------------------------
//		VEC4
// ----------------------------------------------


vec4 &
AttributeValues::getPropf4(Float4Property prop) {

	return m_Float4Props[prop];
}


bool 
AttributeValues::isValidf4(Float4Property prop, vec4 &f) {

	Attribute attr = m_Attribs->get(prop, Enums::VEC4);
	if (attr.getName() == "NO_ATTR")
		return false;
	vec4 *max, *min;
	if (attr.getRangeDefined()) {
		max = (vec4 *)attr.getMax();
		min = (vec4 *)attr.getMin();

		if (max != NULL && (f.x > max->x || f.y > max->y || f.z > max->z || f.w > max->w))
			return false;
		if (min != NULL && (f.x < min->x || f.y < min->y || f.z < min->z || f.w < min->w))
			return false;
	}
	return true;
}


void
AttributeValues::setPropf4(Float4Property prop, vec4 &value) {

	assert(isValidf4(prop, value));
	m_Float4Props[prop] = value;
}


void 
AttributeValues::setPropf4(Float4Property prop, float x, float y, float z, float w) {

	setPropf4(prop, vec4(x,y,z,w));

}


// ----------------------------------------------
//		VEC3
// ----------------------------------------------


vec3 &
AttributeValues::getPropf3(Float3Property prop) {

	return m_Float3Props[prop];
}


bool
AttributeValues::isValidf3(Float3Property prop, vec3 &f) {

	Attribute attr = m_Attribs->get(prop, Enums::VEC3);
	if (attr.getName() == "NO_ATTR")
		return false;
	vec3 *max, *min;
	if (attr.getRangeDefined()) {
		max = (vec3 *)attr.getMax();
		min = (vec3 *)attr.getMin();

		if (max != NULL && (f.x > max->x || f.y > max->y || f.z > max->z ))
			return false;
		if (min != NULL && (f.x < min->x || f.y < min->y || f.z < min->z ))
			return false;
	}
	return true;
}


void
AttributeValues::setPropf3(Float3Property prop, vec3 &value) {

	assert(isValidf3(prop, value));
	m_Float3Props[prop] = value;
}


void
AttributeValues::setPropf3(Float3Property prop, float x, float y, float z) {

	setPropf3(prop, vec3(x, y, z));

}

// ----------------------------------------------
//		VEC2
// ----------------------------------------------


vec2 &
AttributeValues::getPropf2(Float2Property prop) {

	return m_Float2Props[prop];
}


bool 
AttributeValues::isValidf2(Float2Property prop, vec2 &f) {

	Attribute attr = m_Attribs->get(prop, Enums::VEC2);
	if (attr.getName() == "NO_ATTR")
		return false;
	vec2 *max, *min;
	if (attr.getRangeDefined()) {
		max = (vec2 *)attr.getMax();
		min = (vec2 *)attr.getMin();

		if (max != NULL && (f.x > max->x || f.y > max->y ))
			return false;
		if (min != NULL && (f.x < min->x || f.y < min->y ))
			return false;
	}
	return true;
}


void 
AttributeValues::setPropf2(Float2Property prop, vec2 &value) {
	assert(isValidf2(prop, value));
	m_Float2Props[prop] = value;
}


// ----------------------------------------------
//		MAT4
// ----------------------------------------------


const mat4 &
AttributeValues::getPropm4(Mat4Property prop) {

	return m_Mat4Props[prop];
}


bool 
AttributeValues::isValidm4(Mat4Property prop, mat4 &value) {

	Attribute attr = m_Attribs->get(prop, Enums::MAT4);
	if (attr.getName() == "NO_ATTR")
		return false;
	else
		return true;
}


void 
AttributeValues::setPropm4(Mat4Property prop, mat4 &value) {
	assert(isValidm4(prop, value));
	m_Mat4Props[prop] = value;
}


// ----------------------------------------------
//		MAT3
// ----------------------------------------------


const mat3 &
AttributeValues::getPropm3(Mat3Property prop) {

	return m_Mat3Props[prop];
}


bool 
AttributeValues::isValidm3(Mat3Property prop, mat3 &value) {

	Attribute attr = m_Attribs->get(prop, Enums::MAT3);
	if (attr.getName() == "NO_ATTR")
		return false;
	else
		return true;
}


void 
AttributeValues::setPropm3(Mat3Property prop, mat3 &value) {
	assert(isValidm3(prop, value));
	m_Mat3Props[prop] = value;
}


// ----------------------------------------------
//		All
// ----------------------------------------------

const AttributeValues& 
AttributeValues::operator =(const AttributeValues &to) {

	if (this != &to) {
		m_EnumProps = to.m_EnumProps;
		m_IntProps = to.m_IntProps;
		m_Int2Props = to.m_Int2Props;
		m_UIntProps = to.m_UIntProps;
		m_UInt3Props = to.m_UInt3Props;
		m_BoolProps = to.m_BoolProps;
		m_Bool4Props = to.m_Bool4Props;
		m_FloatProps = to.m_FloatProps;
		m_Float2Props = to.m_Float2Props;
		m_Float3Props = to.m_Float3Props;
		m_Float4Props = to.m_Float4Props;
		m_Mat3Props = to.m_Mat3Props;
		m_Mat4Props = to.m_Mat4Props;
		m_Attribs = to.m_Attribs;
	}
	return *this;
}


void
AttributeValues::copy(AttributeValues *to) {

	//to->m_StringProps = m_StringProps;
	m_EnumProps = to->m_EnumProps;
	m_IntProps = to->m_IntProps;
	m_Int2Props = to->m_Int2Props;
	m_UIntProps = to->m_UIntProps;
	m_UInt3Props = to->m_UInt3Props;
	m_BoolProps = to->m_BoolProps;
	m_Bool4Props = to->m_Bool4Props;
	m_FloatProps = to->m_FloatProps;
	m_Float2Props = to->m_Float2Props;
	m_Float3Props = to->m_Float3Props;
	m_Float4Props = to->m_Float4Props;
	m_Mat3Props = to->m_Mat3Props;
	m_Mat4Props = to->m_Mat4Props;
	m_Attribs = to->m_Attribs;
}


void 
AttributeValues::clearArrays() {

	//to->m_StringProps = m_StringProps;
	m_EnumProps.clear();
	m_IntProps.clear();
	m_Int2Props.clear();
	m_UIntProps.clear();
	m_UInt3Props.clear();
	m_BoolProps.clear();
	m_Bool4Props.clear();
	m_FloatProps.clear();
	m_Float2Props.clear();
	m_Float3Props.clear();
	m_Float4Props.clear();
	m_Mat3Props.clear();
	m_Mat4Props.clear();
}


void *
AttributeValues::getProp(unsigned int prop, Enums::DataType type) {

	void *val;

	switch (type) {

		case Enums::ENUM:
			if (m_EnumProps.count(prop) == 0) {
				val = m_Attribs->getDefault(prop, type);
				if (val != NULL)
					m_EnumProps[prop] = *(int *)val;
				else { // life goes on ... except in debug mode
					assert(false && "Accessing undefined attribute of type ENUM");
					SLOG("Accessing undefined attribute of type ENUM - This should never occur");
					m_EnumProps[prop] = 0;
				}
			}
			return(&(m_EnumProps[prop]));
			break;
		case Enums::INT:
			if (m_IntProps.count(prop) == 0) {
				val = m_Attribs->getDefault(prop, type);
				if (val != NULL)
					m_IntProps[prop] = *(int *)val;
				else { // life goes on ... except in debug mode
					assert(false && "Accessing undefined attribute of type INT");
					SLOG("Accessing undefined attribute of type INT - This should never occur");
					m_IntProps[prop] = 0;
				}
			}
			return(&(m_IntProps[prop]));
			break;
		case Enums::IVEC2:
			if (m_Int2Props.count(prop) == 0) {
				val = m_Attribs->getDefault(prop, type);
				if (val != NULL)	
					m_Int2Props[prop] = *(ivec2 *)val;
				else { // life goes on ... except in debug mode
					assert(false && "Accessing undefined attribute of type IVEC2");
					SLOG("Accessing undefined attribute of type IVEC2 - This should never occur");
					m_Int2Props[prop] = 0;
				}
			}
			return(&(m_Int2Props[prop]));
			break;
		case Enums::UINT:
			if (m_UIntProps.count(prop) == 0) {
				val = m_Attribs->getDefault(prop, type);
				if (val != NULL)
					m_UIntProps[prop] = *(unsigned int *)val;
				else { // life goes on ... except in debug mode
					assert(false && "Accessing undefined attribute of type UINT");
					SLOG("Accessing undefined attribute of type UINT - This should never occur");
					m_UIntProps[prop] = 0;
				}
			}
			return(&(m_UIntProps[prop]));
			break;
		case Enums::UIVEC3:
			if (m_UInt3Props.count(prop) == 0) {
				val = m_Attribs->getDefault(prop, type);
				if (val != NULL)
					m_UInt3Props[prop] = *(uivec3 *)val;
				else { // life goes on ... except in debug mode
					assert(false && "Accessing undefined attribute of type UINT3");
					SLOG("Accessing undefined attribute of type UINT - This should never occur");
					m_UInt3Props[prop] = uivec3(0);
				}
			}
			return(&(m_UInt3Props[prop]));
			break;
		case Enums::BOOL:
			if (m_BoolProps.count(prop) == 0) {
				val = m_Attribs->getDefault(prop, type);
				if (val != NULL)
					m_BoolProps[prop] = *(bool *)val;
				else { // life goes on ... except in debug mode
					assert(false && "Accessing undefined attribute of type BOOL");
					SLOG("Accessing undefined attribute of type BOOL - This should never occur");
					m_BoolProps[prop] = 0;
				}
			}
			return(&(m_BoolProps[prop]));
			break;
		case Enums::BVEC4:
			if (m_Bool4Props.count(prop) == 0) {
				val = m_Attribs->getDefault(prop, type);
				if (val != NULL)
					m_Bool4Props[prop] = *(bvec4 *)val;
				else { // life goes on ... except in debug mode
					assert(false && "Accessing undefined attribute of type BVEC4");
					SLOG("Accessing undefined attribute of type BVEC4 - This should never occur");
					m_Bool4Props[prop] = 0;
				}
			}
			return(&(m_Bool4Props[prop]));
			break;
		case Enums::FLOAT:
			if (m_FloatProps.count(prop) == 0) {
				val = m_Attribs->getDefault(prop, type);
				if (val != NULL)
					m_FloatProps[prop] = *(float *)val;
				else { // life goes on ... except in debug mode
					assert(false && "Accessing undefined user attribute of type FLOAT");
					SLOG("Accessing undefined user attribute of type FLOAT - This should never occur");
					m_FloatProps[prop] = 0;
				}
			}
			return(&(m_FloatProps[prop]));
			break;
		case Enums::VEC4:
			if (m_Float4Props.count(prop) == 0) {
				val = m_Attribs->getDefault(prop, type);
				if (val != NULL)
					m_Float4Props[prop] = *(vec4 *)val;
				else { // life goes on ... except in debug mode
					assert(false && "Accessing undefined user attribute of type VEC4");
					SLOG("Accessing undefined user attribute of type VEC4 - This should never occur");
					m_Float4Props[prop] = vec4(0.0f);
				}
			}
			return(&(m_Float4Props[prop]));
			break;
		case Enums::VEC3:
			if (m_Float3Props.count(prop) == 0) {
				val = m_Attribs->getDefault(prop, type);
				if (val != NULL)
					m_Float3Props[prop] = *(vec3 *)val;
				else { // life goes on ... except in debug mode
					assert(false && "Accessing undefined user attribute of type VEC3");
					SLOG("Accessing undefined user attribute of type VEC3 - This should never occur");
					m_Float3Props[prop] = vec3(0.0f);
				}
			}
			return(&(m_Float3Props[prop]));
			break;
		case Enums::VEC2:
			if (m_Float2Props.count(prop) == 0) {
				val = m_Attribs->getDefault(prop, type);
				if (val != NULL)
					m_Float2Props[prop] = *(vec2 *)val;
				else { // life goes on ... except in debug mode
					assert(false && "Accessing undefined user attribute of type VEC2");
					SLOG("Accessing undefined user attribute of type VEC2 - This should never occur");
					m_Float2Props[prop] = vec2(0.0f);
				}
			}
			return(&(m_Float2Props[prop]));
			break;
		case Enums::MAT4:
			if (m_Mat4Props.count(prop) == 0) {
				val = m_Attribs->getDefault(prop, type);
				if (val != NULL)
					m_Mat4Props[prop] = *(mat4 *)val;
				else { // life goes on ... except in debug mode
					assert(false && "Accessing undefined user attribute of type MAT4");
					SLOG("Accessing undefined user attribute of type MAT4 - This should never occur");
					m_Mat4Props[prop] = mat4();
				}
			}
			return(&(m_Mat4Props[prop]));
			break;
		case Enums::MAT3:
			if (m_Mat3Props.count(prop) == 0) {
				val = m_Attribs->getDefault(prop, type);
				if (val != NULL)
					m_Mat3Props[prop] = *(mat3 *)val;
				else { // life goes on ... except in debug mode
					assert(false && "Accessing undefined user attribute of type MAT3");
					SLOG("Accessing undefined user attribute of type MAT3 - This should never occur");
					m_Mat3Props[prop] = mat3();
				}
			}
			return(&(m_Mat3Props[prop]));
			break;
		default:
			assert(false && "Missing Data Type in class attributeValues");
			return NULL;
	}
};


void 
AttributeValues::setProp(unsigned int prop, Enums::DataType type, void *value) {

	switch (type) {

	case Enums::ENUM:
		setPrope((EnumProperty)prop, *(int *)value);
		break;
	case Enums::INT:
		setPropi((IntProperty)prop, *(int *)value);
		break;
	case Enums::IVEC2:
		setPropi2((Int2Property)prop, *(ivec2 *)value);
		break;
	case Enums::UINT:
		setPropui((UIntProperty)prop, *(unsigned int *)value);
		break;
	case Enums::UIVEC3:
		setPropui3((UInt3Property)prop, *(uivec3 *)value);
		break;
	case Enums::BOOL:
		setPropb((BoolProperty)prop, *(bool *)value);
		break;
	case Enums::BVEC4:
		setPropb4((Bool4Property)prop, *(bvec4 *)value);
		break;
	case Enums::FLOAT:
		setPropf((FloatProperty)prop, *(float *)value);
		break;
	case Enums::VEC2:
		setPropf2((Float2Property)prop, *(vec2 *)value);
		break;
	case Enums::VEC3:
		setPropf3((Float3Property)prop, *(vec3 *)value);
		break;
	case Enums::VEC4:
		setPropf4((Float4Property)prop, *(vec4 *)value);
		break;
	case Enums::MAT4:
		setPropm4((Mat4Property)prop, *(mat4 *)value);
		break;
	case Enums::MAT3:
		setPropm3((Mat3Property)prop, *(mat3 *)value);
		break;
	default:
		assert(false && "Missing Data Type in class attributeValues or Invalid prop");
	}	
}


bool
AttributeValues::isValid(unsigned int prop, Enums::DataType type, void *value) {

	switch (type) {

	case Enums::ENUM:
		return isValide((EnumProperty)prop, *(int *)value);
		break;
	case Enums::INT:
		return isValidi((IntProperty)prop, *(int *)value);
		break;
	case Enums::IVEC2:
		return isValidi2((Int2Property)prop, *(ivec2 *)value);
		break;
	case Enums::UINT:
		return isValidui((UIntProperty)prop, *(unsigned int *)value);
		break;
	case Enums::UIVEC3:
		return isValidui3((UInt3Property)prop, *(uivec3 *)value);
		break;
	case Enums::BOOL:
		return isValidb((BoolProperty)prop, *(bool *)value);
		break;
	case Enums::BVEC4:
		return isValidb4((Bool4Property)prop, *(bvec4 *)value);
		break;
	case Enums::FLOAT:
		return isValidf((FloatProperty)prop, *(float *)value);
		break;
	case Enums::VEC2:
		return isValidf2((Float2Property)prop, *(vec2 *)value);
		break;
	case Enums::VEC3:
		return isValidf3((Float3Property)prop, *(vec3 *)value);
		break;
	case Enums::VEC4:
		return isValidf4((Float4Property)prop, *(vec4 *)value);
		break;
	case Enums::MAT4:
		return isValidm4((Mat4Property)prop, *(mat4 *)value);
		break;
	case Enums::MAT3:
		return isValidm3((Mat3Property)prop, *(mat3 *)value);
		break;
	default:
		assert(false && "Missing Data Type in class attributeValues or Invalid prop");
		return false;
	}
}


void 
AttributeValues::registerAndInitArrays(std::string name, AttribSet &attribs) {

	initArrays(attribs);
}


void
AttributeValues::initArrays(AttribSet &attribs) {

	//attribs.initAttribInstanceEnumArray(m_EnumProps);
	//attribs.initAttribInstanceIntArray(m_IntProps);
	//attribs.initAttribInstanceUIntArray(m_UIntProps);
	//attribs.initAttribInstanceUInt3Array(m_UInt3Props);
	//attribs.initAttribInstanceBoolArray(m_BoolProps);
	//attribs.initAttribInstanceBvec4Array(m_Bool4Props);
	//attribs.initAttribInstanceVec4Array(m_Float4Props);
	//attribs.initAttribInstanceVec3Array(m_Float3Props);
	//attribs.initAttribInstanceVec2Array(m_Float2Props);
	//attribs.initAttribInstanceFloatArray(m_FloatProps);
	//attribs.initAttribInstanceMat4Array(m_Mat4Props);
	//attribs.initAttribInstanceMat3Array(m_Mat3Props);

	m_Attribs = &attribs;
	initArrays();
}


void
AttributeValues::initArrays() {

	m_Attribs->initAttribInstanceEnumArray(m_EnumProps);
	m_Attribs->initAttribInstanceIntArray(m_IntProps);
	m_Attribs->initAttribInstanceInt2Array(m_Int2Props);
	m_Attribs->initAttribInstanceUIntArray(m_UIntProps);
	m_Attribs->initAttribInstanceUInt3Array(m_UInt3Props);
	m_Attribs->initAttribInstanceBoolArray(m_BoolProps);
	m_Attribs->initAttribInstanceBvec4Array(m_Bool4Props);
	m_Attribs->initAttribInstanceVec4Array(m_Float4Props);
	m_Attribs->initAttribInstanceVec3Array(m_Float3Props);
	m_Attribs->initAttribInstanceVec2Array(m_Float2Props);
	m_Attribs->initAttribInstanceFloatArray(m_FloatProps);
	m_Attribs->initAttribInstanceMat4Array(m_Mat4Props);
	m_Attribs->initAttribInstanceMat3Array(m_Mat3Props);
}


AttributeValues::AttributeValues() {

}


AttributeValues::AttributeValues(const AttributeValues &to) {

	m_EnumProps = to.m_EnumProps;
	m_IntProps = to.m_IntProps;
	m_Int2Props = to.m_Int2Props;
	m_UIntProps = to.m_UIntProps;
	m_UInt3Props = to.m_UInt3Props;
	m_BoolProps = to.m_BoolProps;
	m_Bool4Props = to.m_Bool4Props;
	m_FloatProps = to.m_FloatProps;
	m_Float2Props = to.m_Float2Props;
	m_Float3Props = to.m_Float3Props;
	m_Float4Props = to.m_Float4Props;
	m_Mat3Props = to.m_Mat3Props;
	m_Mat4Props = to.m_Mat4Props;
	m_Attribs = to.m_Attribs;
}


AttributeValues::~AttributeValues() {

}



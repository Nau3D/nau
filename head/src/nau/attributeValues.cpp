#include <nau/attributeValues.h>

// ----------------------------------------------
//		ENUM
// ----------------------------------------------

int 
AttributeValues::getPrope(EnumProperty prop) {

	return m_EnumProps[prop];
}


bool 
AttributeValues::isValide(EnumProperty prop, int value) {

	Attribute attr = m_Attribs.get(prop, Enums::ENUM);
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

	Attribute attr = m_Attribs.get(prop, Enums::INT);
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
//		UINT
// ----------------------------------------------


unsigned int 
AttributeValues::getPropui(UIntProperty prop) {

	return m_UIntProps[prop];
}


bool 
AttributeValues::isValidui(UIntProperty prop, unsigned int value) {

	Attribute attr = m_Attribs.get(prop, Enums::UINT);
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
//		BOOL
// ----------------------------------------------


bool 
AttributeValues::getPropb(BoolProperty prop) {

	return m_BoolProps[prop];
}


bool 
AttributeValues::isValidb(BoolProperty prop, bool value) {

	Attribute attr = m_Attribs.get(prop, Enums::BOOL);
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
//		BOOL
// ----------------------------------------------


bvec4 &
AttributeValues::getPropb4(Bool4Property prop) {

	return m_Bool4Props[prop];
}


bool 
AttributeValues::isValidb4(Bool4Property prop, bvec4 &value) {

	Attribute attr = m_Attribs.get(prop, Enums::BVEC4);
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
//		BOOL
// ----------------------------------------------


float 
AttributeValues::getPropf(FloatProperty prop) {

	return m_FloatProps[prop];
}


bool 
AttributeValues::isValidf(FloatProperty prop, float f) {

	Attribute attr = m_Attribs.get(prop, Enums::FLOAT);
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

	Attribute attr = m_Attribs.get(prop, Enums::VEC4);
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

	vec4 *v = new vec4(x, y, z, w);
	setPropf4(prop, *v);

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

	Attribute attr = m_Attribs.get(prop, Enums::VEC2);
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

	Attribute attr = m_Attribs.get(prop, Enums::MAT4);
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

	Attribute attr = m_Attribs.get(prop, Enums::MAT3);
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


void
AttributeValues::copy(AttributeValues *to) {

	//to->m_StringProps = m_StringProps;
	to->m_EnumProps = m_EnumProps;
	to->m_IntProps =    m_IntProps;
	to->m_UIntProps =   m_UIntProps;
	to->m_BoolProps =   m_BoolProps;
	to->m_Bool4Props =  m_Bool4Props;
	to->m_FloatProps =  m_FloatProps;
	to->m_Float2Props = m_Float2Props;
	to->m_Float4Props = m_Float4Props;
	to->m_Mat3Props =   m_Mat3Props;
	to->m_Mat4Props =   m_Mat4Props;
}


void 
AttributeValues::clearArrays() {

	//to->m_StringProps = m_StringProps;
	m_EnumProps.clear();
	m_IntProps.clear();
	m_UIntProps.clear();
	m_BoolProps.clear();
	m_Bool4Props.clear();
	m_FloatProps.clear();
	m_Float2Props.clear();
	m_Float4Props.clear();
	m_Mat3Props.clear();
	m_Mat4Props.clear();
}


void *
AttributeValues::getProp(int prop, Enums::DataType type) {

	int c;
	switch (type) {

		case Enums::ENUM:
			c = m_EnumProps.count(prop);
			if (prop < AttribSet::USER_ATTRIBS) 
				assert(c > 0);
			else {
				if (!c)
					m_EnumProps[prop] = *(int *)m_Attribs.getDefault(prop, type);
			}
			return(&(m_EnumProps[prop]));
			break;
		case Enums::INT:
			assert(m_IntProps.count(prop) > 0);
			return(&(m_IntProps[prop]));
			break;
		case Enums::UINT:
			assert(m_UIntProps.count(prop) > 0);
			return(&(m_UIntProps[prop]));
			break;
		case Enums::BOOL:
			assert(m_BoolProps.count(prop) > 0);
			return(&(m_BoolProps[prop]));
			break;
		case Enums::BVEC4:
			assert(m_Bool4Props.count(prop) > 0);
			return(&(m_Bool4Props[prop]));
			break;
		case Enums::FLOAT:
			assert(m_FloatProps.count(prop) > 0);
			return(&(m_FloatProps[prop]));
			break;
		case Enums::VEC4:
			assert(m_Float4Props.count(prop) > 0);
			return(&(m_Float4Props[prop]));
			break;
		case Enums::MAT4:
			assert(m_Mat4Props.count(prop) > 0);
			return(&(m_Mat4Props[prop]));
			break;
		case Enums::MAT3:
			assert(m_Mat3Props.count(prop) > 0);
			return(&(m_Mat3Props[prop]));
			break;
		default:
			assert(false && "Missing Data Type in class attributeValues");
			return NULL;
	}
};


void 
AttributeValues::setProp(int prop, Enums::DataType type, void *value) {

	switch (type) {

	case Enums::ENUM:
		setPrope((EnumProperty)prop, *(int *)value);
		break;
	case Enums::INT:
		setPropi((IntProperty)prop, *(int *)value);
		break;
	case Enums::UINT:
		setPropui((UIntProperty)prop, *(unsigned int *)value);
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


void 
AttributeValues::registerAndInitArrays(std::string name, AttribSet &attribs) {

	initArrays(attribs);
}


void
AttributeValues::initArrays(AttribSet &attribs) {

	attribs.initAttribInstanceEnumArray(m_EnumProps);
	attribs.initAttribInstanceIntArray(m_IntProps);
	attribs.initAttribInstanceUIntArray(m_UIntProps);
	attribs.initAttribInstanceBoolArray(m_BoolProps);
	attribs.initAttribInstanceBvec4Array(m_Bool4Props);
	attribs.initAttribInstanceVec4Array(m_Float4Props);
	attribs.initAttribInstanceVec2Array(m_Float2Props);
	attribs.initAttribInstanceFloatArray(m_FloatProps);
	attribs.initAttribInstanceMat4Array(m_Mat4Props);
	attribs.initAttribInstanceMat3Array(m_Mat3Props);

	m_Attribs = attribs;
}


AttributeValues::AttributeValues() {

}


AttributeValues::~AttributeValues() {

}



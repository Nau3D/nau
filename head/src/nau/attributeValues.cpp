#include "nau/attributeValues.h"

#include "nau.h"
#include "nau/math/number.h"
#include "nau/slogger.h"


int AttributeValues::NextAttrib = 0;

// ----------------------------------------------
//		STRING
// ----------------------------------------------

const std::string & nau::AttributeValues::getProps(StringProperty prop) {

	if (m_StringProps.count(prop))
		return m_StringProps[prop];
	else
		return m_DummyString;
}

void 
AttributeValues::setProps(StringProperty prop, std::string &value) {

	if (isValids(prop, value))
		m_StringProps[prop] = value;
}

bool 
AttributeValues::isValids(StringProperty prop, std::string value) {

	//if (!m_StringProps.count(prop))
	//	return false;

	const std::string &name = m_Attribs->getName(prop, Enums::STRING);
	const std::string &context = m_Attribs->get(name)->getObjType();
	return NAU->validateObjectName(context, value);
}


// ----------------------------------------------
//		ENUM
// ----------------------------------------------


int
AttributeValues::getPrope(EnumProperty prop) {

	// if not a user attrib
	if (prop < AttribSet::USER_ATTRIBS)
		return m_EnumProps[prop];
	// if prop is a user attrib and has already been set
	else if (m_EnumProps.count(prop))
		return m_EnumProps[prop];
	else {
		std::shared_ptr<Data> &d = m_Attribs->get(prop, Enums::ENUM)->getDefault();
		std::shared_ptr<NauInt> &ni = std::dynamic_pointer_cast<NauInt>(d);
		m_EnumProps[prop] = ni->getNumber();
		return m_EnumProps[prop];
	}
}


bool 
AttributeValues::isValide(EnumProperty prop, int value) {

	std::unique_ptr<Attribute> &attr = m_Attribs->get(prop, Enums::ENUM);
	if (attr->getName() == "NO_ATTR")
		return false;
	if (!APISupport->apiSupport(attr->getRequirement()))
		return false;
	if (attr->isValid(value)) 
		return true;
	else
		return false;
}


void 
AttributeValues::setPrope(EnumProperty prop, int value) {
		
	assert(isValide(prop, value));
	m_EnumProps[prop] = value;
}


// ----------------------------------------------
//		INT
// ----------------------------------------------


int 
AttributeValues::getPropi(IntProperty prop) {

	// if not a user attrib
	if (prop < AttribSet::USER_ATTRIBS)
		return m_IntProps[prop];
	// if prop is a user attrib and has already been set
	else if (m_IntProps.count(prop))
		return m_IntProps[prop];
	else {
		std::shared_ptr<Data> &d = m_Attribs->get(prop, Enums::INT)->getDefault();
		std::shared_ptr<NauInt> &ni = std::dynamic_pointer_cast<NauInt>(d);
		m_IntProps[prop] = ni->getNumber();
		return m_IntProps[prop];
	}
}


bool
AttributeValues::isValidi(IntProperty prop, int value) {

	std::unique_ptr<Attribute> &attr = m_Attribs->get(prop, Enums::INT);
	if (attr->getName() == "NO_ATTR")
		return false;
	if (!APISupport->apiSupport(attr->getRequirement()))
		return false;

	if (attr->getRangeDefined()) {
		std::shared_ptr<NauInt> &max = std::dynamic_pointer_cast<NauInt>(attr->getMax());
		std::shared_ptr<NauInt> &min = std::dynamic_pointer_cast<NauInt>(attr->getMin());

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

	// if not a user attrib
	if (prop < AttribSet::USER_ATTRIBS)
		return m_Int2Props[prop];
	// if prop is a user attrib and has already been set
	else if (m_Int2Props.count(prop))
		return m_Int2Props[prop];
	else {
		std::shared_ptr<Data> &d = m_Attribs->get(prop, Enums::IVEC2)->getDefault();
		std::shared_ptr<ivec2> &ni = std::dynamic_pointer_cast<ivec2>(d);
		m_Int2Props[prop] = *ni;
		return m_Int2Props[prop];
	}
}


bool
AttributeValues::isValidi2(Int2Property prop, ivec2 &value) {

	std::unique_ptr<Attribute> &attr = m_Attribs->get(prop, Enums::IVEC2);
	if (attr->getName() == "NO_ATTR")
		return false;
	if (!APISupport->apiSupport(attr->getRequirement()))
		return false;

	if (attr->getRangeDefined()) {
		ivec2 &max = *(std::dynamic_pointer_cast<ivec2>(attr->getMax()));
		ivec2 &min = *(std::dynamic_pointer_cast<ivec2>(attr->getMin()));

		if (max != NULL && value > max)
			return false;
		if (min != NULL && value < min)
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
//		INT3
// ----------------------------------------------


ivec3 &
AttributeValues::getPropi3(Int3Property prop) {

	// if not a user attrib
	if (prop < AttribSet::USER_ATTRIBS)
		return m_Int3Props[prop];
	// if prop is a user attrib and has already been set
	else if (m_Int3Props.count(prop))
		return m_Int3Props[prop];
	else {
		std::shared_ptr<Data> &d = m_Attribs->get(prop, Enums::IVEC3)->getDefault();
		std::shared_ptr<ivec3> &ni = std::dynamic_pointer_cast<ivec3>(d);
		m_Int3Props[prop] = *ni;
		return m_Int3Props[prop];
	}
}


bool
AttributeValues::isValidi3(Int3Property prop, ivec3 &value) {

	std::unique_ptr<Attribute> &attr = m_Attribs->get(prop, Enums::IVEC3);
	if (attr->getName() == "NO_ATTR")
		return false;
	if (!APISupport->apiSupport(attr->getRequirement()))
		return false;

	if (attr->getRangeDefined()) {
		ivec3 &max = *(std::dynamic_pointer_cast<ivec3>(attr->getMax()));
		ivec3 &min = *(std::dynamic_pointer_cast<ivec3>(attr->getMin()));

		if (max != NULL && value > max)
			return false;
		if (min != NULL && value < min)
			return false;
	}
	return true;
}


void
AttributeValues::setPropi3(Int3Property prop, ivec3 &value) {

	assert(isValidi3(prop, value));
	m_Int3Props[prop] = value;
}

// ----------------------------------------------
//		UINT
// ----------------------------------------------


unsigned int 
AttributeValues::getPropui(UIntProperty prop) {

	// if not a user attrib
	if (prop < AttribSet::USER_ATTRIBS)
		return m_UIntProps[prop];
	// if prop is a user attrib and has already been set
	else if (m_UIntProps.count(prop))
		return m_UIntProps[prop];
	else {
		std::shared_ptr<Data> &d = m_Attribs->get(prop, Enums::UINT)->getDefault();
		std::shared_ptr<NauUInt> &ni = std::dynamic_pointer_cast<NauUInt>(d);
		m_UIntProps[prop] = ni->getNumber();
		return m_UIntProps[prop];
	}
}


bool 
AttributeValues::isValidui(UIntProperty prop, unsigned int value) {

	std::unique_ptr<Attribute> &attr = m_Attribs->get(prop, Enums::UINT);
	if (attr->getName() == "NO_ATTR")
		return false;
	if (!APISupport->apiSupport(attr->getRequirement()))
		return false;

	if (attr->getRangeDefined()) {
		std::shared_ptr<NauUInt> &max = std::dynamic_pointer_cast<NauUInt>(attr->getMax());
		std::shared_ptr<NauUInt> &min = std::dynamic_pointer_cast<NauUInt>(attr->getMin());

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
//		UINT2
// ----------------------------------------------


uivec2 &
AttributeValues::getPropui2(UInt2Property prop) {

	// if not a user attrib
	if (prop < AttribSet::USER_ATTRIBS)
		return m_UInt2Props[prop];
	// if prop is a user attrib and has already been set
	else if (m_UInt2Props.count(prop))
		return m_UInt2Props[prop];
	else {
		std::shared_ptr<Data> &d = m_Attribs->get(prop, Enums::UIVEC2)->getDefault();
		std::shared_ptr<uivec2> &ni = std::dynamic_pointer_cast<uivec2>(d);
		m_UInt2Props[prop] = *ni;
		return m_UInt2Props[prop];
	}
}


bool
AttributeValues::isValidui2(UInt2Property prop, uivec2 &value) {

	std::unique_ptr<Attribute> &attr = m_Attribs->get(prop, Enums::UIVEC2);
	if (attr->getName() == "NO_ATTR")
		return false;
	if (!APISupport->apiSupport(attr->getRequirement()))
		return false;

	if (attr->getRangeDefined()) {
		std::shared_ptr<uivec2> &max = std::dynamic_pointer_cast<uivec2>(attr->getMax());
		std::shared_ptr<uivec2> &min = std::dynamic_pointer_cast<uivec2>(attr->getMin());

		if (max != NULL && value > *max)
			return false;
		if (min != NULL && value < *min)
			return false;
	}
	return true;
}


void
AttributeValues::setPropui2(UInt2Property prop, uivec2 &value) {
	assert(isValidui2(prop, value));
	m_UInt2Props[prop] = value;
}


// ----------------------------------------------
//		UINT3
// ----------------------------------------------


uivec3 &
AttributeValues::getPropui3(UInt3Property prop) {

	// if not a user attrib
	if (prop < AttribSet::USER_ATTRIBS)
		return m_UInt3Props[prop];
	// if prop is a user attrib and has already been set
	else if (m_UInt3Props.count(prop))
		return m_UInt3Props[prop];
	else {
		std::shared_ptr<Data> &d = m_Attribs->get(prop, Enums::UIVEC3)->getDefault();
		std::shared_ptr<uivec3> &ni = std::dynamic_pointer_cast<uivec3>(d);
		m_UInt3Props[prop] = *ni;
		return m_UInt3Props[prop];
	}
}


bool
AttributeValues::isValidui3(UInt3Property prop, uivec3 &value) {

	std::unique_ptr<Attribute> &attr = m_Attribs->get(prop, Enums::UIVEC3);
	if (attr->getName() == "NO_ATTR")
		return false;
	if (!APISupport->apiSupport(attr->getRequirement()))
		return false;

	if (attr->getRangeDefined()) {
		std::shared_ptr<uivec3> &max = std::dynamic_pointer_cast<uivec3>(attr->getMax());
		std::shared_ptr<uivec3> &min = std::dynamic_pointer_cast<uivec3>(attr->getMin());

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

	// if not a user attrib
	if (prop < AttribSet::USER_ATTRIBS)
		return m_BoolProps[prop];
	// if prop is a user attrib and has already been set
	else if (m_BoolProps.count(prop))
		return m_BoolProps[prop];
	else {
		std::shared_ptr<Data> &d = m_Attribs->get(prop, Enums::BOOL)->getDefault();
		std::shared_ptr<NauInt> &ni = std::dynamic_pointer_cast<NauInt>(d);
		m_BoolProps[prop] = (ni->getNumber() != 0);
		return m_BoolProps[prop];
	}
}


bool 
AttributeValues::isValidb(BoolProperty prop, bool value) {

	std::unique_ptr<Attribute> &attr = m_Attribs->get(prop, Enums::BOOL);
	if (attr->getName() == "NO_ATTR")
		return false;
	if (!APISupport->apiSupport(attr->getRequirement()))
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

	// if not a user attrib
	if (prop < AttribSet::USER_ATTRIBS)
		return m_Bool4Props[prop];
	// if prop is a user attrib and has already been set
	else if (m_Bool4Props.count(prop))
		return m_Bool4Props[prop];
	else {
		std::shared_ptr<Data> &d = m_Attribs->get(prop, Enums::BVEC4)->getDefault();
		std::shared_ptr<bvec4> &ni = std::dynamic_pointer_cast<bvec4>(d);
		m_Bool4Props[prop] = *ni;
		return m_Bool4Props[prop];
	}
}


bool 
AttributeValues::isValidb4(Bool4Property prop, bvec4 &value) {

	std::unique_ptr<Attribute> &attr = m_Attribs->get(prop, Enums::BVEC4);
	if (attr->getName() == "NO_ATTR")
		return false;
	if (!APISupport->apiSupport(attr->getRequirement()))
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

	// if not a user attrib
	if (prop < AttribSet::USER_ATTRIBS)
		return m_FloatProps[prop];
	// if prop is a user attrib and has already been set
	else if (m_FloatProps.count(prop))
		return m_FloatProps[prop];
	else {
		std::shared_ptr<Data> &d = m_Attribs->get(prop, Enums::FLOAT)->getDefault();
		std::shared_ptr<NauFloat> &ni = std::dynamic_pointer_cast<NauFloat>(d);
		m_FloatProps[prop] = ni->getNumber();
		return m_FloatProps[prop];
	}
	return m_FloatProps[prop];
}


bool 
AttributeValues::isValidf(FloatProperty prop, float f) {

	std::unique_ptr<Attribute> &attr = m_Attribs->get(prop, Enums::FLOAT);
	if (attr->getName() == "NO_ATTR")
		return false;
	if (!APISupport->apiSupport(attr->getRequirement()))
		return false;

	if (attr->getRangeDefined()) {
		std::shared_ptr<NauFloat> &max = std::dynamic_pointer_cast<NauFloat>(attr->getMax());
		std::shared_ptr<NauFloat> &min = std::dynamic_pointer_cast<NauFloat>(attr->getMin());

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

	// if not a user attrib
	if (prop < AttribSet::USER_ATTRIBS)
		return m_Float4Props[prop];
	// if prop is a user attrib and has already been set
	else if (m_Float4Props.count(prop))
		return m_Float4Props[prop];
	else {
		std::shared_ptr<Data> &d = m_Attribs->get(prop, Enums::VEC4)->getDefault();
		std::shared_ptr<vec4> &ni = std::dynamic_pointer_cast<vec4>(d);
		m_Float4Props[prop] = *ni;
		return m_Float4Props[prop];
	}
}


bool 
AttributeValues::isValidf4(Float4Property prop, vec4 &f) {

	std::unique_ptr<Attribute> &attr = m_Attribs->get(prop, Enums::VEC4);
	if (attr->getName() == "NO_ATTR")
		return false;
	if (!APISupport->apiSupport(attr->getRequirement()))
		return false;

	if (attr->getRangeDefined()) {
		std::shared_ptr<vec4> &max = std::dynamic_pointer_cast<vec4>(attr->getMax());
		std::shared_ptr<vec4> &min = std::dynamic_pointer_cast<vec4>(attr->getMin());

		if (max != NULL && f > *max)
			return false;
		if (min != NULL && f < *min)
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

	// if not a user attrib
	if (prop < AttribSet::USER_ATTRIBS)
		return m_Float3Props[prop];
	// if prop is a user attrib and has already been set
	else if (m_Float3Props.count(prop))
		return m_Float3Props[prop];
	else {
		std::shared_ptr<Data> &d = m_Attribs->get(prop, Enums::VEC3)->getDefault();
		std::shared_ptr<vec3> &ni = std::dynamic_pointer_cast<vec3>(d);
		m_Float3Props[prop] = *ni;
		return m_Float3Props[prop];
	}
}


bool
AttributeValues::isValidf3(Float3Property prop, vec3 &f) {

	std::unique_ptr<Attribute> &attr = m_Attribs->get(prop, Enums::VEC3);
	if (attr->getName() == "NO_ATTR")
		return false;
	if (!APISupport->apiSupport(attr->getRequirement()))
		return false;

	if (attr->getRangeDefined()) {
		std::shared_ptr<vec3> &max = std::dynamic_pointer_cast<vec3>(attr->getMax());
		std::shared_ptr<vec3> &min = std::dynamic_pointer_cast<vec3>(attr->getMin());

		if (max != NULL && f > *max)
			return false;
		if (min != NULL && f < *min)
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

	// if not a user attrib
	if (prop < AttribSet::USER_ATTRIBS)
		return m_Float2Props[prop];
	// if prop is a user attrib and has already been set
	else if (m_Float2Props.count(prop))
		return m_Float2Props[prop];
	else {
		std::shared_ptr<Data> &d = m_Attribs->get(prop, Enums::VEC2)->getDefault();
		std::shared_ptr<vec2> &ni = std::dynamic_pointer_cast<vec2>(d);
		m_Float2Props[prop] = *ni;
		return m_Float2Props[prop];
	}
}


bool 
AttributeValues::isValidf2(Float2Property prop, vec2 &f) {

	std::unique_ptr<Attribute> &attr = m_Attribs->get(prop, Enums::VEC2);
	if (attr->getName() == "NO_ATTR")
		return false;
	if (!APISupport->apiSupport(attr->getRequirement()))
		return false;

	if (attr->getRangeDefined()) {
		std::shared_ptr<vec2> &max = std::dynamic_pointer_cast<vec2>(attr->getMax());
		std::shared_ptr<vec2> &min = std::dynamic_pointer_cast<vec2>(attr->getMin());

		if (max != NULL && f > *max)
			return false;
		if (min != NULL && f < *min)
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

	// if not a user attrib
	if (prop < AttribSet::USER_ATTRIBS)
		return m_Mat4Props[prop];
	// if prop is a user attrib and has already been set
	else if (m_Mat4Props.count(prop))
		return m_Mat4Props[prop];
	else {
		std::shared_ptr<Data> &d = m_Attribs->get(prop, Enums::MAT4)->getDefault();
		std::shared_ptr<mat4> &ni = std::dynamic_pointer_cast<mat4>(d);
		m_Mat4Props[prop] = *ni;
		return m_Mat4Props[prop];
	}
}


bool 
AttributeValues::isValidm4(Mat4Property prop, mat4 &value) {

	std::unique_ptr<Attribute> &attr = m_Attribs->get(prop, Enums::MAT4);
	if (attr->getName() == "NO_ATTR")
		return false;
	if (!APISupport->apiSupport(attr->getRequirement()))
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

	// if not a user attrib
	if (prop < AttribSet::USER_ATTRIBS)
		return m_Mat3Props[prop];
	// if prop is a user attrib and has already been set
	else if (m_Mat3Props.count(prop))
		return m_Mat3Props[prop];
	else {
		std::unique_ptr<Attribute> &a = m_Attribs->get(prop, Enums::MAT3);
		if (a->getId() == -1) {
			assert(false && "Attribute Values getProp - invalid MAT3 prop");
			SLOG("Attribute MAT3 with id %d does not exist", prop);
			Data *m = Enums::getDefaultValue(Enums::MAT3);
			m_Mat3Props[prop] = *(mat3 *)m;
			delete m;
		}
		else {
			std::shared_ptr<Data> &d = a->getDefault();
			std::shared_ptr<mat3> &ni = std::dynamic_pointer_cast<mat3>(d);
			m_Mat3Props[prop] = *ni;
		}
		return m_Mat3Props[prop];
	}
}



bool 
AttributeValues::isValidm3(Mat3Property prop, mat3 &value) {

	std::unique_ptr<Attribute> &attr = m_Attribs->get(prop, Enums::MAT3);
	if (attr->getName() == "NO_ATTR")
		return false;
	if (!APISupport->apiSupport(attr->getRequirement()))
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
		m_UInt2Props = to.m_UInt2Props;
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
	m_UInt2Props = to->m_UInt2Props;
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
	m_UInt2Props.clear();
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

	std::unique_ptr<Attribute> &attr = m_Attribs->get(prop, type);
	
	// if attrib does not exist
	if (attr->getId() == -1) {
		assert(false && "Accessing undefined attribute of type ENUM");
		SLOG("Accessing undefined attribute of type ENUM - This should never occur");
		return NULL;

	}

	switch (type) {

		case Enums::ENUM:
		//	if (m_EnumProps.count(prop) == 0) {
				getPrope((AttributeValues::EnumProperty)prop);
				//std::shared_ptr<Data> &d = m_Attribs->get(prop, Enums::INT)->getDefault();
				//std::shared_ptr<NauInt> ni = std::dynamic_pointer_cast<NauInt>(d);

				//std::shared_ptr<Data> &val = attr->getDefault();
				//m_EnumProps[prop] = std::dynamic_pointer_cast<NauInt>(val)->getNumber();
		//	}
			return(&(m_EnumProps[prop]));
			break;
		case Enums::INT:
		//	if (m_IntProps.count(prop) == 0) {
				getPropi((AttributeValues::IntProperty)prop);
				//val = m_Attribs->getDefault(prop, type);
				//if (val != NULL)
				//	m_IntProps[prop] = *(int *)val;
				//else { // life goes on ... except in debug mode
				//	assert(false && "Accessing undefined attribute of type INT");
				//	SLOG("Accessing undefined attribute of type INT - This should never occur");
				//	m_IntProps[prop] = 0;
				//}
		//	}
			return(&(m_IntProps[prop]));
			break;
		case Enums::IVEC2:
		//	if (m_Int2Props.count(prop) == 0) {
				getPropi2((AttributeValues::Int2Property)prop);
				//val = m_Attribs->getDefault(prop, type);
				//if (val != NULL)	
				//	m_Int2Props[prop] = *(ivec2 *)val;
				//else { // life goes on ... except in debug mode
				//	assert(false && "Accessing undefined attribute of type IVEC2");
				//	SLOG("Accessing undefined attribute of type IVEC2 - This should never occur");
				//	m_Int2Props[prop] = 0;
				//}
		//	}
			return(&(m_Int2Props[prop]));
			break;
		case Enums::UINT:
		//	if (m_UIntProps.count(prop) == 0) {
				getPropui((AttributeValues::UIntProperty)prop);
				//val = m_Attribs->getDefault(prop, type);
				//if (val != NULL)
				//	m_UIntProps[prop] = *(unsigned int *)val;
				//else { // life goes on ... except in debug mode
				//	assert(false && "Accessing undefined attribute of type UINT");
				//	SLOG("Accessing undefined attribute of type UINT - This should never occur");
				//	m_UIntProps[prop] = 0;
				//}
		//	}
			return(&(m_UIntProps[prop]));
			break;
		case Enums::UIVEC2:
		//	if (m_UInt2Props.count(prop) == 0) {
				getPropui2((AttributeValues::UInt2Property)prop);
				//val = m_Attribs->getDefault(prop, type);
				//if (val != NULL)
				//	m_UInt2Props[prop] = *(uivec2 *)val;
				//else { // life goes on ... except in debug mode
				//	assert(false && "Accessing undefined attribute of type UINT2");
				//	SLOG("Accessing undefined attribute of type UINT2 - This should never occur");
				//	m_UInt2Props[prop] = uivec2(0);
				//}
		//	}
			return(&(m_UInt3Props[prop]));
			break;
		case Enums::UIVEC3:
		//	if (m_UInt3Props.count(prop) == 0) {
				getPropui3((AttributeValues::UInt3Property)prop);
				//val = m_Attribs->getDefault(prop, type);
				//if (val != NULL)
				//	m_UInt3Props[prop] = *(uivec3 *)val;
				//else { // life goes on ... except in debug mode
				//	assert(false && "Accessing undefined attribute of type UINT3");
				//	SLOG("Accessing undefined attribute of type UINT3 - This should never occur");
				//	m_UInt3Props[prop] = uivec3(0);
				//}
		//	}
			return(&(m_UInt3Props[prop]));
			break;
		case Enums::BOOL:
		//	if (m_BoolProps.count(prop) == 0) {
				getPropb((AttributeValues::BoolProperty)prop);
				//val = m_Attribs->getDefault(prop, type);
				//if (val != NULL)
				//	m_BoolProps[prop] = *(bool *)val;
				//else { // life goes on ... except in debug mode
				//	assert(false && "Accessing undefined attribute of type BOOL");
				//	SLOG("Accessing undefined attribute of type BOOL - This should never occur");
				//	m_BoolProps[prop] = 0;
				//}
		//	}
			return(&(m_BoolProps[prop]));
			break;
		case Enums::BVEC4:
			//if (m_Bool4Props.count(prop) == 0) {
				getPropb4((AttributeValues::Bool4Property)prop);
				//val = m_Attribs->getDefault(prop, type);
				//if (val != NULL)
				//	m_Bool4Props[prop] = *(bvec4 *)val;
				//else { // life goes on ... except in debug mode
				//	assert(false && "Accessing undefined attribute of type BVEC4");
				//	SLOG("Accessing undefined attribute of type BVEC4 - This should never occur");
				//	m_Bool4Props[prop] = 0;
				//}
			//}
			return(&(m_Bool4Props[prop]));
			break;
		case Enums::FLOAT:
			//if (m_FloatProps.count(prop) == 0) {
				getPropf((AttributeValues::FloatProperty)prop);
				//val = m_Attribs->getDefault(prop, type);
				//if (val != NULL)
				//	m_FloatProps[prop] = *(float *)val;
				//else { // life goes on ... except in debug mode
				//	assert(false && "Accessing undefined user attribute of type FLOAT");
				//	SLOG("Accessing undefined user attribute of type FLOAT - This should never occur");
				//	m_FloatProps[prop] = 0;
				//}
			//}
			return(&(m_FloatProps[prop]));
			break;
		case Enums::VEC4:
			//if (m_Float4Props.count(prop) == 0) {
				getPropf4((AttributeValues::Float4Property)prop);
				//val = m_Attribs->getDefault(prop, type);
				//if (val != NULL)
				//	m_Float4Props[prop] = *(vec4 *)val;
				//else { // life goes on ... except in debug mode
				//	assert(false && "Accessing undefined user attribute of type VEC4");
				//	SLOG("Accessing undefined user attribute of type VEC4 - This should never occur");
				//	m_Float4Props[prop] = vec4(0.0f);
				//}
		//	}
			return(&(m_Float4Props[prop]));
			break;
		case Enums::VEC3:
			if (m_Float3Props.count(prop) == 0) {
				getPropf3((AttributeValues::Float3Property)prop);
				//val = m_Attribs->getDefault(prop, type);
				//if (val != NULL)
				//	m_Float3Props[prop] = *(vec3 *)val;
				//else { // life goes on ... except in debug mode
				//	assert(false && "Accessing undefined user attribute of type VEC3");
				//	SLOG("Accessing undefined user attribute of type VEC3 - This should never occur");
				//	m_Float3Props[prop] = vec3(0.0f);
				//}
			}
			return(&(m_Float3Props[prop]));
			break;
		case Enums::VEC2:
		//	if (m_Float2Props.count(prop) == 0) {
				getPropf2((AttributeValues::Float2Property)prop);
				//val = m_Attribs->getDefault(prop, type);
				//if (val != NULL)
				//	m_Float2Props[prop] = *(vec2 *)val;
				//else { // life goes on ... except in debug mode
				//	assert(false && "Accessing undefined user attribute of type VEC2");
				//	SLOG("Accessing undefined user attribute of type VEC2 - This should never occur");
				//	m_Float2Props[prop] = vec2(0.0f);
				//}
		//	}
			return(&(m_Float2Props[prop]));
			break;
		case Enums::MAT4:
//			if (m_Mat4Props.count(prop) == 0) {
				getPropm4((AttributeValues::Mat4Property)prop);
				//val = m_Attribs->getDefault(prop, type);
				//if (val != NULL)
				//	m_Mat4Props[prop] = *(mat4 *)val;
				//else { // life goes on ... except in debug mode
				//	assert(false && "Accessing undefined user attribute of type MAT4");
				//	SLOG("Accessing undefined user attribute of type MAT4 - This should never occur");
				//	m_Mat4Props[prop] = mat4();
				//}
	//		}
			return(&(m_Mat4Props[prop]));
			break;
		case Enums::MAT3:
			//if (m_Mat3Props.count(prop) == 0) {
				getPropm3((AttributeValues::Mat3Property)prop);
				//val = m_Attribs->getDefault(prop, type);
				//if (val != NULL)
				//	m_Mat3Props[prop] = *(mat3 *)val;
				//else { // life goes on ... except in debug mode
				//	assert(false && "Accessing undefined user attribute of type MAT3");
				//	SLOG("Accessing undefined user attribute of type MAT3 - This should never occur");
				//	m_Mat3Props[prop] = mat3();
				//}
			//}
			return(&(m_Mat3Props[prop]));
			break;
		default:
			assert(false && "Missing Data Type in class attributeValues");
			return NULL;
	}
};


void 
AttributeValues::setProp(unsigned int prop, Enums::DataType type, Data *value) {

	switch (type) {

	case Enums::ENUM:
		setPrope((EnumProperty)prop, dynamic_cast<NauInt *>(value)->getNumber());
		break;
	case Enums::INT:
		setPropi((IntProperty)prop, dynamic_cast<NauInt *>(value)->getNumber());
		break;
	case Enums::IVEC2:
		setPropi2((Int2Property)prop, *(dynamic_cast<ivec2 *>(value)));
		break;
	case Enums::UINT:
		setPropui((UIntProperty)prop, dynamic_cast<NauUInt *>(value)->getNumber());
		break;
	case Enums::UIVEC2:
		setPropui2((UInt2Property)prop, *(dynamic_cast<uivec2 *>(value)));
		break;
	case Enums::UIVEC3:
		setPropui3((UInt3Property)prop, *(dynamic_cast<uivec3 *>(value)));
		break;
	case Enums::BOOL:
		setPropb((BoolProperty)prop, dynamic_cast<NauInt *>(value)->getNumber() != 0);
		break;
	case Enums::BVEC4:
		setPropb4((Bool4Property)prop, *(dynamic_cast<bvec4 *>(value)));
		break;
	case Enums::FLOAT:
		setPropf((FloatProperty)prop, dynamic_cast<NauFloat *>(value)->getNumber());
		break;
	case Enums::VEC2:
		setPropf2((Float2Property)prop, *(dynamic_cast<vec2 *>(value)));
		break;
	case Enums::VEC3:
		setPropf3((Float3Property)prop, *(dynamic_cast<vec3 *>(value)));
		break;
	case Enums::VEC4:
		setPropf4((Float4Property)prop, *(dynamic_cast<vec4 *>(value)));
		break;
	case Enums::MAT4:
		setPropm4((Mat4Property)prop, *(dynamic_cast<mat4 *>(value)));
		break;
	case Enums::MAT3:
		setPropm3((Mat3Property)prop, *(dynamic_cast<mat3 *>(value)));
		break;
	default:
		assert(false && "Missing Data Type in class attributeValues or Invalid prop");
	}	
}


bool
AttributeValues::isValid(unsigned int prop, Enums::DataType type, Data *value) {

	switch (type) {

	case Enums::ENUM:
		return isValide((EnumProperty)prop, dynamic_cast<NauInt *>(value)->getNumber());
		break;
	case Enums::INT:
		return isValidi((IntProperty)prop, dynamic_cast<NauInt *>(value)->getNumber());
		break;
	case Enums::IVEC2:
		return isValidi2((Int2Property)prop, *(dynamic_cast<ivec2 *>(value)));
		break;
	case Enums::UINT:
		return isValidui((UIntProperty)prop, dynamic_cast<NauUInt *>(value)->getNumber());
		break;
	case Enums::UIVEC2:
		return isValidui2((UInt2Property)prop, *(dynamic_cast<uivec2 *>(value)));
		break;
	case Enums::UIVEC3:
		return isValidui3((UInt3Property)prop, *(dynamic_cast<uivec3 *>(value)));
		break;
	case Enums::BOOL:
		return isValidb((BoolProperty)prop, dynamic_cast<NauInt *>(value)->getNumber() != 0);
		break;
	case Enums::BVEC4:
		return isValidb4((Bool4Property)prop, *(dynamic_cast<bvec4 *>(value)));
		break;
	case Enums::FLOAT:
		return isValidf((FloatProperty)prop, dynamic_cast<NauFloat *>(value)->getNumber());
		break;
	case Enums::VEC2:
		return isValidf2((Float2Property)prop, *(dynamic_cast<vec2 *>(value)));
		break;
	case Enums::VEC3:
		return isValidf3((Float3Property)prop, *(dynamic_cast<vec3 *>(value)));
		break;
	case Enums::VEC4:
		return isValidf4((Float4Property)prop, *(dynamic_cast<vec4 *>(value)));
		break;
	case Enums::MAT4:
		return isValidm4((Mat4Property)prop, *(dynamic_cast<mat4 *>(value)));
		break;
	case Enums::MAT3:
		return isValidm3((Mat3Property)prop, *(dynamic_cast<mat3 *>(value)));
		break;
	default:
		assert(false && "Missing Data Type in class attributeValues or Invalid prop");
		return false;
	}
}


void 
AttributeValues::registerAndInitArrays(AttribSet &attribs) {

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
	m_Attribs->initAttribInstanceUInt2Array(m_UInt2Props);
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


AttribSet *
AttributeValues::getAttribSet() {

	return m_Attribs;
}


AttributeValues::AttributeValues() {

}


AttributeValues::AttributeValues(const AttributeValues &to) {

	m_EnumProps = to.m_EnumProps;
	m_IntProps = to.m_IntProps;
	m_Int2Props = to.m_Int2Props;
	m_UIntProps = to.m_UIntProps;
	m_UInt2Props = to.m_UInt2Props;
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



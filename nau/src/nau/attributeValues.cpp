#include "nau/attributeValues.h"

#include "nau.h"
#include "nau/math/number.h"
//#include "nau/math/numberArray.h"
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
	if (m_Attribs->get(name)->getMustExist()) {
		const std::string &context = m_Attribs->get(name)->getObjType();
		return NAU->validateObjectName(context, value);
	}
	else return true;
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
		std::shared_ptr<NauInt> ni = std::dynamic_pointer_cast<NauInt>(d);
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
//		INTARRAY
// ----------------------------------------------


NauIntArray
AttributeValues::getPropiv(IntArrayProperty prop) {

	// if not a user attrib
	if (prop < AttribSet::USER_ATTRIBS)
		return m_IntArrayProps[prop];
	// if prop is a user attrib and has already been set
	else if (m_IntArrayProps.count(prop))
		return m_IntArrayProps[prop];
	else {
		m_IntArrayProps[prop] = NauIntArray();
		return m_IntArrayProps[prop];
	}
}


bool
AttributeValues::isValidiv(IntArrayProperty prop, NauIntArray &value) {

	return true;
}


void
AttributeValues::setPropiv(IntArrayProperty prop, NauIntArray &value) {

	m_IntArrayProps[prop] = value;
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
		std::shared_ptr<NauInt> ni = std::dynamic_pointer_cast<NauInt>(d);
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
		const std::shared_ptr<NauInt> &max = std::dynamic_pointer_cast<NauInt>(attr->getMax());
		const std::shared_ptr<NauInt> &min = std::dynamic_pointer_cast<NauInt>(attr->getMin());

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
		std::shared_ptr<ivec2> ni = std::dynamic_pointer_cast<ivec2>(d);
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
		std::shared_ptr<ivec2> max = std::dynamic_pointer_cast<ivec2>(attr->getMax());
		std::shared_ptr<ivec2> min = std::dynamic_pointer_cast<ivec2>(attr->getMin());

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
		std::shared_ptr<ivec3> ni = std::dynamic_pointer_cast<ivec3>(d);
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
		std::shared_ptr<ivec3> max = std::dynamic_pointer_cast<ivec3>(attr->getMax());
		std::shared_ptr<ivec3> min = std::dynamic_pointer_cast<ivec3>(attr->getMin());

		if (max != NULL && value > *max)
			return false;
		if (min != NULL && value < *min)
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
//		INT4
// ----------------------------------------------


ivec4 &
AttributeValues::getPropi4(Int4Property prop) {

	// if not a user attrib
	if (prop < AttribSet::USER_ATTRIBS)
		return m_Int4Props[prop];
	// if prop is a user attrib and has already been set
	else if (m_Int4Props.count(prop))
		return m_Int4Props[prop];
	else {
		std::shared_ptr<Data> &d = m_Attribs->get(prop, Enums::IVEC4)->getDefault();
		std::shared_ptr<ivec4> ni = std::dynamic_pointer_cast<ivec4>(d);
		m_Int4Props[prop] = *ni;
		return m_Int4Props[prop];
	}
}


bool
AttributeValues::isValidi4(Int4Property prop, ivec4 &value) {

	std::unique_ptr<Attribute> &attr = m_Attribs->get(prop, Enums::IVEC4);
	if (attr->getName() == "NO_ATTR")
		return false;
	if (!APISupport->apiSupport(attr->getRequirement()))
		return false;

	if (attr->getRangeDefined()) {
		std::shared_ptr<ivec4> max = (std::dynamic_pointer_cast<ivec4>(attr->getMax()));
		std::shared_ptr<ivec4> min = (std::dynamic_pointer_cast<ivec4>(attr->getMin()));

		if (max != NULL && value > *max)
			return false;
		if (min != NULL && value < *min)
			return false;
	}
	return true;
}


void
AttributeValues::setPropi4(Int4Property prop, ivec4 &value) {

	assert(isValidi4(prop, value));
	m_Int4Props[prop] = value;
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
		std::shared_ptr<NauUInt> ni = std::dynamic_pointer_cast<NauUInt>(d);
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
		std::shared_ptr<NauUInt> max = std::dynamic_pointer_cast<NauUInt>(attr->getMax());
		std::shared_ptr<NauUInt> min = std::dynamic_pointer_cast<NauUInt>(attr->getMin());

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
		std::shared_ptr<uivec2> ni = std::dynamic_pointer_cast<uivec2>(d);
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
		std::shared_ptr<uivec2> max = std::dynamic_pointer_cast<uivec2>(attr->getMax());
		std::shared_ptr<uivec2> min = std::dynamic_pointer_cast<uivec2>(attr->getMin());

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
		std::shared_ptr<uivec3> ni = std::dynamic_pointer_cast<uivec3>(d);
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
		std::shared_ptr<uivec3> max = std::dynamic_pointer_cast<uivec3>(attr->getMax());
		std::shared_ptr<uivec3> min = std::dynamic_pointer_cast<uivec3>(attr->getMin());

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
//		UINT4
// ----------------------------------------------


uivec4 &
AttributeValues::getPropui4(UInt4Property prop) {

	// if not a user attrib
	if (prop < AttribSet::USER_ATTRIBS)
		return m_UInt4Props[prop];
	// if prop is a user attrib and has already been set
	else if (m_UInt4Props.count(prop))
		return m_UInt4Props[prop];
	else {
		std::shared_ptr<Data> &d = m_Attribs->get(prop, Enums::UIVEC4)->getDefault();
		std::shared_ptr<uivec4> ni = std::dynamic_pointer_cast<uivec4>(d);
		m_UInt4Props[prop] = *ni;
		return m_UInt4Props[prop];
	}
}


bool
AttributeValues::isValidui4(UInt4Property prop, uivec4 &value) {

	std::unique_ptr<Attribute> &attr = m_Attribs->get(prop, Enums::UIVEC4);
	if (attr->getName() == "NO_ATTR")
		return false;
	if (!APISupport->apiSupport(attr->getRequirement()))
		return false;

	if (attr->getRangeDefined()) {
		std::shared_ptr<uivec4> max = std::dynamic_pointer_cast<uivec4>(attr->getMax());
		std::shared_ptr<uivec4> min = std::dynamic_pointer_cast<uivec4>(attr->getMin());

		if (max != NULL && value > *max)
			return false;
		if (min != NULL && value < *min)
			return false;
	}
	return true;
}


void
AttributeValues::setPropui4(UInt4Property prop, uivec4 &value) {
	assert(isValidui4(prop, value));
	m_UInt4Props[prop] = value;
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
		std::shared_ptr<NauInt> ni = std::dynamic_pointer_cast<NauInt>(d);
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
//		BOOL2
// ----------------------------------------------


bvec2 &
AttributeValues::getPropb2(Bool2Property prop) {

	// if not a user attrib
	if (prop < AttribSet::USER_ATTRIBS)
		return m_Bool2Props[prop];
	// if prop is a user attrib and has already been set
	else if (m_Bool2Props.count(prop))
		return m_Bool2Props[prop];
	else {
		std::shared_ptr<Data> &d = m_Attribs->get(prop, Enums::BVEC2)->getDefault();
		std::shared_ptr<bvec2> ni = std::dynamic_pointer_cast<bvec2>(d);
		m_Bool2Props[prop] = *ni;
		return m_Bool2Props[prop];
	}
}


bool
AttributeValues::isValidb2(Bool2Property prop, bvec2 &value) {

	std::unique_ptr<Attribute> &attr = m_Attribs->get(prop, Enums::BVEC2);
	if (attr->getName() == "NO_ATTR")
		return false;
	if (!APISupport->apiSupport(attr->getRequirement()))
		return false;
	else
		return true;
}


void
AttributeValues::setPropb2(Bool2Property prop, bvec2 &value) {
	assert(isValidb2(prop, value));
	m_Bool2Props[prop] = value;
}

// ----------------------------------------------
//		BOOL3
// ----------------------------------------------


bvec3 &
AttributeValues::getPropb3(Bool3Property prop) {

	// if not a user attrib
	if (prop < AttribSet::USER_ATTRIBS)
		return m_Bool3Props[prop];
	// if prop is a user attrib and has already been set
	else if (m_Bool3Props.count(prop))
		return m_Bool3Props[prop];
	else {
		std::shared_ptr<Data> &d = m_Attribs->get(prop, Enums::BVEC3)->getDefault();
		std::shared_ptr<bvec3> ni = std::dynamic_pointer_cast<bvec3>(d);
		m_Bool3Props[prop] = *ni;
		return m_Bool3Props[prop];
	}
}


bool
AttributeValues::isValidb3(Bool3Property prop, bvec3 &value) {

	std::unique_ptr<Attribute> &attr = m_Attribs->get(prop, Enums::BVEC3);
	if (attr->getName() == "NO_ATTR")
		return false;
	if (!APISupport->apiSupport(attr->getRequirement()))
		return false;
	else
		return true;
}


void
AttributeValues::setPropb3(Bool3Property prop, bvec3 &value) {
	assert(isValidb3(prop, value));
	m_Bool3Props[prop] = value;
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
		std::shared_ptr<bvec4> ni = std::dynamic_pointer_cast<bvec4>(d);
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
		std::shared_ptr<NauFloat> ni = std::dynamic_pointer_cast<NauFloat>(d);
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
		std::shared_ptr<NauFloat> max = std::dynamic_pointer_cast<NauFloat>(attr->getMax());
		std::shared_ptr<NauFloat> min = std::dynamic_pointer_cast<NauFloat>(attr->getMin());

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
		std::shared_ptr<vec4> ni = std::dynamic_pointer_cast<vec4>(d);
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
		std::shared_ptr<vec4> max = std::dynamic_pointer_cast<vec4>(attr->getMax());
		std::shared_ptr<vec4> min = std::dynamic_pointer_cast<vec4>(attr->getMin());

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

	vec4 v4(x,y,z,w);
	setPropf4(prop, v4);

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
		std::shared_ptr<vec3> ni = std::dynamic_pointer_cast<vec3>(d);
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
		std::shared_ptr<vec3> max = std::dynamic_pointer_cast<vec3>(attr->getMax());
		std::shared_ptr<vec3> min = std::dynamic_pointer_cast<vec3>(attr->getMin());

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

	vec3 v3(x,y,z);
	setPropf3(prop, v3);

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
		std::shared_ptr<vec2> ni = std::dynamic_pointer_cast<vec2>(d);
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
		std::shared_ptr<vec2> max = std::dynamic_pointer_cast<vec2>(attr->getMax());
		std::shared_ptr<vec2> min = std::dynamic_pointer_cast<vec2>(attr->getMin());

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
		std::shared_ptr<mat4> ni = std::dynamic_pointer_cast<mat4>(d);
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
			std::shared_ptr<mat3> ni = std::dynamic_pointer_cast<mat3>(d);
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
//		MAT2
// ----------------------------------------------


const mat2 &
AttributeValues::getPropm2(Mat2Property prop) {

	// if not a user attrib
	if (prop < AttribSet::USER_ATTRIBS)
		return m_Mat2Props[prop];
	// if prop is a user attrib and has already been set
	else if (m_Mat2Props.count(prop))
		return m_Mat2Props[prop];
	else {
		std::unique_ptr<Attribute> &a = m_Attribs->get(prop, Enums::MAT2);
		if (a->getId() == -1) {
			assert(false && "Attribute Values getProp - invalid MAT2 prop");
			SLOG("Attribute MAT2 with id %d does not exist", prop);
			Data *m = Enums::getDefaultValue(Enums::MAT2);
			m_Mat2Props[prop] = *(mat2 *)m;
			delete m;
		}
		else {
			std::shared_ptr<Data> &d = a->getDefault();
			std::shared_ptr<mat2> ni = std::dynamic_pointer_cast<mat2>(d);
			m_Mat2Props[prop] = *ni;
		}
		return m_Mat2Props[prop];
	}
}



bool
AttributeValues::isValidm2(Mat2Property prop, mat2 &value) {

	std::unique_ptr<Attribute> &attr = m_Attribs->get(prop, Enums::MAT2);
	if (attr->getName() == "NO_ATTR")
		return false;
	if (!APISupport->apiSupport(attr->getRequirement()))
		return false;
	else
		return true;
}


void
AttributeValues::setPropm2(Mat2Property prop, mat2 &value) {
	assert(isValidm2(prop, value));
	m_Mat2Props[prop] = value;
}


// ----------------------------------------------
//		DOUBLE
// ----------------------------------------------


double
AttributeValues::getPropd(DoubleProperty prop) {

	// if not a user attrib
	if (prop < AttribSet::USER_ATTRIBS)
		return m_DoubleProps[prop];
	// if prop is a user attrib and has already been set
	else if (m_DoubleProps.count(prop))
		return m_DoubleProps[prop];
	else {
		std::shared_ptr<Data> &d = m_Attribs->get(prop, Enums::DOUBLE)->getDefault();
		std::shared_ptr<NauDouble> ni = std::dynamic_pointer_cast<NauDouble>(d);
		m_DoubleProps[prop] = ni->getNumber();
		return m_DoubleProps[prop];
	}
	return m_DoubleProps[prop];
}


bool
AttributeValues::isValidd(DoubleProperty prop, double f) {

	std::unique_ptr<Attribute> &attr = m_Attribs->get(prop, Enums::DOUBLE);
	if (attr->getName() == "NO_ATTR")
		return false;
	if (!APISupport->apiSupport(attr->getRequirement()))
		return false;

	if (attr->getRangeDefined()) {
		std::shared_ptr<NauDouble> max = std::dynamic_pointer_cast<NauDouble>(attr->getMax());
		std::shared_ptr<NauDouble> min = std::dynamic_pointer_cast<NauDouble>(attr->getMin());

		if (max != NULL && f > *max)
			return false;
		if (min != NULL && f < *min)
			return false;
	}
	return true;
}


void
AttributeValues::setPropd(DoubleProperty prop, double value) {
	assert(isValidd(prop, value));
	m_DoubleProps[prop] = value;
}


// ----------------------------------------------
//		DVEC4
// ----------------------------------------------


dvec4 &
AttributeValues::getPropd4(Double4Property prop) {

	// if not a user attrib
	if (prop < AttribSet::USER_ATTRIBS)
		return m_Double4Props[prop];
	// if prop is a user attrib and has already been set
	else if (m_Double4Props.count(prop))
		return m_Double4Props[prop];
	else {
		std::shared_ptr<Data> &d = m_Attribs->get(prop, Enums::DVEC4)->getDefault();
		std::shared_ptr<dvec4> ni = std::dynamic_pointer_cast<dvec4>(d);
		m_Double4Props[prop] = *ni;
		return m_Double4Props[prop];
	}
}


bool
AttributeValues::isValidd4(Double4Property prop, dvec4 &f) {

	std::unique_ptr<Attribute> &attr = m_Attribs->get(prop, Enums::DVEC4);
	if (attr->getName() == "NO_ATTR")
		return false;
	if (!APISupport->apiSupport(attr->getRequirement()))
		return false;

	if (attr->getRangeDefined()) {
		std::shared_ptr<dvec4> max = std::dynamic_pointer_cast<dvec4>(attr->getMax());
		std::shared_ptr<dvec4> min = std::dynamic_pointer_cast<dvec4>(attr->getMin());

		if (max != NULL && f > *max)
			return false;
		if (min != NULL && f < *min)
			return false;
	}
	return true;
}


void
AttributeValues::setPropd4(Double4Property prop, dvec4 &value) {

	assert(isValidd4(prop, value));
	m_Double4Props[prop] = value;
}


void
AttributeValues::setPropd4(Double4Property prop, double x, double y, double z, double w) {

	dvec4 dv4(x,y,z,w);
	setPropd4(prop, dv4);

}


// ----------------------------------------------
//		DVEC3
// ----------------------------------------------


dvec3 &
AttributeValues::getPropd3(Double3Property prop) {

	// if not a user attrib
	if (prop < AttribSet::USER_ATTRIBS)
		return m_Double3Props[prop];
	// if prop is a user attrib and has already been set
	else if (m_Double3Props.count(prop))
		return m_Double3Props[prop];
	else {
		std::shared_ptr<Data> &d = m_Attribs->get(prop, Enums::DVEC3)->getDefault();
		std::shared_ptr<dvec3> ni = std::dynamic_pointer_cast<dvec3>(d);
		m_Double3Props[prop] = *ni;
		return m_Double3Props[prop];
	}
}


bool
AttributeValues::isValidd3(Double3Property prop, dvec3 &f) {

	std::unique_ptr<Attribute> &attr = m_Attribs->get(prop, Enums::DVEC3);
	if (attr->getName() == "NO_ATTR")
		return false;
	if (!APISupport->apiSupport(attr->getRequirement()))
		return false;

	if (attr->getRangeDefined()) {
		std::shared_ptr<dvec3> max = std::dynamic_pointer_cast<dvec3>(attr->getMax());
		std::shared_ptr<dvec3> min = std::dynamic_pointer_cast<dvec3>(attr->getMin());

		if (max != NULL && f > *max)
			return false;
		if (min != NULL && f < *min)
			return false;
	}
	return true;
}


void
AttributeValues::setPropd3(Double3Property prop, dvec3 &value) {

	assert(isValidd3(prop, value));
	m_Double3Props[prop] = value;
}


void
AttributeValues::setPropd3(Double3Property prop, double x, double y, double z) {
	dvec3 dv3(x,y,z);
	setPropd3(prop, dv3);

}

// ----------------------------------------------
//		DVEC2
// ----------------------------------------------


dvec2 &
AttributeValues::getPropd2(Double2Property prop) {

	// if not a user attrib
	if (prop < AttribSet::USER_ATTRIBS)
		return m_Double2Props[prop];
	// if prop is a user attrib and has already been set
	else if (m_Double2Props.count(prop))
		return m_Double2Props[prop];
	else {
		std::shared_ptr<Data> &d = m_Attribs->get(prop, Enums::DVEC2)->getDefault();
		std::shared_ptr<dvec2> ni = std::dynamic_pointer_cast<dvec2>(d);
		m_Double2Props[prop] = *ni;
		return m_Double2Props[prop];
	}
}


bool
AttributeValues::isValidd2(Double2Property prop, dvec2 &f) {

	std::unique_ptr<Attribute> &attr = m_Attribs->get(prop, Enums::DVEC2);
	if (attr->getName() == "NO_ATTR")
		return false;
	if (!APISupport->apiSupport(attr->getRequirement()))
		return false;

	if (attr->getRangeDefined()) {
		std::shared_ptr<dvec2> max = std::dynamic_pointer_cast<dvec2>(attr->getMax());
		std::shared_ptr<dvec2> min = std::dynamic_pointer_cast<dvec2>(attr->getMin());

		if (max != NULL && f > *max)
			return false;
		if (min != NULL && f < *min)
			return false;
	}
	return true;
}


void
AttributeValues::setPropd2(Double2Property prop, dvec2 &value) {
	assert(isValidd2(prop, value));
	m_Double2Props[prop] = value;
}


// ----------------------------------------------
//		DMAT4
// ----------------------------------------------


const dmat4 &
AttributeValues::getPropdm4(DMat4Property prop) {

	// if not a user attrib
	if (prop < AttribSet::USER_ATTRIBS)
		return m_DMat4Props[prop];
	// if prop is a user attrib and has already been set
	else if (m_DMat4Props.count(prop))
		return m_DMat4Props[prop];
	else {
		std::shared_ptr<Data> &d = m_Attribs->get(prop, Enums::DMAT4)->getDefault();
		std::shared_ptr<dmat4> ni = std::dynamic_pointer_cast<dmat4>(d);
		m_DMat4Props[prop] = *ni;
		return m_DMat4Props[prop];
	}
}


bool
AttributeValues::isValiddm4(DMat4Property prop, dmat4 &value) {

	std::unique_ptr<Attribute> &attr = m_Attribs->get(prop, Enums::DMAT4);
	if (attr->getName() == "NO_ATTR")
		return false;
	if (!APISupport->apiSupport(attr->getRequirement()))
		return false;
	else
		return true;
}


void
AttributeValues::setPropdm4(DMat4Property prop, dmat4 &value) {
	assert(isValiddm4(prop, value));
	m_DMat4Props[prop] = value;
}


// ----------------------------------------------
//		DMAT3
// ----------------------------------------------


const dmat3 &
AttributeValues::getPropdm3(DMat3Property prop) {

	// if not a user attrib
	if (prop < AttribSet::USER_ATTRIBS)
		return m_DMat3Props[prop];
	// if prop is a user attrib and has already been set
	else if (m_DMat3Props.count(prop))
		return m_DMat3Props[prop];
	else {
		std::unique_ptr<Attribute> &a = m_Attribs->get(prop, Enums::DMAT3);
		if (a->getId() == -1) {
			assert(false && "Attribute Values getProp - invalid DMAT3 prop");
			SLOG("Attribute DMAT3 with id %d does not exist", prop);
			Data *m = Enums::getDefaultValue(Enums::DMAT3);
			m_DMat3Props[prop] = *(dmat3 *)m;
			delete m;
		}
		else {
			std::shared_ptr<Data> &d = a->getDefault();
			std::shared_ptr<dmat3> ni = std::dynamic_pointer_cast<dmat3>(d);
			m_DMat3Props[prop] = *ni;
		}
		return m_DMat3Props[prop];
	}
}



bool
AttributeValues::isValiddm3(DMat3Property prop, dmat3 &value) {

	std::unique_ptr<Attribute> &attr = m_Attribs->get(prop, Enums::DMAT3);
	if (attr->getName() == "NO_ATTR")
		return false;
	if (!APISupport->apiSupport(attr->getRequirement()))
		return false;
	else
		return true;
}


void
AttributeValues::setPropdm3(DMat3Property prop, dmat3 &value) {
	assert(isValiddm3(prop, value));
	m_DMat3Props[prop] = value;
}


// ----------------------------------------------
//		DMAT2
// ----------------------------------------------


const dmat2 &
AttributeValues::getPropdm2(DMat2Property prop) {

	// if not a user attrib
	if (prop < AttribSet::USER_ATTRIBS)
		return m_DMat2Props[prop];
	// if prop is a user attrib and has already been set
	else if (m_DMat2Props.count(prop))
		return m_DMat2Props[prop];
	else {
		std::unique_ptr<Attribute> &a = m_Attribs->get(prop, Enums::DMAT2);
		if (a->getId() == -1) {
			assert(false && "Attribute Values getProp - invalid DMAT2 prop");
			SLOG("Attribute DMAT2 with id %d does not exist", prop);
			Data *m = Enums::getDefaultValue(Enums::DMAT2);
			m_DMat2Props[prop] = *(dmat2 *)m;
			delete m;
		}
		else {
			std::shared_ptr<Data> &d = a->getDefault();
			std::shared_ptr<dmat2> ni = std::dynamic_pointer_cast<dmat2>(d);
			m_DMat2Props[prop] = *ni;
		}
		return m_DMat2Props[prop];
	}
}



bool
AttributeValues::isValiddm2(DMat2Property prop, dmat2 &value) {

	std::unique_ptr<Attribute> &attr = m_Attribs->get(prop, Enums::DMAT2);
	if (attr->getName() == "NO_ATTR")
		return false;
	if (!APISupport->apiSupport(attr->getRequirement()))
		return false;
	else
		return true;
}


void
AttributeValues::setPropdm2(DMat2Property prop, dmat2 &value) {
	assert(isValiddm2(prop, value));
	m_DMat2Props[prop] = value;
}

// ----------------------------------------------
//		All
// ----------------------------------------------

const AttributeValues& 
AttributeValues::operator =(const AttributeValues &to) {

	if (this != &to) {
		m_EnumProps = to.m_EnumProps;

		m_StringProps = to.m_StringProps;

		m_IntArrayProps = to.m_IntArrayProps;

		m_IntProps = to.m_IntProps;
		m_Int2Props = to.m_Int2Props;
		m_Int3Props = to.m_Int3Props;
		m_Int4Props = to.m_Int4Props;

		m_UIntProps = to.m_UIntProps;
		m_UInt2Props = to.m_UInt2Props;
		m_UInt3Props = to.m_UInt3Props;
		m_UInt4Props = to.m_UInt4Props;

		m_BoolProps = to.m_BoolProps;
		m_Bool2Props = to.m_Bool2Props;
		m_Bool3Props = to.m_Bool3Props;
		m_Bool4Props = to.m_Bool4Props;

		m_FloatProps = to.m_FloatProps;
		m_Float2Props = to.m_Float2Props;
		m_Float3Props = to.m_Float3Props;
		m_Float4Props = to.m_Float4Props;
		m_Mat2Props = to.m_Mat2Props;
		m_Mat3Props = to.m_Mat3Props;
		m_Mat4Props = to.m_Mat4Props;

		m_DoubleProps = to.m_DoubleProps;
		m_Double2Props = to.m_Double2Props;
		m_Double3Props = to.m_Double3Props;
		m_Double4Props = to.m_Double4Props;
		m_DMat2Props = to.m_DMat2Props;
		m_DMat3Props = to.m_DMat3Props;
		m_DMat4Props = to.m_DMat4Props;

		m_Attribs = to.m_Attribs;
	}
	return *this;
}


void
AttributeValues::copy(AttributeValues *to) {

	//to->m_StringProps = m_StringProps;
	m_EnumProps = to->m_EnumProps;

	m_StringProps = to->m_StringProps;

	m_IntArrayProps = to->m_IntArrayProps;

	m_IntProps = to->m_IntProps;
	m_Int2Props = to->m_Int2Props;
	m_Int3Props = to->m_Int3Props;
	m_Int4Props = to->m_Int4Props;

	m_UIntProps = to->m_UIntProps;
	m_UInt2Props = to->m_UInt2Props;
	m_UInt3Props = to->m_UInt3Props;
	m_UInt4Props = to->m_UInt4Props;

	m_BoolProps = to->m_BoolProps;
	m_Bool4Props = to->m_Bool4Props;
	m_Bool2Props = to->m_Bool2Props;
	m_Bool3Props = to->m_Bool3Props;

	m_FloatProps = to->m_FloatProps;
	m_Float2Props = to->m_Float2Props;
	m_Float3Props = to->m_Float3Props;
	m_Float4Props = to->m_Float4Props;
	m_Mat2Props = to->m_Mat2Props;
	m_Mat3Props = to->m_Mat3Props;
	m_Mat4Props = to->m_Mat4Props;

	m_DoubleProps = to->m_DoubleProps;
	m_Double2Props = to->m_Double2Props;
	m_Double3Props = to->m_Double3Props;
	m_Double4Props = to->m_Double4Props;
	m_DMat2Props = to->m_DMat2Props;
	m_DMat3Props = to->m_DMat3Props;
	m_DMat4Props = to->m_DMat4Props;

	m_Attribs = to->m_Attribs;
}


void 
AttributeValues::clearArrays() {

	//to->m_StringProps = m_StringProps;
	m_EnumProps.clear();
	m_StringProps.clear();

	m_IntArrayProps.clear();

	m_IntProps.clear();
	m_Int2Props.clear();
	m_Int3Props.clear();
	m_Int4Props.clear();
	
	m_UIntProps.clear();
	m_UInt2Props.clear();
	m_UInt3Props.clear();
	m_UInt4Props.clear();
	
	m_BoolProps.clear();
	m_Bool2Props.clear();
	m_Bool3Props.clear();
	m_Bool4Props.clear();

	m_FloatProps.clear();
	m_Float2Props.clear();
	m_Float3Props.clear();
	m_Float4Props.clear();

	m_DoubleProps.clear();
	m_Double2Props.clear();
	m_Double3Props.clear();
	m_Double4Props.clear();

	m_Mat2Props.clear();
	m_Mat3Props.clear();
	m_Mat4Props.clear();

	m_DMat2Props.clear();
	m_DMat3Props.clear();
	m_DMat4Props.clear();
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
			getPrope((AttributeValues::EnumProperty)prop);
			return(&(m_EnumProps[prop]));
			break;
		case Enums::STRING:
			getProps((AttributeValues::StringProperty)prop);
			return(&(m_StringProps[prop]));
			break;
		case Enums::INTARRAY: 
			getPropiv((AttributeValues::IntArrayProperty)prop);
			return(&(m_IntArrayProps[prop]));
			break;
		case Enums::INT:
			getPropi((AttributeValues::IntProperty)prop);
			return(&(m_IntProps[prop]));
			break;
		case Enums::IVEC2:
			getPropi2((AttributeValues::Int2Property)prop);
			return(&(m_Int2Props[prop]));
			break;
		case Enums::IVEC3:
			getPropi3((AttributeValues::Int3Property)prop);
			return(&(m_Int3Props[prop]));
			break;
		case Enums::IVEC4:
			getPropi4((AttributeValues::Int4Property)prop);
			return(&(m_Int4Props[prop]));
			break;
		case Enums::UINT:
			getPropui((AttributeValues::UIntProperty)prop);
			return(&(m_UIntProps[prop]));
			break;
		case Enums::UIVEC2:
			getPropui2((AttributeValues::UInt2Property)prop);
			return(&(m_UInt3Props[prop]));
			break;
		case Enums::UIVEC3:
			getPropui3((AttributeValues::UInt3Property)prop);
			return(&(m_UInt3Props[prop]));
			break;
		case Enums::BOOL:
			getPropb((AttributeValues::BoolProperty)prop);
			return(&(m_BoolProps[prop]));
			break;
		case Enums::BVEC4:
			getPropb4((AttributeValues::Bool4Property)prop);
			return(&(m_Bool4Props[prop]));
			break;
		case Enums::FLOAT:
			getPropf((AttributeValues::FloatProperty)prop);
			return(&(m_FloatProps[prop]));
			break;
		case Enums::VEC4:
			getPropf4((AttributeValues::Float4Property)prop);
			return(&(m_Float4Props[prop]));
			break;
		case Enums::VEC3:
			getPropf3((AttributeValues::Float3Property)prop);
			return(&(m_Float3Props[prop]));
			break;
		case Enums::VEC2:
			getPropf2((AttributeValues::Float2Property)prop);
			return(&(m_Float2Props[prop]));
			break;
		case Enums::MAT4:
			getPropm4((AttributeValues::Mat4Property)prop);
			return(&(m_Mat4Props[prop]));
			break;
		case Enums::MAT3:
			getPropm3((AttributeValues::Mat3Property)prop);
			return(&(m_Mat3Props[prop]));
			break;
		case Enums::DOUBLE:
			getPropd((AttributeValues::DoubleProperty)prop);
			return(&(m_DoubleProps[prop]));
			break;
		case Enums::DVEC4:
			getPropd4((AttributeValues::Double4Property)prop);
			return(&(m_Double4Props[prop]));
			break;
		case Enums::DVEC3:
			getPropd3((AttributeValues::Double3Property)prop);
			return(&(m_Double3Props[prop]));
			break;
		case Enums::DVEC2:
			getPropd2((AttributeValues::Double2Property)prop);
			return(&(m_Double2Props[prop]));
			break;
		case Enums::DMAT4:
			getPropdm4((AttributeValues::DMat4Property)prop);
			return(&(m_DMat4Props[prop]));
			break;
		case Enums::DMAT3:
			getPropdm3((AttributeValues::DMat3Property)prop);
			return(&(m_DMat3Props[prop]));
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
	case Enums::INTARRAY:
		setPropiv((IntArrayProperty)prop, *(dynamic_cast<NauIntArray *>(value)));
	case Enums::INT:
		setPropi((IntProperty)prop, dynamic_cast<NauInt *>(value)->getNumber());
		break;
	case Enums::IVEC2:
		setPropi2((Int2Property)prop, *(dynamic_cast<ivec2 *>(value)));
		break;
	case Enums::IVEC3:
		setPropi3((Int3Property)prop, *(dynamic_cast<ivec3 *>(value)));
		break;
	case Enums::IVEC4:
		setPropi4((Int4Property)prop, *(dynamic_cast<ivec4 *>(value)));
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
	case Enums::DOUBLE:
		setPropd((DoubleProperty)prop, dynamic_cast<NauDouble *>(value)->getNumber());
		break;
	case Enums::DVEC2:
		setPropd2((Double2Property)prop, *(dynamic_cast<dvec2 *>(value)));
		break;
	case Enums::DVEC3:
		setPropd3((Double3Property)prop, *(dynamic_cast<dvec3 *>(value)));
		break;
	case Enums::DVEC4:
		setPropd4((Double4Property)prop, *(dynamic_cast<dvec4 *>(value)));
		break;
	case Enums::DMAT4:
		setPropdm4((DMat4Property)prop, *(dynamic_cast<dmat4 *>(value)));
		break;
	case Enums::DMAT3:
		setPropdm3((DMat3Property)prop, *(dynamic_cast<dmat3 *>(value)));
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
	case Enums::IVEC3:
		return isValidi3((Int3Property)prop, *(dynamic_cast<ivec3 *>(value)));
		break;
	case Enums::IVEC4:
		return isValidi4((Int4Property)prop, *(dynamic_cast<ivec4 *>(value)));
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
	case Enums::UIVEC4:
		return isValidui4((UInt4Property)prop, *(dynamic_cast<uivec4 *>(value)));
		break;
	case Enums::BOOL:
		return isValidb((BoolProperty)prop, dynamic_cast<NauInt *>(value)->getNumber() != 0);
		break;
	case Enums::BVEC2:
		return isValidb2((Bool2Property)prop, *(dynamic_cast<bvec2 *>(value)));
		break;
	case Enums::BVEC3:
		return isValidb3((Bool3Property)prop, *(dynamic_cast<bvec3 *>(value)));
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
	case Enums::MAT2:
		return isValidm2((Mat2Property)prop, *(dynamic_cast<mat2 *>(value)));
		break;
	case Enums::DOUBLE:
		return isValidd((DoubleProperty)prop, dynamic_cast<NauDouble *>(value)->getNumber());
		break;
	case Enums::DVEC2:
		return isValidd2((Double2Property)prop, *(dynamic_cast<dvec2 *>(value)));
		break;
	case Enums::DVEC3:
		return isValidd3((Double3Property)prop, *(dynamic_cast<dvec3 *>(value)));
		break;
	case Enums::DVEC4:
		return isValidd4((Double4Property)prop, *(dynamic_cast<dvec4 *>(value)));
		break;
	case Enums::DMAT4:
		return isValiddm4((DMat4Property)prop, *(dynamic_cast<dmat4 *>(value)));
		break;
	case Enums::DMAT3:
		return isValiddm3((DMat3Property)prop, *(dynamic_cast<dmat3 *>(value)));
		break;
	case Enums::DMAT2:
		return isValiddm2((DMat2Property)prop, *(dynamic_cast<dmat2 *>(value)));
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

	m_Attribs = &attribs;
	initArrays();
}


void
AttributeValues::initArrays() {

	m_Attribs->initAttribInstanceStringArray(m_StringProps);
	m_Attribs->initAttribInstanceEnumArray(m_EnumProps);
	m_Attribs->initAttribInstanceIntArrayArray(m_IntArrayProps);
	m_Attribs->initAttribInstanceIntArray(m_IntProps);
	m_Attribs->initAttribInstanceInt2Array(m_Int2Props);
	m_Attribs->initAttribInstanceInt3Array(m_Int3Props);
	m_Attribs->initAttribInstanceInt4Array(m_Int4Props);
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

	m_Attribs->initAttribInstanceDVec4Array(m_Double4Props);
	m_Attribs->initAttribInstanceDVec3Array(m_Double3Props);
	m_Attribs->initAttribInstanceDVec2Array(m_Double2Props);
	m_Attribs->initAttribInstanceDoubleArray(m_DoubleProps);
	m_Attribs->initAttribInstanceDMat4Array(m_DMat4Props);
	m_Attribs->initAttribInstanceDMat3Array(m_DMat3Props);
}


AttribSet *
AttributeValues::getAttribSet() {

	return m_Attribs;
}


AttributeValues::AttributeValues() {

}


AttributeValues::AttributeValues(const AttributeValues &to) {

	m_StringProps = to.m_StringProps;
	m_IntArrayProps = to.m_IntArrayProps;
	m_EnumProps = to.m_EnumProps;
	m_IntProps = to.m_IntProps;
	m_Int2Props = to.m_Int2Props;
	m_Int3Props = to.m_Int3Props;
	m_Int4Props = to.m_Int4Props;
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

	m_DoubleProps = to.m_DoubleProps;
	m_Double2Props = to.m_Double2Props;
	m_Double3Props = to.m_Double3Props;
	m_Double4Props = to.m_Double4Props;
	m_DMat3Props = to.m_DMat3Props;
	m_DMat4Props = to.m_DMat4Props;

	m_Attribs = to.m_Attribs;
}


AttributeValues::~AttributeValues() {

}



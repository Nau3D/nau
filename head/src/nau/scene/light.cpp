#include <nau/scene/light.h>
#include <nau/errors.h>

using namespace nau;


bool
Light::Init() {

	// VEC4
	Attribs.add(Attribute(POSITION, "POSITION", Enums::DataType::VEC4, false, new vec4(0.0f, 0.0f, 0.0f, 1.0f)));
	Attribs.add(Attribute(DIRECTION, "DIRECTION", Enums::DataType::VEC4, false, new vec4(0.0f, 0.0f, 1.0f, 0.0f)));
	Attribs.add(Attribute(NORMALIZED_DIRECTION, "NORMALIZED_DIRECTION", Enums::DataType::VEC4,true, new vec4(0.0f, 0.0f, 1.0f, 0.0f)));
	Attribs.add(Attribute(COLOR, "COLOR", Enums::DataType::VEC4, false, new vec4(1.0f, 1.0f, 1.0f, 1.0f)));
	Attribs.add(Attribute(AMBIENT, "AMBIENT", Enums::DataType::VEC4, false, new vec4(0.2f, 0.2f, 0.2f, 0.2f)));
	Attribs.add(Attribute(SPECULAR, "SPECULAR", Enums::DataType::VEC4, false, new vec4(0.0f, 0.0f, 0.0f, 0.0f)));
	// FLOAT
	Attribs.add(Attribute(SPOT_EXPONENT, "SPOT_EXPONENT", Enums::DataType::FLOAT, false, new float(0.0f)));
	Attribs.add(Attribute(SPOT_CUTOFF, "SPOT_CUTOFF", Enums::DataType::FLOAT, false, new float (180.0f)));
	Attribs.add(Attribute(CONSTANT_ATT, "CONSTANT_ATT", Enums::DataType::FLOAT, false, new float(1.0f)));
	Attribs.add(Attribute(LINEAR_ATT, "LINEAR_ATT", Enums::DataType::FLOAT,false, new float(0.0f)));
	Attribs.add(Attribute(QUADRATIC_ATT, "QUADRATIC_ATT", Enums::DataType::FLOAT,false, new float (0.0)));
	// ENUM
	Attribs.add(Attribute(TYPE, "TYPE", Enums::DataType::ENUM, false, new int(DIRECTIONAL)));
	Attribs.listAdd("TYPE", "DIRECTIONAL", DIRECTIONAL);
	Attribs.listAdd("TYPE", "POSITIONAL", POSITIONAL);
	Attribs.listAdd("TYPE", "SPOTLIGHT", SPOT_LIGHT);
	Attribs.listAdd("TYPE", "OMINLIGHT", OMNILIGHT);
	// BOOL
	Attribs.add(Attribute(ENABLED, "ENABLED", Enums::DataType::BOOL, false, new bool(true)));
	//INT
	Attribs.add(Attribute(ID, "ID", Enums::DataType::INT,true, new int(-1)));

	return true;
}


AttribSet Light::Attribs;
bool Light::Inited = Init();


Light::Light (std::string &name) 
{
	m_Name = name;

	setDefault();
}


Light::~Light(void)
{
}


void
Light::setDefault()
{
	Attribs.initAttribInstanceEnumArray(m_EnumProps);
	Attribs.initAttribInstanceFloatArray(m_FloatProps);
	Attribs.initAttribInstanceVec4Array(m_Float4Props);
	Attribs.initAttribInstanceBoolArray(m_BoolProps);
	Attribs.initAttribInstanceIntArray(m_IntProps);
}


void
Light::init (nau::math::vec3 position,
				  nau::math::vec3 direction,
				  nau::math::vec4 color, int enabled, LightType type)
{
	setDefault();
	m_Float4Props[POSITION].set(position.x, position.y, position.z, 1.0f);
	m_Float4Props[DIRECTION].set(direction.x, direction.y, direction.z, 0.0f);
	m_Float4Props[NORMALIZED_DIRECTION].set(direction.x, direction.y, direction.z, 0.0f);
	m_Float4Props[NORMALIZED_DIRECTION].normalize();
	m_Float4Props[COLOR].set (color.x, color.y, color.z, color.w);
	m_BoolProps[ENABLED] = (enabled!=0);
	m_EnumProps[TYPE] = type;
}


std::string
Light::getType() {

	return("LIGHT");
}

void 
Light::setProp(int prop, Enums::DataType type, void *value) {

	switch (type) {

		case Enums::FLOAT:
			if (prop < COUNT_FLOATPROPERTY)
				setProp((FloatProperty)prop, *(float *)value);
			else
				m_FloatProps[prop] = *(float *)value;
			break;
		case Enums::VEC4:
			if (prop < COUNT_FLOAT4PROPERTY)
				setProp((Float4Property)prop, *(vec4 *)value);
			else
				m_Float4Props[prop].set((vec4 *)value);
			break;
		case Enums::INT:
			if (prop < COUNT_INTPROPERTY)
				setProp((IntProperty)prop, *(int *)value);
			else
				m_IntProps[prop] = *(int *)value;
			break;
	}
}


void *
Light::getProp(int prop, Enums::DataType type) {

	switch (type) {

	case Enums::FLOAT:
		assert(m_FloatProps.count(prop) > 0);
		return(&(m_FloatProps[prop]));
		break;
	case Enums::VEC4:
		assert(m_Float4Props.count(prop) > 0);
		return(&(m_Float4Props[prop]));
		break;
	case Enums::INT:
		assert(m_IntProps.count(prop) > 0);
		return(&(m_IntProps[prop]));
		break;
		
	}
	return NULL;
}


void
Light::setProp(FloatProperty prop, float value) {

	float final = value;

	switch (prop) {
		case SPOT_EXPONENT:
			if (value < 0.0f || value > 128.0f)
				final = 0.0f;
			break;
		case SPOT_CUTOFF:
			if (value != 180.0f || value < 0.0f || value > 90.0f)
				final = 180.0f;
			break;
		case CONSTANT_ATT:
		case LINEAR_ATT:
		case QUADRATIC_ATT:
			if (value < 0)
				final = 0;
			break;
	}
	m_FloatProps[prop] = final;
}


void
Light::setProp(Float4Property prop, float x, float y, float z, float w){

	vec4 v;

	v.set(x,y,z,w);

	switch(prop) {
		case POSITION:
			v.w = 1;
			break;
		case DIRECTION:
			v.w = 0;
//			m_Float4Props[DIRECTION].set(v);
			m_Float4Props[NORMALIZED_DIRECTION].set(v);
			m_Float4Props[NORMALIZED_DIRECTION].normalize();
			break;
		case NORMALIZED_DIRECTION:
			v.w = 0;
			v.normalize();
			m_Float4Props[DIRECTION].set(v);
//			m_Float4Props[NORMALIZED_DIRECTION].set(v);
			break;
	}
	m_Float4Props[prop].set(v);
}


void
Light::setProp(Float4Property prop, const vec4& values){

	setProp(prop, values.x, values.y, values.z, values.w);
}


void
Light::setProp(BoolProperty prop, bool value){

	m_BoolProps[prop] = value;
}


void 
Light::setProp(IntProperty prop, int value) {

	m_IntProps[prop] = value;
}


void
Light::setProp(EnumProperty prop, int value){

	if (prop == TYPE && value > COUNT_LIGHTTYPE)
			
		NAU_THROW("Invalid Light Type Value %d", value);

	m_EnumProps[prop] = value;
}


float
Light::getPropf(FloatProperty prop) {

	return m_FloatProps[prop];
}


const vec4& 
Light::getPropf4(Float4Property prop) {

	return m_Float4Props[prop];
}


bool 
Light::getPropb(BoolProperty prop) {

	return m_BoolProps[prop];
}


int 
Light::getPrope(EnumProperty prop) {

	return m_EnumProps[prop];
}


int 
Light::getPropi(IntProperty prop) {

	return m_IntProps[prop];
}



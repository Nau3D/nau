#include "nau/material/programValue.h"

#include "nau.h"

#include "nau/system/TextUtil.h"

using namespace nau::material;
using namespace nau::math;
using namespace nau::render;
using namespace nau::scene;
using namespace nau::system;


ProgramValue::ProgramValue () {
}


ProgramValue::ProgramValue (std::string name, std::string type,
		std::string context,std::string valueof, int id, bool inSpecML) : 
		m_Cardinality (0) {

	int attr;
	nau::Enums::DataType dt;
	m_Values = NULL;
//	m_IntValue = NULL;
	m_InSpecML = inSpecML;
	m_TypeString = type;

	m_Name = name;
	m_Context = context;
	m_ValueOfString = valueof;
	m_Id = id;
	std::string what;

	//if (type == "CURRENT")
	//	what = context;
	//else
		what = type;

	AttribSet *attrSet = NAU->getAttribs(what);
	if (attrSet == NULL)
		NAU_THROW("Exception creating a program value. name=%s, type=%s, context=%s, component=%s, int=%d", name.c_str(), type.c_str(), context.c_str(), valueof.c_str(), id);
	
	attrSet->getPropTypeAndId(valueof, &dt, &attr);
	m_ValueOf = attr;
	m_ValueType = dt;
	m_Cardinality = Enums::getCardinality(dt);
	void *def = (void *)attrSet->get(attr,dt)->getDefault().get();
	if (def != NULL)
		m_Values = def;
	else
		m_Values = (void *)malloc(Enums::getSize(dt));
}


ProgramValue::~ProgramValue () {

	//if (m_Value) {
	//	free (m_Value);
	//	m_Value = NULL;
	//}
	//if (m_IntValue) {
	//	free (m_IntValue);
	//	m_IntValue = NULL;
	//}
}


void 
ProgramValue::clone(ProgramValue &pv) 
{
//	m_Type = pv.m_Type;
	m_Name = pv.m_Name;
	m_Id = pv.m_Id;
	m_ValueOf = pv.m_ValueOf;
	m_ValueOfString = pv.m_ValueOfString;
	m_ValueType = pv.m_ValueType;
	m_Context = pv.m_Context;
	m_Cardinality = pv.m_Cardinality;
	m_InSpecML = pv.m_InSpecML;
	m_Values = (void *)malloc(Enums::getSize(m_ValueType));
	memcpy(m_Values, pv.m_Values, Enums::getSize(m_ValueType));
}


const std::string &
ProgramValue::getType() {

	return m_TypeString;
}


int 
ProgramValue::getId() {

	return m_Id;
}


void
ProgramValue::setId(int id) {

	m_Id = id;
}


const std::string &
ProgramValue::getContext() {

	return(m_Context);
}


const std::string &
ProgramValue::getValueOf() {

	return m_ValueOfString;
}


bool
ProgramValue::isInSpecML() {

	return m_InSpecML;
}
	

void 
ProgramValue::setContext(std::string s) {

	m_Context = s;
}


const std::string &
ProgramValue::getName() {

	return(m_Name);
}


int
ProgramValue::getCardinality (void) {

   return m_Cardinality;
}


void*
ProgramValue::getValues (void) {

	AttributeValues *attr = NULL;
	if (m_Context != "CURRENT") {
//	if (m_TypeString != "CURRENT") {
		attr = NAU->getObjectAttributes(m_TypeString, m_Context, m_Id);
	}
	else {
		attr = NAU->getCurrentObjectAttributes(m_TypeString, m_Id);
	}

	if (attr != NULL) {
		m_Values = attr->getProp(m_ValueOf, m_ValueType);
	}
	// otherwise m_Values will have the default value
	return m_Values;
}


int 
ProgramValue::getSemanticValueOf() {

	return m_ValueOf;
}

void 
ProgramValue::setSemanticValueOf(int s) {

	m_ValueOf = s;
}


nau::Enums::DataType
ProgramValue::getValueType() {

	return m_ValueType;
}


void 
ProgramValue::setValueType(nau::Enums::DataType s) {

	m_ValueType = s;
}


void
ProgramValue::setValueOfUniform(void *values) {

	memcpy(m_Values, values, Enums::getSize(m_ValueType));
}


void
ProgramValue::setLoc(int l) {

	m_Loc = l;
}


int
ProgramValue::getLoc() {

	return m_Loc;
}

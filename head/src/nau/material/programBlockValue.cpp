#include "nau/material/programBlockValue.h"

#include "nau/material/uniformBlockManager.h"

#include "nau.h"

#include "nau/system/TextUtil.h"

using namespace nau::material;
using namespace nau::math;
using namespace nau::render;
using namespace nau::scene;
using namespace nau::system;


ProgramBlockValue::ProgramBlockValue () {
}


ProgramBlockValue::ProgramBlockValue (std::string name, std::string block, std::string type, std::string context,std::string valueof, int id, bool inSpecML) : m_Cardinality (0) {

	IUniformBlock *aBlock = UNIFORMBLOCKMANAGER->getBlock(name);
	//if (aBlock == NULL) {
	//	NAU_THROW("Uniform Block %s is not defined", block.c_str());
	//}
	//if (!aBlock->hasUniform(name))
	//	NAU_THROW("Uniform Block %s does not hava a uniform named %s", block.c_str(), name.c_str());

	//if (!aBlock->getUniformType(name) != Enums::getType(type))
	//	NAU_THROW("Uniform Block %s, uniform %s - type does not match", block.c_str(), name.c_str());

	int attr;
	nau::Enums::DataType dt;
	m_InSpecML = inSpecML;
	m_TypeString = type;
	m_ValueOfString = valueof;

	m_Name = name;
	m_Block = block;
	m_Context = context;
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
	void *def = (void *)attrSet->get(attr, dt)->getDefault().get();
	if (def != NULL)
		m_Values = def;
	else
		m_Values = (void *)malloc(Enums::getSize(dt));
}


ProgramBlockValue::~ProgramBlockValue () {

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
ProgramBlockValue::clone(ProgramBlockValue &pv) 
{
//	m_Type = pv.m_Type;
	m_Name = pv.m_Name;
	m_Block = pv.m_Block;
	m_Id = pv.m_Id;
	m_ValueOf = pv.m_ValueOf;
	m_ValueType = pv.m_ValueType;
	m_Context = pv.m_Context;
	m_Cardinality = pv.m_Cardinality;
	m_InSpecML = pv.m_InSpecML;
	m_Values = (void *)malloc(Enums::getSize(m_ValueType));
	memcpy(m_Values, pv.m_Values, Enums::getSize(m_ValueType));
}


const std::string &
ProgramBlockValue::getType() {

	return m_TypeString;
}

int 
ProgramBlockValue::getId() {

	return m_Id;
}


void
ProgramBlockValue::setId(int id) {

	m_Id = id;
}


const std::string &
ProgramBlockValue::getContext() {

	return(m_Context);
}

const std::string &
ProgramBlockValue::getValueOf() {

	return m_ValueOfString;
}


bool
ProgramBlockValue::isInSpecML() {

	return m_InSpecML;
}
	

void 
ProgramBlockValue::setContext(std::string s) {

	m_Context = s;
}


const std::string &
ProgramBlockValue::getName() {

	return(m_Name);
}


int
ProgramBlockValue::getCardinality (void) {

   return m_Cardinality;
}


void*
ProgramBlockValue:: getValues (void) {

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
ProgramBlockValue::getSemanticValueOf() {

	return m_ValueOf;
}

void 
ProgramBlockValue::setSemanticValueOf(int s) {

	m_ValueOf = s;
}


nau::Enums::DataType
ProgramBlockValue::getValueType() {

	return m_ValueType;
}


void 
ProgramBlockValue::setValueType(nau::Enums::DataType s) {

	m_ValueType = s;
}


void
ProgramBlockValue::setValueOfUniform(void *values) {

	memcpy(m_Values, values, Enums::getSize(m_ValueType));
}


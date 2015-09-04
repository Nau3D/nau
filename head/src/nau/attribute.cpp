#include "nau/attribute.h"

#include "nau/render/iRenderer.h"

#include <vector>

using namespace nau::render;

std::vector<std::string> Attribute::m_DummyVS;

Attribute::Attribute() : m_Id(-1), m_Default(NULL), m_RangeDefined(false), m_ListDefined(false) {

}


Attribute::Attribute(int id, std::string name, Enums::DataType type, 
	bool readOnlyFlag, void *defaultV,
	void *min, void *max, IAPISupport::APIFeatureSupport requires) :
	m_Id(id), m_Name(name), m_Type(type), m_ReadOnlyFlag(readOnlyFlag), m_Default(NULL), m_Min(NULL), m_Max(NULL),
		m_ListDefined(false), m_RangeDefined(false),
		m_Requires(requires){
		
	int s = Enums::getSize(m_Type);

	if (min != NULL) {
		m_RangeDefined = true;
		m_Min = malloc(s);
		memcpy(m_Min, min, Enums::getSize(m_Type));
	}

	if (max != NULL) {
		m_RangeDefined = true;
		m_Max = malloc(s);
		memcpy(m_Max, max, Enums::getSize(m_Type));
	}

	if (defaultV) {
		m_Default = malloc(s);
		memcpy(m_Default, defaultV, s);
	}
	else {
		m_Default = Enums::getDefaultValue(m_Type);
				
	}
}



Attribute::~Attribute() {
	// can't free this memory because it may be in use in other attribute ?
};


std::vector<std::string> &
Attribute::getValidUserAttrTypes() {

	m_DummyVS.clear();
	m_DummyVS.push_back("INT");
	m_DummyVS.push_back("FLOAT");
	m_DummyVS.push_back("VEC4");

	return m_DummyVS;
}


bool 
Attribute::isValidUserAttrType(std::string s) {

	if (s == "FLOAT" || s == "INT" || s == "VEC4")
		return true;
	else
		return false;
}


std::string &
Attribute::getName() {

	return m_Name;
}


void *
Attribute::getDefault() {

	return m_Default;
}


void 
Attribute::setRequirement(IAPISupport::APIFeatureSupport req) {

	m_Requires = req;
}


IAPISupport::APIFeatureSupport 
Attribute::getRequirement() {

	return m_Requires;
}


void 
Attribute::setRange(void *min, void *max) {
				
	assert(m_Type != Enums::STRING);

	if (min == NULL && max == NULL)
		return;
			
	m_RangeDefined = true;

	if (min != NULL) {
		m_Min = malloc(Enums::getSize(m_Type));
		memcpy(m_Min, min, Enums::getSize(m_Type));
	}
	else
		m_Min = NULL;

	if (max != NULL) {
		m_Max = malloc(Enums::getSize(m_Type));
		memcpy(m_Max, max, Enums::getSize(m_Type));
	}
	else
		m_Max = NULL;
};


Enums::DataType 
Attribute::getType() {

	return m_Type;
}


bool 
Attribute::getReadOnlyFlag() {

	return m_ReadOnlyFlag;
}


int 
Attribute::getId() {

	return m_Id;
}


void 
Attribute::listAdd(std::string name, int id, IAPISupport::APIFeatureSupport requires) {
		
	m_ListDefined = true;
	m_ListValues.push_back(id);
	m_ListString.push_back(name);
	m_ListRequire.push_back(requires);
};

		 
bool 
Attribute::isValid(std::string value) {
		
	if (m_ListDefined) {				
		for ( unsigned int i = 0; i < m_ListString.size(); ++i) {
			if (m_ListString[i] == value) {
				if (APISupport->apiSupport(m_ListRequire[i]))
					return true;
				else return false;
			}
		}
		return false;
	}
	return false;
};


bool 
Attribute::getRangeDefined() {

	return m_RangeDefined; 
};


bool 
Attribute::getListDefined() {

	return m_ListDefined; 
};


void *
Attribute::getMax() {
		
	return m_Max;
}


void *
Attribute::getMin() {

	return m_Min;
}


int 
Attribute::getOptionValue(std::string &s) {
		
	for(unsigned int i = 0 ; i < m_ListString.size(); ++i) {	
		if (m_ListString[i] == s)
			return m_ListValues[i];
	}
	return -1;
}


std::string &
Attribute::getOptionString(int v) {
		
	for(unsigned int i = 0 ; i < m_ListValues.size(); ++i) {		
		if (m_ListValues[i] == v)
			return m_ListString[i];
	}
	return m_DummyS;
}


bool 
Attribute::isValid(int v) {

	for (unsigned int i = 0; i < m_ListValues.size(); ++i) {
			if (m_ListValues[i] == v) {
				if (APISupport->apiSupport(m_ListRequire[i]))
					return true;
				else return false;
			}
	}
	return false;
}


const std::vector<std::string> &
Attribute::getOptionStringList() {

	return m_ListString;
}


void 
Attribute::getOptionStringListSupported(std::vector<std::string> *result) {

	result->clear();
	for (unsigned int i = 0; i < m_ListString.size(); ++i) {

		if (APISupport->apiSupport(m_ListRequire[i]))
			result->push_back(m_ListString[i]);
	}

}


// -------------------------------------------------------------------------------------------
//    Attribute Set 
// -------------------------------------------------------------------------------------------



AttribSet::AttribSet() : m_NextFreeID(USER_ATTRIBS), m_DummyS("") { 
	
	m_Dummy.m_Name = "NO_ATTR"; 
};


AttribSet::~AttribSet() {};

int 
AttribSet::getNextFreeID() {
		
	return m_NextFreeID++;
}


void 
AttribSet::deleteUserAttributes() {

	std::map<std::string, Attribute>::iterator iter;
	iter = m_Attributes.begin();
	while (iter != m_Attributes.end()) {

		if (iter->second.m_Id >= USER_ATTRIBS)
			m_Attributes.erase(iter++);
		else
			++iter;
	}
}

void 
AttribSet::add(Attribute a) {
			
	if (a.m_Id != -1) {
		m_Attributes[a.m_Name] = a;
	}
	Enums::DataType dt = a.getType();
	if (mDataTypeCounter.count(dt))
		++mDataTypeCounter[dt];
	else
		mDataTypeCounter[dt] = 1;
}


int 
AttribSet::getDataTypeCount(Enums::DataType dt) {

	if (mDataTypeCounter.count(dt))
		return mDataTypeCounter[dt];
	else
		return 0;
}

		
const Attribute &
AttribSet::get(std::string name) {

	if (m_Attributes.find(name) != m_Attributes.end()) 

		return(m_Attributes[name]);
	else
		return m_Dummy;

}


const Attribute &
AttribSet::get(int id, Enums::DataType dt) {

	std::map<std::string, Attribute>::iterator it;
	it = m_Attributes.begin();
	for (; it != m_Attributes.end(); ++it) {

		if (it->second.m_Id == id && it->second.m_Type == dt)
			return (it->second);
	}
	return m_Dummy;
}


int 
AttribSet::getID(std::string name) {

	if (m_Attributes.find(name) != m_Attributes.end()) 

		return(m_Attributes[name].m_Id);
	else
		return -1;

}


const std::map<std::string, Attribute> &
AttribSet::getAttributes() {

	return (m_Attributes);
}


const std::string &
AttribSet::getName(int id, Enums::DataType dt) {

	std::map<std::string, Attribute>::iterator it;
	it = m_Attributes.begin();
	for ( ; it != m_Attributes.end(); ++it) {

		if (it->second.m_Id == id && it->second.m_Type == dt)
			return (it->first);
	}
	return m_DummyS;
}


void 
AttribSet::getPropTypeAndId(std::string &s, nau::Enums::DataType *dt, int *id) {
			
	Attribute a = get(s);
	*id = a.m_Id;

	if (a.m_Id != -1) {

		*dt = a.m_Type;
	}
}


const std::vector<std::string> &
AttribSet::getListString(int id) {
		
	std::map<std::string, Attribute>::iterator it;
	it = m_Attributes.begin();
	for ( ; it != m_Attributes.end(); ++it) {

		if (it->second.m_Id == id && it->second.m_Type == Enums::DataType::ENUM)
			return (it->second.m_ListString);
	}
	return m_DummyVS;
}


const std::vector<int> &
AttribSet::getListValues(int id) {
		
	std::map<std::string, Attribute>::iterator it;
	it = m_Attributes.begin();
	for ( ; it != m_Attributes.end(); ++it) {

		if (it->second.m_Id == id && it->second.m_Type == Enums::DataType::ENUM)
			return (it->second.m_ListValues);
	}
	return m_DummyVI;
}


std::string 
AttribSet::getListStringOp(std::string s, int prop) {
		
	Attribute a = get(s);
	return (a.getOptionString(prop));
}


std::string 
AttribSet::getListStringOp(int id, int prop) {
		
	std::map<std::string, Attribute>::iterator it;
	it = m_Attributes.begin();
	for ( ; it != m_Attributes.end(); ++it) {

		if (it->second.m_Id == id && it->second.m_Type == Enums::DataType::ENUM)
			return (it->second.getOptionString(prop));
	}
	return m_DummyS;
}


int 
AttribSet::getListValueOp(std::string s, std::string prop) {
		
	Attribute a = get(s);
	return (a.getOptionValue(prop));
}


int 
AttribSet::getListValueOp(int id, std::string prop) {
		
	std::map<std::string, Attribute>::iterator it;
	it = m_Attributes.begin();
	for ( ; it != m_Attributes.end(); ++it) {

		if (it->second.m_Id == id && it->second.m_Type == Enums::DataType::ENUM)
			return (it->second.getOptionValue(prop));
	}
	return -1;
}


void 
AttribSet::listAdd(std::string attrName, std::string elemS, int elem_Id, IAPISupport::APIFeatureSupport requires) {

	std::map<std::string, Attribute>::iterator it;
	it = m_Attributes.begin();
	for ( ; it != m_Attributes.end(); ++it) {

		if (it->first == attrName) {
			it->second.listAdd(elemS, elem_Id, requires);
			return;
		}
	}
}


bool 
AttribSet::isValid(std::string attr, std::string value) {

	Attribute a = get(attr);
	return a.isValid(value);
}


void 
AttribSet::setDefault(std::string attr, void *value) {
		
	if (m_Attributes.find(attr) != m_Attributes.end()) {
		assert(m_Attributes[attr].m_Type != Enums::STRING);
		if (m_Attributes[attr].m_Type != Enums::STRING) {
			int s = Enums::getSize(m_Attributes[attr].m_Type);
			m_Attributes[attr].m_Default = malloc(s);
			memcpy(m_Attributes[attr].m_Default, value, s);
		}
	}
}


void *
AttribSet::getDefault(int id, Enums::DataType type) {

	std::map<std::string, Attribute>::iterator it;
	it = m_Attributes.begin();
	for ( ; it != m_Attributes.end(); ++it) {

		if (it->second.m_Id == id && it->second.m_Type == type)
			return (it->second.m_Default);
	}
	return NULL;
}


void 
AttribSet::initAttribInstanceIntArray(std::map<int, int> &m) {

	std::map<std::string, Attribute>::iterator it;
	it = m_Attributes.begin();
	for (; it != m_Attributes.end(); ++it) {
		if (it->second.m_Type == Enums::DataType::INT) {

			m[it->second.m_Id] = *(int *)(it->second.m_Default);
		}
	}
}


void
AttribSet::initAttribInstanceInt2Array(std::map<int, ivec2> &m) {

	std::map<std::string, Attribute>::iterator it;
	it = m_Attributes.begin();
	for (; it != m_Attributes.end(); ++it) {
		if (it->second.m_Type == Enums::DataType::IVEC2) {

			m[it->second.m_Id] = *(ivec2 *)(it->second.m_Default);
		}
	}
}


void 
AttribSet::initAttribInstanceEnumArray(std::map<int, int> &m) {

	std::map<std::string, Attribute>::iterator it;
	it = m_Attributes.begin();
	for ( ; it != m_Attributes.end(); ++it) {
		if (it->second.m_Type == Enums::DataType::ENUM) {

				m[it->second.m_Id] = *(int *)(it->second.m_Default);
		}
	}
}


void 
AttribSet::initAttribInstanceUIntArray(std::map<int, unsigned int> &m) {

	std::map<std::string, Attribute>::iterator it;
	it = m_Attributes.begin();
	for ( ; it != m_Attributes.end(); ++it) {
		if (it->second.m_Type == Enums::DataType::UINT) {

				m[it->second.m_Id] = *(unsigned int *)(it->second.m_Default);
		}
	}
}


void 
AttribSet::initAttribInstanceUInt2Array(std::map<int, uivec2> &m) {

	std::map<std::string, Attribute>::iterator it;
	it = m_Attributes.begin();
	for ( ; it != m_Attributes.end(); ++it) {
		if (it->second.m_Type == Enums::DataType::UIVEC2) {

				m[it->second.m_Id] = *(uivec2 *)(it->second.m_Default);
		}
	}
}


void
AttribSet::initAttribInstanceUInt3Array(std::map<int, uivec3> &m) {

	std::map<std::string, Attribute>::iterator it;
	it = m_Attributes.begin();
	for (; it != m_Attributes.end(); ++it) {
		if (it->second.m_Type == Enums::DataType::UIVEC3) {

			m[it->second.m_Id] = *(uivec3 *)(it->second.m_Default);
		}
	}
}


void 
AttribSet::initAttribInstanceFloatArray(std::map<int, float> &m) {

	std::map<std::string, Attribute>::iterator it;
	it = m_Attributes.begin();
	for ( ; it != m_Attributes.end(); ++it) {
		if (it->second.m_Type == Enums::DataType::FLOAT) {

				m[it->second.m_Id] = *(float *)(it->second.m_Default);
		}
	}
}


void 
AttribSet::initAttribInstanceVec4Array(std::map<int, vec4> &m) {

	std::map<std::string, Attribute>::iterator it;
	it = m_Attributes.begin();
	for ( ; it != m_Attributes.end(); ++it) {
		if (it->second.m_Type == Enums::DataType::VEC4) {

				m[it->second.m_Id] = *(vec4 *)(it->second.m_Default);
		}
	}
}


void
AttribSet::initAttribInstanceVec3Array(std::map<int, vec3> &m) {

	std::map<std::string, Attribute>::iterator it;
	it = m_Attributes.begin();
	for (; it != m_Attributes.end(); ++it) {
		if (it->second.m_Type == Enums::DataType::VEC3) {

			m[it->second.m_Id] = *(vec3 *)(it->second.m_Default);
		}
	}
}

void 
AttribSet::initAttribInstanceVec2Array(std::map<int, vec2> &m) {

	std::map<std::string, Attribute>::iterator it;
	it = m_Attributes.begin();
	for (; it != m_Attributes.end(); ++it) {
		if (it->second.m_Type == Enums::DataType::VEC2) {

			m[it->second.m_Id] = *(vec2 *)(it->second.m_Default);
		}
	}
}

void 
AttribSet::initAttribInstanceMat4Array(std::map<int, mat4> &m) {

	std::map<std::string, Attribute>::iterator it;
	it = m_Attributes.begin();
	for (; it != m_Attributes.end(); ++it) {
		if (it->second.m_Type == Enums::DataType::MAT4) {
			m[it->second.m_Id] = *(mat4 *)(it->second.m_Default);
		}
	}
}

void 
AttribSet::initAttribInstanceMat3Array(std::map<int, mat3> &m) {

	std::map<std::string, Attribute>::iterator it;
	it = m_Attributes.begin();
	for (; it != m_Attributes.end(); ++it) {
		if (it->second.m_Type == Enums::DataType::MAT3) {
			m[it->second.m_Id] = *(mat3 *)(it->second.m_Default);
		}
	}
}


void 
AttribSet::initAttribInstanceBvec4Array(std::map<int, bvec4> &m) {

	std::map<std::string, Attribute>::iterator it;
	it = m_Attributes.begin();
	for ( ; it != m_Attributes.end(); ++it) {
		if (it->second.m_Type == Enums::DataType::BVEC4) {

				m[it->second.m_Id] = *(bvec4 *)(it->second.m_Default);
		}
	}
}


void 
AttribSet::initAttribInstanceBoolArray(std::map<int, bool> &m) {

	std::map<std::string, Attribute>::iterator it;
	it = m_Attributes.begin();
	for ( ; it != m_Attributes.end(); ++it) {
		if (it->second.m_Type == Enums::DataType::BOOL) {

				m[it->second.m_Id] = *(bool *)(it->second.m_Default);
		}
	}
}


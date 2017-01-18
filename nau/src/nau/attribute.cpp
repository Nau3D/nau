#include "nau/attribute.h"

#include "nau.h"
#include "nau/render/iRenderer.h"

#include <vector>

using namespace nau::render;

std::vector<std::string> Attribute::m_DummyVS;


Attribute::Attribute() : m_Id(-1), 
			m_RangeDefined(false), m_ListDefined(false) {

}


Attribute::Attribute(unsigned int id, std::string name, Enums::DataType type,
	bool readOnlyFlag, Data *defaultV,
	Data *min, Data *max,
	IAPISupport::APIFeatureSupport requires,
	Semantics sem) :
	m_Id(id), m_Name(name), m_Type(type), m_ReadOnlyFlag(readOnlyFlag),
	m_ListDefined(false), m_RangeDefined(false), m_Semantics(sem),
	m_Requires(requires) {

	if (min != NULL) {
		m_RangeDefined = true;
		m_Min = std::shared_ptr<Data>(min);
	}

	if (max != NULL) {
		m_RangeDefined = true;
		m_Max = std::shared_ptr<Data>(max);
	}

	if (defaultV != NULL) {
		m_Default = std::shared_ptr<Data>(defaultV);
	}
	else {
		m_Default = std::shared_ptr<Data>(Enums::getDefaultValue(m_Type));
	}
}


nau::Attribute::Attribute(unsigned int id, std::string name, std::string objType, 
							bool readOnlyFlag, bool mustExist, std::string defaults):
	m_Id(id),
	m_Name(name),
	m_ObjType(objType),
	m_ReadOnlyFlag(readOnlyFlag),
	m_MustExist(mustExist),
	m_DefaultS(defaults),
	m_Type(Enums::STRING)
	{

}


Attribute::~Attribute() {

}


Attribute::Attribute(const Attribute & source):
	m_Id(source.m_Id), m_Name(source.m_Name), m_Type(source.m_Type), m_ObjType(source.m_ObjType),
	m_ReadOnlyFlag(source.m_ReadOnlyFlag), m_Requires(source.m_Requires), m_MustExist(source.m_MustExist),
	m_Semantics(source.m_Semantics), m_RangeDefined(source.m_RangeDefined) {

	if (source.m_Max) {
		m_Max = std::shared_ptr<Data>(source.m_Max->clone());
	}
	if (source.m_Min) {
		m_Min = std::shared_ptr<Data>(source.m_Min->clone());
	}
	if (source.m_Default) {
		m_Default = std::shared_ptr<Data>(source.m_Default->clone());
	}
}


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


void
Attribute::setDefault(Data &d) {

	m_Default.reset(d.clone());
}


std::string &
Attribute::getName() {

	return m_Name;
}


shared_ptr<Data> &
Attribute::getDefault() {

	return m_Default;
}


const std::string &
Attribute::getDefaultString() {

	return m_DefaultS;
}


const std::string &
Attribute::getObjType() {

	return m_ObjType;
}


bool nau::Attribute::getMustExist() {

	return m_MustExist;
}


void 
Attribute::setRequirement(IAPISupport::APIFeatureSupport req) {

	m_Requires = req;
}


IAPISupport::APIFeatureSupport 
Attribute::getRequirement() {

	return m_Requires;
}


Enums::DataType 
Attribute::getType() {

	return m_Type;
}


bool 
Attribute::getReadOnlyFlag() {

	return m_ReadOnlyFlag;
}


Attribute::Semantics
Attribute::getSemantics() {

	return m_Semantics;
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
}

		 
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
}


bool 
Attribute::getRangeDefined() {

	return m_RangeDefined; 
}


bool 
Attribute::getListDefined() {

	return m_ListDefined; 
}


std::shared_ptr<Data> &
Attribute::getMax() {
		
	return m_Max;
}


std::shared_ptr<Data> &
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
Attribute::getOptionListSupported(std::vector<int> *result) {

	result->clear();
	for (unsigned int i = 0; i < m_ListValues.size(); ++i) {

		if (APISupport->apiSupport(m_ListRequire[i]))
			result->push_back(m_ListValues[i]);
	}
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



AttribSet::AttribSet() : m_NextFreeID(USER_ATTRIBS) { 
	
	m_Dummy = std::unique_ptr<Attribute>(new Attribute);
	m_Dummy->m_Name = "NO_ATTR"; 
};


AttribSet::~AttribSet() {};


unsigned int 
AttribSet::getNextFreeID() {
		
	return m_NextFreeID++;
}


void 
AttribSet::deleteUserAttributes() {

	std::map<std::string, std::unique_ptr<Attribute>>::iterator iter;
	iter = m_Attributes.begin();
	while (iter != m_Attributes.end()) {

		if (iter->second->m_Id >= USER_ATTRIBS)
			m_Attributes.erase(iter++);
		else
			++iter;
	}
}


void 
AttribSet::add(Attribute a) {
			
	if (a.m_Id != -1) {

		if (0 == m_Attributes.count(a.getName())) { // already exists

			Enums::DataType dt = a.getType();
			if (mDataTypeCounter.count(dt))
				++mDataTypeCounter[dt];
			else
				mDataTypeCounter[dt] = 1;
		}
		m_Attributes[a.m_Name] = std::unique_ptr<Attribute>(new Attribute(a));
	}
}


int 
AttribSet::getDataTypeCount(Enums::DataType dt) {

	if (mDataTypeCounter.count(dt))
		return mDataTypeCounter[dt];
	else
		return 0;
}

		
std::unique_ptr<Attribute> &
AttribSet::get(std::string name) {

	if (m_Attributes.find(name) != m_Attributes.end()) 

		return(m_Attributes[name]);
	else
		return m_Dummy;

}


std::unique_ptr<Attribute> &
AttribSet::get(int id, Enums::DataType dt) {

	std::map<std::string, std::unique_ptr<Attribute>>::iterator it;
	it = m_Attributes.begin();
	for (; it != m_Attributes.end(); ++it) {

		if (it->second->m_Id == id && it->second->m_Type == dt)
			return (it->second);
	}
	
	return m_Dummy;
}


int 
AttribSet::getID(std::string name) {

	if (m_Attributes.find(name) != m_Attributes.end()) 

		return(m_Attributes[name]->m_Id);
	else
		return -1;

}


std::map<std::string, std::unique_ptr<Attribute>> &
AttribSet::getAttributes() {

	return (m_Attributes);
}


const std::string &
AttribSet::getName(int id, Enums::DataType dt) {

	std::map<std::string, std::unique_ptr<Attribute>>::iterator it;
	it = m_Attributes.begin();
	for ( ; it != m_Attributes.end(); ++it) {

		if (it->second->m_Id == id && it->second->m_Type == dt)
			return (it->first);
	}
	return m_DummyS;
}


void 
AttribSet::getPropTypeAndId(const std::string &s, nau::Enums::DataType *dt, int *id) {
			
	std::unique_ptr<Attribute> &a = get(s);
	*id = a->m_Id;

	if (a->m_Id != -1) {

		*dt = a->m_Type;
	}
}


const std::vector<std::string> &
AttribSet::getListString(int id) {
		
	std::map<std::string, std::unique_ptr<Attribute>>::iterator it;
	it = m_Attributes.begin();
	for ( ; it != m_Attributes.end(); ++it) {

		if (it->second->m_Id == id && it->second->m_Type == Enums::DataType::ENUM)
			return (it->second->m_ListString);
	}
	return m_DummyVS;
}


const std::vector<int> &
AttribSet::getListValues(int id) {
		
	std::map<std::string, std::unique_ptr<Attribute>>::iterator it;
	it = m_Attributes.begin();
	for ( ; it != m_Attributes.end(); ++it) {

		if (it->second->m_Id == id && it->second->m_Type == Enums::DataType::ENUM)
			return (it->second->m_ListValues);
	}
	return m_DummyVI;
}


std::string 
AttribSet::getListStringOp(std::string s, int prop) {
		
	std::unique_ptr<Attribute> &a = get(s);
	return (a->getOptionString(prop));
}


std::string 
AttribSet::getListStringOp(int id, int prop) {
		
	std::map<std::string, std::unique_ptr<Attribute>>::iterator it;
	it = m_Attributes.begin();
	for ( ; it != m_Attributes.end(); ++it) {

		if (it->second->m_Id == id && it->second->m_Type == Enums::DataType::ENUM)
			return (it->second->getOptionString(prop));
	}
	return m_DummyS;
}


int 
AttribSet::getListValueOp(std::string s, std::string prop) {
		
	std::unique_ptr<Attribute> &a = get(s);
	return (a->getOptionValue(prop));
}


int 
AttribSet::getListValueOp(int id, std::string prop) {
		
	std::map<std::string, std::unique_ptr<Attribute>>::iterator it;
	it = m_Attributes.begin();
	for ( ; it != m_Attributes.end(); ++it) {

		if (it->second->m_Id == id && it->second->m_Type == Enums::DataType::ENUM)
			return (it->second->getOptionValue(prop));
	}
	return -1;
}


void 
AttribSet::listAdd(std::string attrName, std::string elemS, int elem_Id, IAPISupport::APIFeatureSupport requires) {

	std::map<std::string, std::unique_ptr<Attribute>>::iterator it;
	it = m_Attributes.begin();
	for ( ; it != m_Attributes.end(); ++it) {

		if (it->first == attrName) {
			it->second->listAdd(elemS, elem_Id, requires);
			return;
		}
	}
}


bool 
AttribSet::isValid(std::string attr, std::string value) {

	std::unique_ptr<Attribute> &a = get(attr);
	return a->isValid(value);
}


void 
AttribSet::setDefault(std::string attr, Data &value) {
		
	if (m_Attributes.find(attr) != m_Attributes.end()) {
		assert(m_Attributes[attr]->getType() != Enums::STRING);
		if (m_Attributes[attr]->getType() != Enums::STRING) {
			m_Attributes[attr]->setDefault(value);
			//m_Attributes[attr]->getDefault().reset(Enums::getDefaultValue(m_Attributes[attr]->m_Type));
		}
	}
}


//std::unique_ptr<Data> &
//AttribSet::getDefault(int id, Enums::DataType type) {
//
//	std::map<std::string, std::unique_ptr<Attribute>>::iterator it;
//	it = m_Attributes.begin();
//	for ( ; it != m_Attributes.end(); ++it) {
//
//		if (it->second->m_Id == id && it->second->m_Type == type)
//			return (it->second->m_Default);
//	}
//	return std::unique_ptr<Data>(Enums::get;
//}

//template <typename T>
//void
//AttribSet::initAttribInstanceArray(Enums::DataType dt, std::map<int, T> &m) {
//
//	for (auto& attr : m_Attributes) {
//		if (attr.second->m_Type == dt) {
//
//			m[attr.second->m_Id] = T(it->second->m_Default);
//		}
//	}
//}

void 
AttribSet::initAttribInstanceIntArray(std::map<int, int> &m) {

	for (auto & attr: m_Attributes) {
		if (attr.second->m_Type == Enums::DataType::INT) {
			std::shared_ptr<NauInt> ni = std::dynamic_pointer_cast<NauInt>(attr.second->getDefault());
			m[attr.second->m_Id] = *ni;
		}
	}
}


void
AttribSet::initAttribInstanceInt2Array(std::map<int, ivec2> &m) {

	for (auto & attr : m_Attributes) {
		if (attr.second->m_Type == Enums::DataType::IVEC2) {
			std::shared_ptr<Data> &d = attr.second->getDefault();
			std::shared_ptr<ivec2> ni = std::dynamic_pointer_cast<ivec2>(d);
			m[attr.second->m_Id] = *ni;
		}
	}
}


void
AttribSet::initAttribInstanceInt3Array(std::map<int, ivec3> &m) {

	for (auto & attr : m_Attributes) {
		if (attr.second->m_Type == Enums::DataType::IVEC3) {
			std::shared_ptr<Data> &d = attr.second->getDefault();
			std::shared_ptr<ivec3> ni = std::dynamic_pointer_cast<ivec3>(d);
			m[attr.second->m_Id] = *ni;
		}
	}
}


void
AttribSet::initAttribInstanceInt4Array(std::map<int, ivec4> &m) {

	for (auto & attr : m_Attributes) {
		if (attr.second->m_Type == Enums::DataType::IVEC4) {
			std::shared_ptr<Data> &d = attr.second->getDefault();
			std::shared_ptr<ivec4> ni = std::dynamic_pointer_cast<ivec4>(d);
			m[attr.second->m_Id] = *ni;
		}
	}
}


void
AttribSet::initAttribInstanceIntArrayArray(std::map<int, NauIntArray> &m) {

	for (auto & attr : m_Attributes) {
		if (attr.second->m_Type == Enums::DataType::INTARRAY) {
			m[attr.second->m_Id] = NauIntArray();
		}
	}
}


void
AttribSet::initAttribInstanceEnumArray(std::map<int, int> &m) {

	for (auto & attr : m_Attributes) {
		if (attr.second->m_Type == Enums::DataType::ENUM) {
			std::shared_ptr<Data> &d = attr.second->getDefault();
			std::shared_ptr<NauInt> ni = std::dynamic_pointer_cast<NauInt>(d);
			m[attr.second->m_Id] = *ni;
		}
	}
}


void
AttribSet::initAttribInstanceStringArray(std::map<int, std::string> &m) {

	for (auto & attr : m_Attributes) {
		if (attr.second->m_Type == Enums::DataType::STRING) {
			m[attr.second->m_Id] = "";
		}
	}
}


void
AttribSet::initAttribInstanceUIntArray(std::map<int, unsigned int> &m) {

	for (auto & attr : m_Attributes) {
		if (attr.second->m_Type == Enums::DataType::UINT) {
			std::shared_ptr<Data> &d = attr.second->getDefault();
			std::shared_ptr<NauUInt> ni = std::dynamic_pointer_cast<NauUInt>(d);
			m[attr.second->m_Id] = *ni;
		}
	}
}


void 
AttribSet::initAttribInstanceUInt2Array(std::map<int, uivec2> &m) {

	for (auto & attr : m_Attributes) {
		if (attr.second->m_Type == Enums::DataType::UIVEC2) {
			std::shared_ptr<Data> &d = attr.second->getDefault();
			std::shared_ptr<uivec2> ni = std::dynamic_pointer_cast<uivec2>(d);
			m[attr.second->m_Id] = *ni;
		}
	}
}


void
AttribSet::initAttribInstanceUInt3Array(std::map<int, uivec3> &m) {

	for (auto & attr : m_Attributes) {
		if (attr.second->m_Type == Enums::DataType::UIVEC3) {
			std::shared_ptr<Data> &d = attr.second->getDefault();
			std::shared_ptr<uivec3> ni = std::dynamic_pointer_cast<uivec3>(d);
			m[attr.second->m_Id] = *ni;
		}
	}
}


void
AttribSet::initAttribInstanceUInt4Array(std::map<int, uivec4> &m) {

	for (auto & attr : m_Attributes) {
		if (attr.second->m_Type == Enums::DataType::UIVEC4) {
			std::shared_ptr<Data> &d = attr.second->getDefault();
			std::shared_ptr<uivec4> ni = std::dynamic_pointer_cast<uivec4>(d);
			m[attr.second->m_Id] = *ni;
		}
	}
}


void 
AttribSet::initAttribInstanceFloatArray(std::map<int, float> &m) {

	for (auto & attr : m_Attributes) {
		if (attr.second->m_Type == Enums::DataType::FLOAT) {
			std::shared_ptr<Data> &d = attr.second->getDefault();
			std::shared_ptr<NauFloat> ni = std::dynamic_pointer_cast<NauFloat>(d);
			m[attr.second->m_Id] = *ni;
		}
	}
}


void 
AttribSet::initAttribInstanceVec4Array(std::map<int, vec4> &m) {

	for (auto & attr : m_Attributes) {
		if (attr.second->m_Type == Enums::DataType::VEC4) {
			std::shared_ptr<Data> &d = attr.second->getDefault();
			std::shared_ptr<vec4> ni = std::dynamic_pointer_cast<vec4>(d);
			m[attr.second->m_Id] = *ni;
		}
	}
}


void
AttribSet::initAttribInstanceVec3Array(std::map<int, vec3> &m) {

	for (auto & attr : m_Attributes) {
		if (attr.second->m_Type == Enums::DataType::VEC3) {
			std::shared_ptr<Data> &d = attr.second->getDefault();
			std::shared_ptr<vec3> ni = std::dynamic_pointer_cast<vec3>(d);
			m[attr.second->m_Id] = *ni;
		}
	}
}


void 
AttribSet::initAttribInstanceVec2Array(std::map<int, vec2> &m) {

	for (auto & attr : m_Attributes) {
		if (attr.second->m_Type == Enums::DataType::VEC2) {
			std::shared_ptr<Data> &d = attr.second->getDefault();
			std::shared_ptr<vec2> ni = std::dynamic_pointer_cast<vec2>(d);
			m[attr.second->m_Id] = *ni;
		}
	}
}


void 
AttribSet::initAttribInstanceMat4Array(std::map<int, mat4> &m) {

	for (auto & attr : m_Attributes) {
		if (attr.second->m_Type == Enums::DataType::MAT4) {
			std::shared_ptr<Data> &d = attr.second->getDefault();
			std::shared_ptr<mat4> ni = std::dynamic_pointer_cast<mat4>(d);
			m[attr.second->m_Id] = *ni;
		}
	}
}


void 
AttribSet::initAttribInstanceMat3Array(std::map<int, mat3> &m) {

	for (auto & attr : m_Attributes) {
		if (attr.second->m_Type == Enums::DataType::MAT3) {
			std::shared_ptr<Data> &d = attr.second->getDefault();
			std::shared_ptr<mat3> ni = std::dynamic_pointer_cast<mat3>(d);
			m[attr.second->m_Id] = *ni;
		}
	}
}


void
AttribSet::initAttribInstanceMat2Array(std::map<int, mat2> &m) {

	for (auto & attr : m_Attributes) {
		if (attr.second->m_Type == Enums::DataType::MAT2) {
			std::shared_ptr<Data> &d = attr.second->getDefault();
			std::shared_ptr<mat2> ni = std::dynamic_pointer_cast<mat2>(d);
			m[attr.second->m_Id] = *ni;
		}
	}
}


void
AttribSet::initAttribInstanceDoubleArray(std::map<int, double> &m) {

	for (auto & attr : m_Attributes) {
		if (attr.second->m_Type == Enums::DataType::FLOAT) {
			std::shared_ptr<Data> &d = attr.second->getDefault();
			std::shared_ptr<NauDouble> ni = std::dynamic_pointer_cast<NauDouble>(d);
			m[attr.second->m_Id] = *ni;
		}
	}
}


void
AttribSet::initAttribInstanceDVec4Array(std::map<int, dvec4> &m) {

	for (auto & attr : m_Attributes) {
		if (attr.second->m_Type == Enums::DataType::DVEC4) {
			std::shared_ptr<Data> &d = attr.second->getDefault();
			std::shared_ptr<dvec4> ni = std::dynamic_pointer_cast<dvec4>(d);
			m[attr.second->m_Id] = *ni;
		}
	}
}


void
AttribSet::initAttribInstanceDVec3Array(std::map<int, dvec3> &m) {

	for (auto & attr : m_Attributes) {
		if (attr.second->m_Type == Enums::DataType::DVEC3) {
			std::shared_ptr<Data> &d = attr.second->getDefault();
			std::shared_ptr<dvec3> ni = std::dynamic_pointer_cast<dvec3>(d);
			m[attr.second->m_Id] = *ni;
		}
	}
}


void
AttribSet::initAttribInstanceDVec2Array(std::map<int, dvec2> &m) {

	for (auto & attr : m_Attributes) {
		if (attr.second->m_Type == Enums::DataType::DVEC2) {
			std::shared_ptr<Data> &d = attr.second->getDefault();
			std::shared_ptr<dvec2> ni = std::dynamic_pointer_cast<dvec2>(d);
			m[attr.second->m_Id] = *ni;
		}
	}
}


void
AttribSet::initAttribInstanceDMat4Array(std::map<int, dmat4> &m) {

	for (auto & attr : m_Attributes) {
		if (attr.second->m_Type == Enums::DataType::DMAT4) {
			std::shared_ptr<Data> &d = attr.second->getDefault();
			std::shared_ptr<dmat4> ni = std::dynamic_pointer_cast<dmat4>(d);
			m[attr.second->m_Id] = *ni;
		}
	}
}


void
AttribSet::initAttribInstanceDMat3Array(std::map<int, dmat3> &m) {

	for (auto & attr : m_Attributes) {
		if (attr.second->m_Type == Enums::DataType::DMAT3) {
			std::shared_ptr<Data> &d = attr.second->getDefault();
			std::shared_ptr<dmat3> ni = std::dynamic_pointer_cast<dmat3>(d);
			m[attr.second->m_Id] = *ni;
		}
	}
}


void
AttribSet::initAttribInstanceDMat2Array(std::map<int, dmat2> &m) {

	for (auto & attr : m_Attributes) {
		if (attr.second->m_Type == Enums::DataType::DMAT2) {
			std::shared_ptr<Data> &d = attr.second->getDefault();
			std::shared_ptr<dmat2> ni = std::dynamic_pointer_cast<dmat2>(d);
			m[attr.second->m_Id] = *ni;
		}
	}
}


void
AttribSet::initAttribInstanceBoolArray(std::map<int, bool> &m) {

	for (auto & attr : m_Attributes) {
		if (attr.second->m_Type == Enums::DataType::BOOL) {
			std::shared_ptr<Data> &d = attr.second->getDefault();
			std::shared_ptr<NauInt> ni = std::dynamic_pointer_cast<NauInt>(d);
			m[attr.second->m_Id] = (*ni != 0);
		}
	}
}


void
AttribSet::initAttribInstanceBvec2Array(std::map<int, bvec2> &m) {

	for (auto & attr : m_Attributes) {
		if (attr.second->m_Type == Enums::DataType::BVEC2) {
			std::shared_ptr<Data> &d = attr.second->getDefault();
			std::shared_ptr<bvec2> ni = std::dynamic_pointer_cast<bvec2>(d);
			m[attr.second->m_Id] = *ni;
		}
	}
}


void
AttribSet::initAttribInstanceBvec3Array(std::map<int, bvec3> &m) {

	for (auto & attr : m_Attributes) {
		if (attr.second->m_Type == Enums::DataType::BVEC3) {
			std::shared_ptr<Data> &d = attr.second->getDefault();
			std::shared_ptr<bvec3> ni = std::dynamic_pointer_cast<bvec3>(d);
			m[attr.second->m_Id] = *ni;
		}
	}
}


void 
AttribSet::initAttribInstanceBvec4Array(std::map<int, bvec4> &m) {

	for (auto & attr : m_Attributes) {
		if (attr.second->m_Type == Enums::DataType::BVEC4) {
			std::shared_ptr<Data> &d = attr.second->getDefault();
			std::shared_ptr<bvec4> ni = std::dynamic_pointer_cast<bvec4>(d);
			m[attr.second->m_Id] = *ni;
		}
	}
}



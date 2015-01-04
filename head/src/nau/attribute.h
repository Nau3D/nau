#ifndef ATTRIBUTE_H
#define ATTRIBUTE_H



#include <nau/enums.h>
#include <nau/math/vec4.h>
#include <nau/math/vec2.h>
#include <nau/math/bvec4.h>
#include <nau/math/mat3.h>
#include <nau/math/mat4.h>

#include <map>
#include <string>
#include <vector>
#include <assert.h>

using namespace nau::math;

namespace nau {

	class AttribSet;

	class Attribute {
		
	public:
		friend class AttribSet;

		typedef enum {
			NONE,
			NORMALIZED,
			COLOUR
		} Semantics;


		Attribute(): m_Id(-1), m_Default(NULL), m_RangeDefined(false), m_ListDefined(false) {};

		Attribute(int id, std::string name, Enums::DataType type, bool readOnlyFlag = false, void *defaultV = NULL ): 
			m_Id(id), m_Name(name), m_Type(type), m_ReadOnlyFlag(readOnlyFlag), m_Default(NULL), m_Min(NULL), m_Max(NULL),
				m_ListDefined(false), m_RangeDefined(false)  {
		
			if (defaultV) {
				int s = Enums::getSize(m_Type);
				m_Default = malloc(s);
				memcpy(m_Default, defaultV, s);
			}
			else {
				m_Default = Enums::getDefaultValue(m_Type);
				
			}
		};


		~Attribute() {
			// can't free this memory because it may be in use in other attribute
			//bool isBasic = Enums::isBasicType(m_Type);
			//if (m_RangeDefined) {
			//	if (m_Min != NULL)
			//		if (isBasic)
			//			free(m_Min);
			//		else
			//			delete m_Min;
			//	if (m_Max != NULL)
			//		if (isBasic)
			//			free(m_Max);
			//		else
			//			delete m_Max;
			//}
			//if (m_Default != NULL)
			//	if (isBasic)
			//		free(m_Default);
			//	else
			//		delete m_Default;
		};


		static bool isValidUserAttrType(std::string s) {

			if (s == "FLOAT" || s == "INT" || s == "VEC4")
				return true;
			else
				return false;
		}


		std::string getName() {

			return m_Name;
		};


		void setRange(void *min, void *max) { 
				
			assert(m_Type != Enums::STRING);

			if (min == NULL && max == NULL)
				return;
			
			m_RangeDefined = true;

			if (min != NULL) {
				m_Min = (void *)malloc(Enums::getSize(m_Type));
				memcpy(m_Min, min, Enums::getSize(m_Type));
			}
			else
				m_Min = NULL;

			if (max != NULL) {
				m_Max = (void *)malloc(Enums::getSize(m_Type));
				memcpy(m_Max, max, Enums::getSize(m_Type));
			}
			else
				m_Max = NULL;
		};


		Enums::DataType getType() {

			return m_Type;
		}


		int getId() {

			return m_Id;
		}


		void listAdd(std::string name, int id) {
		
			m_ListDefined = true;
			mListValues.push_back(id);
			mListString.push_back(name);
		};

		 
		bool isValid(std::string value) {
		
			if (m_ListDefined) {				
				std::vector<std::string>::iterator it = mListString.begin();
				for ( ; it != mListString.end(); ++it) {
					if (*it == value)
						return true;
				}
				return false;
			}
			return false;
		};


		bool getRangeDefined() { return m_RangeDefined; };
		bool getListDefined() { return m_ListDefined; };

		void *getMax() {
		
			return m_Max;
		}

		void *getMin() {

			return m_Min;
		}

		int getListValue(std::string &s) {
		
			for(unsigned int i = 0 ; i < mListString.size(); ++i) {	
				if (mListString[i] == s)
					return mListValues[i];
			}
			return 0;
		};


		std::string getListString(int v) {
		
			for(unsigned int i = 0 ; i < mListValues.size(); ++i) {		
				if (mListValues[i] == v)
					return mListString[i];
			}
			return m_DummyS;
		};


		bool isValid(int v) {

			for (unsigned int i = 0; i < mListValues.size(); ++i) {
				if (mListValues[i] == v)
					return true;
			}
			return false;

		}


		int m_Id;
		std::string m_Name;
		Enums::DataType m_Type;
		bool m_ReadOnlyFlag;
		void *m_Default;
		bool m_ListDefined;
		bool m_RangeDefined;
		void *m_Min, *m_Max;
		std::vector<int> mListValues;
		std::vector<std::string> mListString;
		std::string m_DummyS;
	};


	// -------------------------------------------------------------------------------------------
	//    Attribute Set Float
	// -------------------------------------------------------------------------------------------



	class AttribSet {

	public:

		static const int USER_ATTRIBS = 1000;
		AttribSet() : m_NextFreeID(USER_ATTRIBS), m_DummyS("") { m_Dummy.m_Name = "NO_ATTR"; };
		~AttribSet() {};

		int getNextFreeID() {
		
			return m_NextFreeID++;
		}


		// TO DO: erase all attributes whose ID >= USER_ATTRIBS
		void deleteUserAttributes() {

			//std::map<int, Attribute>::iterator iter;
			//iter = m_Attributes.begin;
			//for (; iter != m_Attributes.end; ++iter) {

			//	if (iter->second.m_Id >= USER_ATTRIBS)
			//		m_Attributes.erase(*iter);
			//}
		}

		void add(Attribute a) {
			
			if (a.m_Id != -1) {
				m_Attributes[a.m_Name] = a;
			}
			Enums::DataType dt = a.getType();
			if (mDataTypeCounter.count(dt))
				++mDataTypeCounter[dt];
			else
				mDataTypeCounter[dt] = 1;
		}


		int getDataTypeCount(Enums::DataType dt ) {

			if (mDataTypeCounter.count(dt))
				return mDataTypeCounter[dt];
			else
				return 0;
		}

		
		const Attribute &get(std::string name) {

			if (m_Attributes.find(name) != m_Attributes.end()) 

				return(m_Attributes[name]);
			else
				return m_Dummy;

		}

		const Attribute &get(int id, Enums::DataType dt) {

			std::map<std::string, Attribute>::iterator it;
			it = m_Attributes.begin();
			for (; it != m_Attributes.end(); ++it) {

				if (it->second.m_Id == id && it->second.m_Type == dt)
					return (it->second);
			}
			return m_Dummy;
		}


		int getID(std::string name) {

			if (m_Attributes.find(name) != m_Attributes.end()) 

				return(m_Attributes[name].m_Id);
			else
				return -1;

		}


		const std::map<std::string, Attribute> &getAttributes() {

			return (m_Attributes);
		}


		const std::string &getName(int id, Enums::DataType dt) {

			std::map<std::string, Attribute>::iterator it;
			it = m_Attributes.begin();
			for ( ; it != m_Attributes.end(); ++it) {

				if (it->second.m_Id == id && it->second.m_Type == dt)
					return (it->first);
			}
			return m_DummyS;
		}


		void getPropTypeAndId(std::string &s, nau::Enums::DataType *dt , int *id) {
			
			Attribute a = get(s);
			*id = a.m_Id;

			if (a.m_Id != -1) {

				*dt = a.m_Type;
			}
		}


		


		const std::vector<std::string> &getListString(int id) {
		
			std::map<std::string, Attribute>::iterator it;
			it = m_Attributes.begin();
			for ( ; it != m_Attributes.end(); ++it) {

				if (it->second.m_Id == id && it->second.m_Type == Enums::DataType::ENUM)
					return (it->second.mListString);
			}
			return m_DummyVS;
		}


		const std::vector<int> &getListValues(int id) {
		
			std::map<std::string, Attribute>::iterator it;
			it = m_Attributes.begin();
			for ( ; it != m_Attributes.end(); ++it) {

				if (it->second.m_Id == id && it->second.m_Type == Enums::DataType::ENUM)
					return (it->second.mListValues);
			}
			return m_DummyVI;
		}


		std::string getListStringOp(std::string s, int prop) {
		
			Attribute a = get(s);
			return (a.getListString(prop));
		}




		std::string getListStringOp(int id, int prop) {
		
			std::map<std::string, Attribute>::iterator it;
			it = m_Attributes.begin();
			for ( ; it != m_Attributes.end(); ++it) {

				if (it->second.m_Id == id && it->second.m_Type == Enums::DataType::ENUM)
					return (it->second.getListString(prop));
			}
			return m_DummyS;
		}


		int getListValueOp(std::string s, std::string prop) {
		
			Attribute a = get(s);
			return (a.getListValue(prop));
		}


		int getListValueOp(int id, std::string prop) {
		
			std::map<std::string, Attribute>::iterator it;
			it = m_Attributes.begin();
			for ( ; it != m_Attributes.end(); ++it) {

				if (it->second.m_Id == id && it->second.m_Type == Enums::DataType::ENUM)
					return (it->second.getListValue(prop));
			}
			return -1;
		}


		void listAdd(std::string attrName, std::string elemS, int elem_Id) {

			std::map<std::string, Attribute>::iterator it;
			it = m_Attributes.begin();
			for ( ; it != m_Attributes.end(); ++it) {

				if (it->first == attrName) {
					it->second.listAdd(elemS, elem_Id);
					return;
				}
			}
		}


		bool isValid(std::string attr, std::string value) {

			Attribute a = get(attr);
			return a.isValid(value);
		}


		void setDefault(std::string attr, void *value) {
		
			if (m_Attributes.find(attr) != m_Attributes.end()) {
				assert(m_Attributes[attr].m_Type != Enums::STRING);
				if (m_Attributes[attr].m_Type != Enums::STRING) {
					int s = Enums::getSize(m_Attributes[attr].m_Type);
					m_Attributes[attr].m_Default = malloc(s);
					memcpy(m_Attributes[attr].m_Default, value, s);
				}
			}
		}


		//void setDefaultString(std::string attr, std::string value) {

		//	if (m_Attributes.find(attr) != m_Attributes.end()) {
		//		assert(m_Attributes[attr].m_Type == Enums::STRING);
		//		if (m_Attributes[attr].m_Type == Enums::STRING) {
		//			m_Attributes[attr].m_Default = malloc(value.size());
		//			memcpy(m_Attributes[attr].m_Default, &value, value.size());
		//		}
		//	}
		//}


		void *getDefault(int id, Enums::DataType type) {

			std::map<std::string, Attribute>::iterator it;
			it = m_Attributes.begin();
			for ( ; it != m_Attributes.end(); ++it) {

				if (it->second.m_Id == id && it->second.m_Type == type)
					return (it->second.m_Default);
			}
			return NULL;
		}


		//const std::string &getDefaultString(int id) {

		//	std::map<std::string, Attribute>::iterator it;
		//	it = m_Attributes.begin();
		//	for (; it != m_Attributes.end(); ++it) {

		//		if (it->second.m_Id == id && it->second.m_Type == Enums::STRING)
		//			return *(static_cast<std::string*>(it->second.m_Default));
		//	}
		//	return m_DummyS;
		//}


		void initAttribInstanceStringArray(std::map<int, std::string> &m) {

			//std::map<std::string, Attribute>::iterator it;
			//it = m_Attributes.begin();
			//for ( ; it != m_Attributes.end(); ++it) {
			//	if (it->second.m_Type == Enums::DataType::INT) {

			//			m[it->second.m_Id] = *(std::string *)(it->second.m_Default);
			//	}
			//}
		}

		void initAttribInstanceIntArray(std::map<int, int> &m) {

			std::map<std::string, Attribute>::iterator it;
			it = m_Attributes.begin();
			for (; it != m_Attributes.end(); ++it) {
				if (it->second.m_Type == Enums::DataType::INT) {

					m[it->second.m_Id] = *(int *)(it->second.m_Default);
				}
			}
		}

		void initAttribInstanceEnumArray(std::map<int,int> &m) {

			std::map<std::string, Attribute>::iterator it;
			it = m_Attributes.begin();
			for ( ; it != m_Attributes.end(); ++it) {
				if (it->second.m_Type == Enums::DataType::ENUM) {

						m[it->second.m_Id] = *(int *)(it->second.m_Default);
				}
			}
		}


		void initAttribInstanceUIntArray(std::map<int,unsigned int> &m) {

			std::map<std::string, Attribute>::iterator it;
			it = m_Attributes.begin();
			for ( ; it != m_Attributes.end(); ++it) {
				if (it->second.m_Type == Enums::DataType::UINT) {

						m[it->second.m_Id] = *(unsigned int *)(it->second.m_Default);
				}
			}
		}


		void initAttribInstanceFloatArray(std::map<int,float> &m) {

			std::map<std::string, Attribute>::iterator it;
			it = m_Attributes.begin();
			for ( ; it != m_Attributes.end(); ++it) {
				if (it->second.m_Type == Enums::DataType::FLOAT) {

						m[it->second.m_Id] = *(float *)(it->second.m_Default);
				}
			}
		}


		void initAttribInstanceVec4Array(std::map<int,vec4> &m) {

			std::map<std::string, Attribute>::iterator it;
			it = m_Attributes.begin();
			for ( ; it != m_Attributes.end(); ++it) {
				if (it->second.m_Type == Enums::DataType::VEC4) {

						m[it->second.m_Id] = *(vec4 *)(it->second.m_Default);
				}
			}
		}

		void initAttribInstanceVec2Array(std::map<int, vec2> &m) {

			std::map<std::string, Attribute>::iterator it;
			it = m_Attributes.begin();
			for (; it != m_Attributes.end(); ++it) {
				if (it->second.m_Type == Enums::DataType::VEC2) {

					m[it->second.m_Id] = *(vec2 *)(it->second.m_Default);
				}
			}
		}

		void initAttribInstanceMat4Array(std::map<int, mat4> &m) {

			std::map<std::string, Attribute>::iterator it;
			it = m_Attributes.begin();
			for (; it != m_Attributes.end(); ++it) {
				if (it->second.m_Type == Enums::DataType::MAT4) {
					m[it->second.m_Id] = *(mat4 *)(it->second.m_Default);
				}
			}
		}

		void initAttribInstanceMat3Array(std::map<int, mat3> &m) {

			std::map<std::string, Attribute>::iterator it;
			it = m_Attributes.begin();
			for (; it != m_Attributes.end(); ++it) {
				if (it->second.m_Type == Enums::DataType::MAT3) {
					m[it->second.m_Id] = *(mat3 *)(it->second.m_Default);
				}
			}
		}


		void initAttribInstanceBvec4Array(std::map<int, bvec4> &m) {

			std::map<std::string, Attribute>::iterator it;
			it = m_Attributes.begin();
			for ( ; it != m_Attributes.end(); ++it) {
				if (it->second.m_Type == Enums::DataType::BVEC4) {

						m[it->second.m_Id] = *(bvec4 *)(it->second.m_Default);
				}
			}
		}


		void initAttribInstanceBoolArray(std::map<int,bool> &m) {

			std::map<std::string, Attribute>::iterator it;
			it = m_Attributes.begin();
			for ( ; it != m_Attributes.end(); ++it) {
				if (it->second.m_Type == Enums::DataType::BOOL) {

						m[it->second.m_Id] = *(bool *)(it->second.m_Default);
				}
			}
		}


	protected:
		std::map<std::string, Attribute> m_Attributes;
		Attribute m_Dummy;
		std::string m_DummyS;
		std::vector<std::string> m_DummyVS;
		std::vector<int> m_DummyVI;
		int m_NextFreeID;
		std::map<int, int> mDataTypeCounter;
	};




};


#endif
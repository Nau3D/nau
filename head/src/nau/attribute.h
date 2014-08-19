#ifndef ATTRIBUTE_H
#define ATTRIBUTE_H

#include <map>
#include <string>
#include <vector>
#include <assert.h>

#include "enums.h"
#include "nau/math/vec4.h"
#include "nau/math/bvec4.h"
#include "nau/math/mat3.h"
#include "nau/math/mat4.h"

using namespace nau::math;

namespace nau {

	class AttribSet;

	class Attribute {
		
	public:
		friend class AttribSet;


		Attribute(): mId(-1), mDefault(NULL), mRangeDefined(false), mListDefined(false) {};

		Attribute(int id, std::string name, Enums::DataType type, bool readOnlyFlag = false, void *defaultV = NULL ): 
				mId(id),mName(name), mType(type),mReadOnlyFlag(readOnlyFlag), mDefault(NULL),
				mListDefined(false), mRangeDefined(false)  {
		
			if (defaultV) {
				int s = Enums::getSize(mType);
				mDefault = malloc(s);
				memcpy(mDefault, defaultV, s);
			}
			else {
				mDefault = Enums::getDefaultValue(mType);
				
			}
		};


		~Attribute() {

			if (mRangeDefined) {
				free (mMin);
				free (mMax);
			}
		};


		static bool isValidUserAttrType(std::string s) {

			if (s == "FLOAT" || s == "INT" )
				return true;
			else
				return false;
		}


		std::string getName() {

			return mName;
		};


		void setRange(void *min, void *max) { 
				
			mRangeDefined = true;
			mMin = (void *)malloc(Enums::getSize(mType));
			memcpy(mMin, min, Enums::getSize(mType));
			mMax = (void *)malloc(Enums::getSize(mType));
			memcpy(mMax, max, Enums::getSize(mType));
		};


		Enums::DataType getType() {

			return mType;
		}


		int getId() {

			return mId;
		}


		void listAdd(std::string name, int id) {
		
			mListDefined = true;
			mListValues.push_back(id);
			mListString.push_back(name);
		};

		 
		bool isValid(std::string value) {
		
			if (mListDefined) {				
				std::vector<std::string>::iterator it = mListString.begin();
				for ( ; it != mListString.end(); ++it) {
					if (*it == value)
						return true;
				}
				return false;
			}
			return false;
		};


		bool getRangeDefined() { return mRangeDefined; };
		bool getListDefined() { return mListDefined; };


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
			return mDummyS;
		};


		int mId;
		std::string mName;
		Enums::DataType mType;
		bool mReadOnlyFlag;
		void *mDefault;
		bool mListDefined;
		bool mRangeDefined;
		void *mMin, *mMax;
		std::vector<int> mListValues;
		std::vector<std::string> mListString;
		std::string mDummyS;
	};


	// -------------------------------------------------------------------------------------------
	//    Attribute Set Float
	// -------------------------------------------------------------------------------------------



	class AttribSet {

	public:

		AttribSet(): mNextFreeID(1000), mDummyS("") {mDummy.mName = "NO_ATTR"; };
		~AttribSet() {};

		int getNextFreeID() {
		
			return mNextFreeID++;
		}


		void add(Attribute a) {
			
			if (a.mId != -1) {
				mAttributes[a.mName] = a;
			}
			Enums::DataType dt = a.getType();
			if (mDataTypeCounter.count(dt))
				++mDataTypeCounter[dt];
			else
				mDataTypeCounter[dt] = 1;
		};


		int getDataTypeCount(Enums::DataType dt ) {

			if (mDataTypeCounter.count(dt))
				return mDataTypeCounter[dt];
			else
				return 0;
		}

		
		const Attribute &get(std::string name) {

			if (mAttributes.find(name) != mAttributes.end()) 

				return(mAttributes[name]);
			else
				return mDummy;

		};


		int getID(std::string name) {

			if (mAttributes.find(name) != mAttributes.end()) 

				return(mAttributes[name].mId);
			else
				return -1;

		};


		const std::map<std::string, Attribute> &getAttributes() {

			return (mAttributes);
		}


		const std::string &getName(int id, Enums::DataType dt) {

			std::map<std::string, Attribute>::iterator it;
			it = mAttributes.begin();
			for ( ; it != mAttributes.end(); ++it) {

				if (it->second.mId == id && it->second.mType == dt)
					return (it->first);
			}
			return mDummyS;
		}


		void getPropTypeAndId(std::string &s, nau::Enums::DataType *dt , int *id) {
			
			Attribute a = get(s);
			*id = a.mId;

			if (a.mId != -1) {

				*dt = a.mType;
			}
		}


		const std::vector<std::string> &getListString(int id) {
		
			std::map<std::string, Attribute>::iterator it;
			it = mAttributes.begin();
			for ( ; it != mAttributes.end(); ++it) {

				if (it->second.mId == id && it->second.mType == Enums::DataType::ENUM)
					return (it->second.mListString);
			}
			return mDummyVS;
		}


		const std::vector<int> &getListValues(int id) {
		
			std::map<std::string, Attribute>::iterator it;
			it = mAttributes.begin();
			for ( ; it != mAttributes.end(); ++it) {

				if (it->second.mId == id && it->second.mType == Enums::DataType::ENUM)
					return (it->second.mListValues);
			}
			return mDummyVI;
		}


		std::string getListStringOp(std::string s, int prop) {
		
			Attribute a = get(s);
			return (a.getListString(prop));
		}




		std::string getListStringOp(int id, int prop) {
		
			std::map<std::string, Attribute>::iterator it;
			it = mAttributes.begin();
			for ( ; it != mAttributes.end(); ++it) {

				if (it->second.mId == id && it->second.mType == Enums::DataType::ENUM)
					return (it->second.getListString(prop));
			}
			return mDummyS;
		}


		int getListValueOp(std::string s, std::string prop) {
		
			Attribute a = get(s);
			return (a.getListValue(prop));
		}


		int getListValueOp(int id, std::string prop) {
		
			std::map<std::string, Attribute>::iterator it;
			it = mAttributes.begin();
			for ( ; it != mAttributes.end(); ++it) {

				if (it->second.mId == id && it->second.mType == Enums::DataType::ENUM)
					return (it->second.getListValue(prop));
			}
			return -1;
		}


		void listAdd(std::string attrName, std::string elemS, int elemId) {

			std::map<std::string, Attribute>::iterator it;
			it = mAttributes.begin();
			for ( ; it != mAttributes.end(); ++it) {

				if (it->first == attrName) {
					it->second.listAdd(elemS, elemId);
					return;
				}
			}
		}


		bool isValid(std::string attr, std::string value) {

			Attribute a = get(attr);
			return a.isValid(value);
		}


		void setDefault(std::string attr, void *value) {
		
			if (mAttributes.find(attr) != mAttributes.end()) {
				int s = Enums::getSize(mAttributes[attr].mType);
				mAttributes[attr].mDefault = malloc(s);
				memcpy(mAttributes[attr].mDefault, value, s);
			}
		}


		void *getDefault(int id, Enums::DataType type) {

			std::map<std::string, Attribute>::iterator it;
			it = mAttributes.begin();
			for ( ; it != mAttributes.end(); ++it) {

				if (it->second.mId == id && it->second.mType == type)
					return (it->second.mDefault);
			}
			return NULL;
		}


		void initAttribInstanceIntArray(std::map<int,int> &m) {

			std::map<std::string, Attribute>::iterator it;
			it = mAttributes.begin();
			for ( ; it != mAttributes.end(); ++it) {
				if (it->second.mType == Enums::DataType::INT) {

					//if (it->second.mDefault == NULL) 
					//	m[it->second.mId] = 0;
					//else
						m[it->second.mId] = *(int *)(it->second.mDefault);
				}
			}
		}


		void initAttribInstanceEnumArray(std::map<int,int> &m) {

			std::map<std::string, Attribute>::iterator it;
			it = mAttributes.begin();
			for ( ; it != mAttributes.end(); ++it) {
				if (it->second.mType == Enums::DataType::ENUM) {

					//if (it->second.mDefault == NULL) 
					//	m[it->second.mId] = 0;
					//else
						m[it->second.mId] = *(int *)(it->second.mDefault);
				}
			}
		}


		void initAttribInstanceUIntArray(std::map<int,unsigned int> &m) {

			std::map<std::string, Attribute>::iterator it;
			it = mAttributes.begin();
			for ( ; it != mAttributes.end(); ++it) {
				if (it->second.mType == Enums::DataType::UINT) {

					//if (it->second.mDefault == NULL) 
					//	m[it->second.mId] = 0;
					//else
						m[it->second.mId] = *(unsigned int *)(it->second.mDefault);
				}
			}
		}


		void initAttribInstanceFloatArray(std::map<int,float> &m) {

			std::map<std::string, Attribute>::iterator it;
			it = mAttributes.begin();
			for ( ; it != mAttributes.end(); ++it) {
				if (it->second.mType == Enums::DataType::FLOAT) {

					//if (it->second.mDefault == NULL) 
					//	m[it->second.mId] = 0;
					//else
						m[it->second.mId] = *(float *)(it->second.mDefault);
				}
			}
		}


		void initAttribInstanceVec4Array(std::map<int,vec4> &m) {

			std::map<std::string, Attribute>::iterator it;
			it = mAttributes.begin();
			for ( ; it != mAttributes.end(); ++it) {
				if (it->second.mType == Enums::DataType::VEC4) {

					//if (it->second.mDefault == NULL) 
					//	m[it->second.mId] = vec4();
					//else
						m[it->second.mId] = *(vec4 *)(it->second.mDefault);
				}
			}
		}


		void initAttribInstanceMat4Array(std::map<int, mat4> &m) {

			std::map<std::string, Attribute>::iterator it;
			it = mAttributes.begin();
			for (; it != mAttributes.end(); ++it) {
				if (it->second.mType == Enums::DataType::MAT4) {
					m[it->second.mId] = *(mat4 *)(it->second.mDefault);
				}
			}
		}

		void initAttribInstanceMat3Array(std::map<int, mat3> &m) {

			std::map<std::string, Attribute>::iterator it;
			it = mAttributes.begin();
			for (; it != mAttributes.end(); ++it) {
				if (it->second.mType == Enums::DataType::MAT3) {
					m[it->second.mId] = *(mat3 *)(it->second.mDefault);
				}
			}
		}


		void initAttribInstanceBvec4Array(std::map<int, bvec4> &m) {

			std::map<std::string, Attribute>::iterator it;
			it = mAttributes.begin();
			for ( ; it != mAttributes.end(); ++it) {
				if (it->second.mType == Enums::DataType::BVEC4) {

					//if (it->second.mDefault == NULL) 
					//	m[it->second.mId] = bvec4();
					//else
						m[it->second.mId] = *(bvec4 *)(it->second.mDefault);
				}
			}
		}


		void initAttribInstanceBoolArray(std::map<int,bool> &m) {

			std::map<std::string, Attribute>::iterator it;
			it = mAttributes.begin();
			for ( ; it != mAttributes.end(); ++it) {
				if (it->second.mType == Enums::DataType::BOOL) {

					//if (it->second.mDefault == NULL) 
					//	m[it->second.mId] = 0;
					//else
						m[it->second.mId] = *(bool *)(it->second.mDefault);
				}
			}
		}


	protected:
		std::map<std::string, Attribute> mAttributes;
		Attribute mDummy;
		std::string mDummyS;
		std::vector<std::string> mDummyVS;
		std::vector<int> mDummyVI;
		int mNextFreeID;
		std::map<int, int> mDataTypeCounter;
	};




};


#endif
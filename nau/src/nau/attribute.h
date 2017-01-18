#ifndef ATTRIBUTE_H
#define ATTRIBUTE_H


#include "nau/enums.h"
#include "nau/math/data.h"
#include "nau/math/number.h"
#include "nau/math/numberArray.h"
#include "nau/math/matrix.h"
#include "nau/math/vec4.h"
#include "nau/math/vec2.h"
#include "nau/render/iAPISupport.h"

#include <assert.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace nau::math;
using namespace nau::render;

#ifdef _WINDLL
#ifdef nau_EXPORTS
#define nau_API __declspec(dllexport)   
#else  
#define nau_API __declspec(dllimport)   
#endif 
#else
#define nau_API
#endif

namespace nau {

	class AttribSet;

	class Attribute {
		
	public:
		friend class AttribSet;

		typedef enum {
			NONE,
			NORMALIZED,
			COLOR
		} Semantics;


		nau_API Attribute();
		nau_API Attribute(unsigned int id, std::string name, Enums::DataType type,
			bool readOnlyFlag = false, Data *defaultV = NULL,
			Data *min=NULL, Data *max = NULL, 
			IAPISupport::APIFeatureSupport requires = IAPISupport::OK, Semantics sem=NONE);
		
		nau_API Attribute(unsigned int id, std::string name, std::string objType, bool readOnlyFlag = false, bool mustExist = true, std::string defaults = "");

		nau_API ~Attribute();

		nau_API Attribute(const Attribute & source);

		//Attribute & operator=(const Attribute &rhs);

		static nau_API bool isValidUserAttrType(std::string s);
		static nau_API std::vector<std::string> &getValidUserAttrTypes();

		nau_API std::string &getName();
		nau_API Enums::DataType getType();
		nau_API int getId();
		nau_API bool getRangeDefined();
		nau_API bool getReadOnlyFlag();
		nau_API Semantics getSemantics();

		nau_API std::shared_ptr<Data> &getMax();
		nau_API std::shared_ptr<Data> &getMin();
		nau_API std::shared_ptr<Data> &getDefault();

		nau_API const std::string & getDefaultString();
		nau_API bool getMustExist();
		nau_API const std::string & getObjType();

		nau_API bool getListDefined();
		nau_API int getOptionValue(std::string &s);
		nau_API std::string &getOptionString(int v);
		nau_API const std::vector<std::string> &getOptionStringList();
		nau_API void getOptionListSupported(std::vector<int>* result);
		nau_API void getOptionStringListSupported(std::vector<std::string> *result);

		nau_API void setDefault(Data &);

		nau_API void setRequirement(IAPISupport::APIFeatureSupport req);
		nau_API IAPISupport::APIFeatureSupport getRequirement();

		nau_API void listAdd(std::string name, int id, IAPISupport::APIFeatureSupport requires = IAPISupport::OK);

		// for enum types only. Checks if it is a valid option. 
		nau_API bool isValid(std::string value);
		nau_API bool isValid(int v);

	protected:
		static std::vector<std::string> m_DummyVS;

		int m_Id;
		std::string m_Name;
		std::string m_ObjType;
		std::string m_DefaultS;
		bool m_MustExist;
		Enums::DataType m_Type;
		bool m_ReadOnlyFlag;
		std::shared_ptr<Data> m_Default;
		bool m_ListDefined;
		bool m_RangeDefined;
		std::shared_ptr<Data> m_Min;
		std::shared_ptr<Data> m_Max;

		Semantics m_Semantics;
		IAPISupport::APIFeatureSupport m_Requires;
		std::vector<int> m_ListValues;
		std::vector<IAPISupport::APIFeatureSupport> m_ListRequire;
		std::vector<std::string> m_ListString;
		std::string m_DummyS;
	};


	// -------------------------------------------------------------------------------------------
	//    Attribute Set
	// -------------------------------------------------------------------------------------------



	class AttribSet {

	public:

		static const int USER_ATTRIBS = 1000;
		AttribSet();
		~AttribSet() ;

		nau_API unsigned int getNextFreeID();

		nau_API void deleteUserAttributes();

		nau_API void add(Attribute a);

		nau_API std::map<std::string, std::unique_ptr<Attribute>> &getAttributes();

		nau_API int getDataTypeCount(Enums::DataType dt);
		nau_API std::unique_ptr<Attribute> &get(std::string name);
		nau_API std::unique_ptr<Attribute> &get(int id, Enums::DataType dt);
		// returns the ID of the attribute
		// -1 if attribute does not exist
		nau_API int getID(std::string name);
		nau_API const std::string &getName(int id, Enums::DataType dt);
		nau_API void getPropTypeAndId(const std::string &s, nau::Enums::DataType *dt, int *id);

		nau_API const std::vector<std::string> &getListString(int id);
		nau_API const std::vector<int> &getListValues(int id);
		nau_API std::string getListStringOp(std::string s, int prop);
		nau_API std::string getListStringOp(int id, int prop);
		nau_API int getListValueOp(std::string s, std::string prop);
		nau_API int getListValueOp(int id, std::string prop);
		nau_API void listAdd(std::string attrName, std::string elemS, int elem_Id, IAPISupport::APIFeatureSupport requires = IAPISupport::OK);
		nau_API bool isValid(std::string attr, std::string value);
		nau_API void setDefault(std::string attr, Data &value);
		//std::unique_ptr<Data> &getDefault(int id, Enums::DataType type);

		//template <typename T>
		//void initAttribInstanceArray(Enums::DataType, std::map<int, T> &m);

		void initAttribInstanceIntArray(std::map<int, int> &m);
		void initAttribInstanceInt2Array(std::map<int, ivec2> &m);
		void initAttribInstanceInt3Array(std::map<int, ivec3> &m);
		void initAttribInstanceInt4Array(std::map<int, ivec4> &m);

		void initAttribInstanceIntArrayArray(std::map<int, NauIntArray> &m);

		void initAttribInstanceEnumArray(std::map<int, int> &m);
		void initAttribInstanceStringArray(std::map<int, std::string> &m);

		void initAttribInstanceUIntArray(std::map<int, unsigned int> &m);
		void initAttribInstanceUInt2Array(std::map<int, uivec2> &m);
		void initAttribInstanceUInt3Array(std::map<int, uivec3> &m);
		void initAttribInstanceUInt4Array(std::map<int, uivec4> &m);

		void initAttribInstanceFloatArray(std::map<int, float> &m);
		void initAttribInstanceVec4Array(std::map<int, vec4> &m);
		void initAttribInstanceVec2Array(std::map<int, vec2> &m);
		void initAttribInstanceVec3Array(std::map<int, vec3> &m);

		void initAttribInstanceMat4Array(std::map<int, mat4> &m);
		void initAttribInstanceMat3Array(std::map<int, mat3> &m);
		void initAttribInstanceMat2Array(std::map<int, mat2> &m);

		void initAttribInstanceDoubleArray(std::map<int, double> &m);
		void initAttribInstanceDVec4Array(std::map<int, dvec4> &m);
		void initAttribInstanceDVec2Array(std::map<int, dvec2> &m);
		void initAttribInstanceDVec3Array(std::map<int, dvec3> &m);

		void initAttribInstanceDMat4Array(std::map<int, dmat4> &m);
		void initAttribInstanceDMat3Array(std::map<int, dmat3> &m);
		void initAttribInstanceDMat2Array(std::map<int, dmat2> &m);

		void initAttribInstanceBoolArray(std::map<int, bool> &m);
		void initAttribInstanceBvec2Array(std::map<int, bvec2> &m);
		void initAttribInstanceBvec3Array(std::map<int, bvec3> &m);
		void initAttribInstanceBvec4Array(std::map<int, bvec4> &m);


	protected:
		std::map<std::string, std::unique_ptr<Attribute>> m_Attributes;
		std::unique_ptr<Attribute> m_Dummy;
		std::string m_DummyS;
		std::vector<std::string> m_DummyVS;
		std::vector<int> m_DummyVI;
		int m_NextFreeID;
		std::map<int, int> mDataTypeCounter;
	};




};


#endif

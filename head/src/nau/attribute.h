#ifndef ATTRIBUTE_H
#define ATTRIBUTE_H


#include "nau/enums.h"
#include "nau/math/matrix.h"
#include "nau/math/vec4.h"
#include "nau/math/vec2.h"
#include "nau/render/iAPISupport.h"

#include <assert.h>
#include <map>
#include <string>
#include <vector>

using namespace nau::math;
using namespace nau::render;

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


		Attribute();
		Attribute(int id, std::string name, Enums::DataType type, 
			bool readOnlyFlag = false, void *defaultV = NULL,
			void *min=NULL, void *max = NULL, IAPISupport::APIFeatureSupport requires = IAPISupport::OK);
		
		~Attribute();


		static bool isValidUserAttrType(std::string s);
		static std::vector<std::string> &getValidUserAttrTypes();

		std::string &getName();
		Enums::DataType getType();
		int getId();
		bool getRangeDefined();
		bool getListDefined();
		void *getMax();
		void *getMin();
		bool getReadOnlyFlag();
		int getOptionValue(std::string &s);
		std::string &getOptionString(int v);
		const std::vector<std::string> &getOptionStringList();
		void getOptionStringListSupported(std::vector<std::string> *result);
		void *getDefault();

		void setRange(void *min, void *max);

		void setRequirement(IAPISupport::APIFeatureSupport req);
		IAPISupport::APIFeatureSupport getRequirement();

		void listAdd(std::string name, int id, IAPISupport::APIFeatureSupport requires = IAPISupport::OK);

		// for enum types only. Checks if it is a valid option. 
		bool isValid(std::string value);
		bool isValid(int v);


		static std::vector<std::string> m_DummyVS;

		int m_Id;
		std::string m_Name;
		Enums::DataType m_Type;
		bool m_ReadOnlyFlag;
		void *m_Default;
		bool m_ListDefined;
		bool m_RangeDefined;
		void *m_Min, *m_Max;
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

		int getNextFreeID();

		void deleteUserAttributes();

		void add(Attribute a);

		const std::map<std::string, Attribute> &getAttributes();

		int getDataTypeCount(Enums::DataType dt);
		const Attribute &get(std::string name);
		const Attribute &get(int id, Enums::DataType dt);
		// returns the ID of the attribute
		// -1 if attribute does not exist
		int getID(std::string name);
		const std::string &getName(int id, Enums::DataType dt);
		void getPropTypeAndId(std::string &s, nau::Enums::DataType *dt, int *id);

		const std::vector<std::string> &getListString(int id);
		const std::vector<int> &getListValues(int id);
		std::string getListStringOp(std::string s, int prop);
		std::string getListStringOp(int id, int prop);
		int getListValueOp(std::string s, std::string prop);
		int getListValueOp(int id, std::string prop);
		void listAdd(std::string attrName, std::string elemS, int elem_Id, IAPISupport::APIFeatureSupport requires = IAPISupport::OK);
		bool isValid(std::string attr, std::string value);
		void setDefault(std::string attr, void *value);
		void *getDefault(int id, Enums::DataType type);

		void initAttribInstanceIntArray(std::map<int, int> &m);
		void initAttribInstanceInt2Array(std::map<int, ivec2> &m);
		void initAttribInstanceEnumArray(std::map<int, int> &m);
		void initAttribInstanceUIntArray(std::map<int, unsigned int> &m);
		void initAttribInstanceUInt2Array(std::map<int, uivec2> &m);
		void initAttribInstanceUInt3Array(std::map<int, uivec3> &m);
		void initAttribInstanceFloatArray(std::map<int, float> &m);
		void initAttribInstanceVec4Array(std::map<int, vec4> &m);
		void initAttribInstanceVec2Array(std::map<int, vec2> &m);
		void initAttribInstanceVec3Array(std::map<int, vec3> &m);
		void initAttribInstanceMat4Array(std::map<int, mat4> &m);
		void initAttribInstanceMat3Array(std::map<int, mat3> &m);
		void initAttribInstanceBvec4Array(std::map<int, bvec4> &m);
		void initAttribInstanceBoolArray(std::map<int, bool> &m);


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
#ifndef ATTRIBUTEVALUES_H
#define ATTRIBUTEVALUES_H

#include <map>
#include <assert.h>
#include <nau/attribute.h>

#define ENUM(A,B) static const EnumProperty A = (EnumProperty)B 

namespace nau {
	class AttributeValues {
	public:
		typedef  enum {} EnumProperty;
		typedef enum {} Mat4Property;
		typedef enum {} Mat3Property;

		std::map<int,int> m_EnumProps;

		int getPrope(EnumProperty prop) {
			return m_EnumProps[prop];
		}

		AttributeValues() {}

		void InitArrrays(AttribSet Attribs) {
			Attribs.initAttribInstanceEnumArray(m_EnumProps);
		}
	};

};

#endif


//std::map<int,int> m_IntProps;
//	std::map<int,int> m_EnumProps;
//	std::map<int,unsigned int> m_UIntProps;
//	std::map<int,bool> m_BoolProps;
//	std::map<int,float> m_FloatProps;
//
//	int getPropi(IntProperty prop) {
//		assert(m_UIntProps.find(prop) != m_UIntProps.end());
//		return(m_UIntProps[prop]);
//	};
//
//	int getPrope(EnumProperty prop) {
//		assert(m_EnumProps.find(prop) != m_EnumProps.end());
//		return m_EnumProps[prop];		
//	};
//
//	unsigned int getPropui(UIntProperty prop) {
//		assert(m_UIntProps.find(prop) != m_UIntProps.end());
//		return(m_UIntProps[prop]);
//	};
//
//	bool getPropb(BoolProperty prop) {
//		assert(m_BoolProps.find(prop) != m_BoolProps.end());
//		return m_BoolProps[prop];
//	};
//
//	float getPropf(FloatProperty prop) {
//		assert(m_FloatProps.find(prop) != m_FloatProps.end());
//		return m_FloatProps[prop];
//	};
//	
//	void initArrays() {
//
//		Attribs.initAttribInstanceEnumArray(m_EnumProps);
//		Attribs.initAttribInstanceIntArray(m_IntProps);
//		Attribs.initAttribInstanceUIntArray(m_UIntProps);
//	}
//
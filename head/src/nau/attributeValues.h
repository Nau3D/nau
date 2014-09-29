#ifndef ATTRIBUTEVALUES_H
#define ATTRIBUTEVALUES_H

#include <map>
#include <assert.h>

#include <nau/attribute.h>
#include <nau/math/mat3.h>
#include <nau/math/mat4.h>

#define ENUM_PROP(A,B) static const EnumProperty A = (EnumProperty)B 
#define INT_PROP(A,B) static const IntProperty A = (IntProperty)B 
#define UINT_PROP(A,B) static const UIntProperty A = (UIntProperty)B 
#define BOOL_PROP(A,B) static const BoolProperty A = (BoolProperty)B 
#define BOOL4_PROP(A,B) static const Bool4Property A = (Bool4Property)B 
#define FLOAT_PROP(A,B) static const FloatProperty A = (FloatProperty)B 
#define FLOAT4_PROP(A,B) static const Float4Property A = (Float4Property)B 
#define MAT4_PROP(A,B) static const Mat4Property A = (Mat4Property)B 
#define MAT3_PROP(A,B) static const Mat3Property A = (Mat3Property)B 

namespace nau {
	class AttributeValues {
	public:
		typedef enum {} EnumProperty;
		typedef enum {} IntProperty;
		typedef enum {} UIntProperty;
		typedef enum {} BoolProperty;
		typedef enum {} Bool4Property;
		typedef enum {} FloatProperty;
		typedef enum {} Float4Property;
		typedef enum {} Mat4Property;
		typedef enum {} Mat3Property;

		std::map<int,int> m_EnumProps;
		std::map<int,int> m_IntProps;
		std::map<int,unsigned int> m_UIntProps;
		std::map<int,bool> m_BoolProps;
		std::map<int, bvec4> m_Bool4Props;
		std::map<int, vec4> m_Float4Props;
		std::map<int, float> m_FloatProps;
		std::map<int, mat3> m_Mat3Props;
		std::map<int, mat4> m_Mat4Props;


		void copy(AttributeValues *to) {

			to->m_EnumProps =   m_EnumProps;
			to->m_IntProps =    m_IntProps;
			to->m_UIntProps =   m_UIntProps;
			to->m_BoolProps =   m_BoolProps;
			to->m_Bool4Props =  m_Bool4Props;
			to->m_FloatProps =  m_FloatProps;
			to->m_Float4Props = m_Float4Props;
			to->m_Mat3Props =   m_Mat3Props;
			to->m_Mat4Props =   m_Mat4Props;
		}

		int getPrope(EnumProperty prop) {
			return m_EnumProps[prop];
		}

		int getPropi(IntProperty prop) {
			return m_IntProps[prop];
		}

		unsigned int getPropui(UIntProperty prop) {
			return m_UIntProps[prop];
		}

		bool getPropb(BoolProperty prop) {
			return m_BoolProps[prop];
		}

		vec4 &getPropf4(Float4Property prop) {
			return m_Float4Props[prop];
		}

		float getPropf(FloatProperty prop) {
			return m_FloatProps[prop];
		}

		bvec4 &getPropb4(Bool4Property prop) {
			return m_Bool4Props[prop];
		}

		mat4 &getPropMat4(Mat4Property prop) {
			return m_Mat4Props[prop];
		}

		mat3 &getPropMat3(Mat3Property prop) {
			return m_Mat3Props[prop];
		}

		void *getProp(int prop, Enums::DataType type) {

			switch (type) {

			case Enums::ENUM:
				assert(m_EnumProps.count(prop) > 0);
				return(&(m_EnumProps[prop]));
				break;
			case Enums::INT:
				assert(m_IntProps.count(prop) > 0);
				return(&(m_IntProps[prop]));
				break;
			case Enums::UINT:
				assert(m_UIntProps.count(prop) > 0);
				return(&(m_UIntProps[prop]));
				break;
			case Enums::BOOL:
				assert(m_BoolProps.count(prop) > 0);
				return(&(m_BoolProps[prop]));
				break;
			case Enums::BVEC4:
				assert(m_Bool4Props.count(prop) > 0);
				return(&(m_Bool4Props[prop]));
				break;
			case Enums::FLOAT:
				assert(m_FloatProps.count(prop) > 0);
				return(&(m_FloatProps[prop]));
				break;
			case Enums::VEC4:
				assert(m_Float4Props.count(prop) > 0);
				return(&(m_Float4Props[prop]));
				break;
			case Enums::MAT4:
				assert(m_Mat4Props.count(prop) > 0);
				return(&(m_Mat4Props[prop]));
				break;
			case Enums::MAT3:
				assert(m_Mat3Props.count(prop) > 0);
				return(&(m_Mat3Props[prop]));
				break;

			}
			assert(false && "Missibng Data Type in class attributeValues");
			return NULL;
		}



		void setProp(int prop, Enums::DataType type, void *value) {

			switch (type) {

			case Enums::ENUM:
				assert(m_EnumProps.count(prop) > 0);
				m_EnumProps[prop] = *(int *)value;
				break;
			case Enums::INT:
				assert(m_IntProps.count(prop) > 0);
				m_IntProps[prop] = *(int *)value;
				break;
			case Enums::UINT:
				assert(m_UIntProps.count(prop) > 0);
				m_UIntProps[prop] = *(unsigned int *)value;
				break;
			case Enums::BOOL:
				assert(m_BoolProps.count(prop) > 0);
				m_BoolProps[prop] = *(bool *)value;
				break;
			case Enums::BVEC4:
				assert(m_Bool4Props.count(prop) > 0);
				m_Bool4Props[prop] = *(bvec4 *)value;
				break;
			case Enums::FLOAT:
				assert(m_FloatProps.count(prop) > 0);
				m_FloatProps[prop] = *(float *)value;
				break;
			case Enums::VEC4:
				assert(m_Float4Props.count(prop) > 0);
				m_Float4Props[prop] = *(vec4 *)value;
				break;
			case Enums::MAT4:
				assert(m_Mat4Props.count(prop) > 0);
				m_Mat4Props[prop] = *(mat4 *)value;
				break;
			case Enums::MAT3:
				assert(m_Mat3Props.count(prop) > 0);
				m_Mat3Props[prop] = *(mat3 *)value;
				break;
			default:
				assert(false && "Missing Data Type in class attributeValues or Invalid prop");
			}	
		}


		void initArrays(AttribSet Attribs) {
			Attribs.initAttribInstanceEnumArray(m_EnumProps);
			Attribs.initAttribInstanceIntArray(m_IntProps);
			Attribs.initAttribInstanceUIntArray(m_UIntProps);
			Attribs.initAttribInstanceBoolArray(m_BoolProps);
			Attribs.initAttribInstanceBvec4Array(m_Bool4Props);
			Attribs.initAttribInstanceVec4Array(m_Float4Props);
			Attribs.initAttribInstanceFloatArray(m_FloatProps);
			Attribs.initAttribInstanceMat4Array(m_Mat4Props);
			Attribs.initAttribInstanceMat3Array(m_Mat3Props);
		}

		AttributeValues() {
		}
	};

};

#endif



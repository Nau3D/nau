#ifndef ATTRIBUTEVALUES_H
#define ATTRIBUTEVALUES_H

#include <nau/attribute.h>
#include <nau/math/mat3.h>
#include <nau/math/mat4.h>

#include <map>
#include <string>
#include <assert.h>

#define STRING_PROP(A,B) static const StringProperty A = (StringProperty)B 
#define ENUM_PROP(A,B) static const EnumProperty A = (EnumProperty)B 
#define INT_PROP(A,B) static const IntProperty A = (IntProperty)B 
#define UINT_PROP(A,B) static const UIntProperty A = (UIntProperty)B 
#define BOOL_PROP(A,B) static const BoolProperty A = (BoolProperty)B 
#define BOOL4_PROP(A,B) static const Bool4Property A = (Bool4Property)B 
#define FLOAT_PROP(A,B) static const FloatProperty A = (FloatProperty)B 
#define FLOAT2_PROP(A,B) static const Float2Property A = (Float2Property)B 
#define FLOAT4_PROP(A,B) static const Float4Property A = (Float4Property)B 
#define MAT4_PROP(A,B) static const Mat4Property A = (Mat4Property)B 
#define MAT3_PROP(A,B) static const Mat3Property A = (Mat3Property)B 

namespace nau {
	class AttributeValues {

	protected:
		AttribSet m_Attribs;

	public:
		/// STRING
		//typedef enum {} StringProperty;
		//std::map<int, std::string> m_StringProps;

		//std::string &getProps(StringProperty prop) {
		//	return m_StringProps[prop];
		//}

		// ENUMS
		typedef enum {} EnumProperty;
		std::map<int, int> m_EnumProps;

		virtual int getPrope(EnumProperty prop) {
			return m_EnumProps[prop];
		}

		virtual bool isValide(EnumProperty prop, int value) {

			if (m_EnumProps.count(prop) == 0)
				return false;

			Attribute attr = m_Attribs.get(prop, Enums::ENUM);
			if (attr.isValid(value)) 
				return true;
			else
				return false;
		}

		virtual void setPrope(EnumProperty prop, int value) {
			assert(m_EnumProps.count(prop) > 0 && isValide(prop,value));
			m_EnumProps[prop] = value;
		}

		// INT
		typedef enum {} IntProperty;
		std::map<int,int> m_IntProps;

		virtual int getPropi(IntProperty prop) {
			return m_IntProps[prop];
		}

		virtual bool isValidi(IntProperty prop, int value) {

			if (m_IntProps.count(prop) == 0)
				return false;

			Attribute attr = m_Attribs.get(prop, Enums::INT);
			int *max, *min;
			if (attr.getRangeDefined()) {
				max = (int *)attr.getMax();
				min = (int *)attr.getMin();

				if (max != NULL && value > *max)
					return false;
				if (min != NULL && value < *min)
					return false;
			}
			return true;
		}

		virtual void setPropi(IntProperty prop, int value) {
			assert(m_IntProps.count(prop) > 0 && isValidi(prop, value));
			m_IntProps[prop] = value;
		}

		// UINT
		typedef enum {} UIntProperty;
		std::map<int,unsigned int> m_UIntProps;

		virtual unsigned int getPropui(UIntProperty prop) {
			return m_UIntProps[prop];
		}

		virtual bool isValidui(UIntProperty prop, unsigned int value) {

			if (m_UIntProps.count(prop) == 0)
				return false;

			Attribute attr = m_Attribs.get(prop, Enums::UINT);
			unsigned int *max, *min;
			if (attr.getRangeDefined()) {
				max = (unsigned int *)attr.getMax();
				min = (unsigned int *)attr.getMin();

				if (max != NULL && value > *max)
					return false;
				if (min != NULL && value < *min)
					return false;
			}
			return true;
		}

		virtual void setPropui(UIntProperty prop, int unsigned value) {
			assert(m_UIntProps.count(prop) > 0 && isValidui(prop, value));
			m_UIntProps[prop] = value;
		}

		// BOOL
		typedef enum {} BoolProperty;
		std::map<int,bool> m_BoolProps;

		virtual bool getPropb(BoolProperty prop) {
			return m_BoolProps[prop];
		}

		virtual bool isValidb(BoolProperty prop, bool value) {

			if (m_BoolProps.count(prop) == 0)
				return false;
			else
				return true;
		}

		virtual void setPropb(BoolProperty prop, bool value) {
			assert(m_BoolProps.count(prop) > 0);
			m_BoolProps[prop] = value;
		}

		// BOOL4
		typedef enum {} Bool4Property;
		std::map<int, bvec4> m_Bool4Props;

		virtual bvec4 &getPropb4(Bool4Property prop) {
			return m_Bool4Props[prop];
		}

		virtual bool isValidb4(Bool4Property prop, bvec4 &value) {

			if (m_Bool4Props.count(prop) == 0)
				return false;
			else
				return true;
		}

		virtual void setPropb4(Bool4Property prop, bvec4 &value) {
			assert(m_Bool4Props.count(prop) > 0);
			m_Bool4Props[prop] = value;
		}

		// FLOAT
		typedef enum {} FloatProperty;
		std::map<int, float> m_FloatProps;

		virtual float getPropf(FloatProperty prop) {
			return m_FloatProps[prop];
		}


		virtual bool isValidf(FloatProperty prop, float f) {

			if (m_FloatProps.count(prop) == 0)
				return false;

			Attribute attr = m_Attribs.get(prop, Enums::FLOAT);
			float *max, *min;
			if (attr.getRangeDefined()) {
				max = (float *)attr.getMax();
				min = (float *)attr.getMin();

				if (max != NULL && f > *max)
					return false;
				if (min != NULL && f < *min)
					return false;
			}
			return true;
		}

		virtual void setPropf(FloatProperty prop, float value) {
			assert(m_FloatProps.count(prop) > 0 && isValidf(prop, value));
			m_FloatProps[prop] = value;
		}

		// VEC4
		typedef enum {} Float4Property;
		std::map<int, vec4> m_Float4Props;

		virtual vec4 &getPropf4(Float4Property prop) {
			return m_Float4Props[prop];
		}

		virtual bool isValidf4(Float4Property prop, vec4 &f) {

			if (m_Float4Props.count(prop) == 0)
				return false;

			Attribute attr = m_Attribs.get(prop, Enums::VEC4);
			vec4 *max, *min;
			if (attr.getRangeDefined()) {
				max = (vec4 *)attr.getMax();
				min = (vec4 *)attr.getMin();

				if (max != NULL && (f.x > max->x || f.y > max->y || f.z > max->z || f.w > max->w))
					return false;
				if (min != NULL && (f.x < min->x || f.y < min->y || f.z < min->z || f.w < min->w))
					return false;
			}
			return true;
		}

		virtual void setPropf4(Float4Property prop, vec4 &value) {
			assert(m_Float4Props.count(prop) > 0 && isValidf4(prop, value));
			m_Float4Props[prop] = value;
		}


		// VEC2
		typedef enum {} Float2Property;
		std::map<int, vec2> m_Float2Props;

		virtual vec2 &getPropf2(Float2Property prop) {
			return m_Float2Props[prop];
		}

		virtual bool isValidf2(Float2Property prop, vec2 &f) {

			if (m_Float2Props.count(prop) == 0)
				return false;

			Attribute attr = m_Attribs.get(prop, Enums::VEC2);
			vec2 *max, *min;
			if (attr.getRangeDefined()) {
				max = (vec2 *)attr.getMax();
				min = (vec2 *)attr.getMin();

				if (max != NULL && (f.x > max->x || f.y > max->y ))
					return false;
				if (min != NULL && (f.x < min->x || f.y < min->y ))
					return false;
			}
			return true;
		}

		virtual void setPropf2(Float2Property prop, vec2 &value) {
			assert(m_Float2Props.count(prop) > 0 && isValidf2(prop, value));
			m_Float2Props[prop] = value;
		}

		// MAT4
		typedef enum {} Mat4Property;
		std::map<int, mat4> m_Mat4Props;

		virtual const mat4 &getPropm4(Mat4Property prop) {
			return m_Mat4Props[prop];
		}

		virtual void setPropm4(Mat4Property prop, mat4 &value) {
			assert(m_Mat4Props.count(prop) > 0);
			m_Mat4Props[prop] = value;
		}

		// MAT3
		typedef enum {} Mat3Property;
		std::map<int, mat3> m_Mat3Props;

		virtual const mat3 &getPropm3(Mat3Property prop) {
			return m_Mat3Props[prop];
		}

		virtual void setPropm3(Mat3Property prop, mat3 &value) {
			assert(m_Mat3Props.count(prop) > 0 );
			m_Mat3Props[prop] = value;
		}


		void copy(AttributeValues *to) {

			//to->m_StringProps = m_StringProps;
			to->m_EnumProps = m_EnumProps;
			to->m_IntProps =    m_IntProps;
			to->m_UIntProps =   m_UIntProps;
			to->m_BoolProps =   m_BoolProps;
			to->m_Bool4Props =  m_Bool4Props;
			to->m_FloatProps =  m_FloatProps;
			to->m_Float2Props = m_Float2Props;
			to->m_Float4Props = m_Float4Props;
			to->m_Mat3Props =   m_Mat3Props;
			to->m_Mat4Props =   m_Mat4Props;
		}

		virtual void *getProp(int prop, Enums::DataType type) {

			void *res = malloc(sizeof(Enums::getSize(type)));
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
				////case Enums::STRING:
				////	assert(m_StringProps.count(prop) > 0);
				////	return(&(m_StringProps[prop]));
				////	break;
				//case Enums::ENUM:
				//	getPropf((FloatProperty)prop),size);
				//	break;
				//case Enums::INT:
				//	res = getPropi((IntProperty)prop);
				//	break;
				//case Enums::UINT:
				//	res = getPropui((UIntProperty)prop);
				//	break;
				//case Enums::BOOL:
				//	res = getPrope((EnumProperty)prop);
				//	break;
				//case Enums::BVEC4:
				//	res = getPropb4((Bool4Property)prop);
				//	break;
				//case Enums::FLOAT:
				//	res = getPropf((FloatProperty)prop);
				//	break;
				//case Enums::VEC4:
				//	res = getPropf4((Float4Property)prop);
				//	break;
				//case Enums::VEC2:
				//	res = getPropf2((Float2Property)prop);
				//	break;
				//case Enums::MAT4:
				//	res = getPropm4((Mat4Property)prop);
				//	break;
				//case Enums::MAT3:
				//	res = getPropm3((Mat3Property)prop);
				//	break;
				default:
					assert(false && "Missing Data Type in class attributeValues");
					return NULL;
			}
			return res;
		};




		virtual void setProp(int prop, Enums::DataType type, void *value) {

			switch (type) {

			//case Enums::STRING:
			//	assert(m_StringProps.count(prop) > 0);
			//	m_StringProps[prop] = *(std::string *)value;
			//	break;
			case Enums::ENUM:
				setPrope((EnumProperty)prop, *(int *)value);
				break;
			case Enums::INT:
				setPropi((IntProperty)prop, *(int *)value);
				break;
			case Enums::UINT:
				setPropui((UIntProperty)prop, *(unsigned int *)value);
				break;
			case Enums::BOOL:
				setPropb((BoolProperty)prop, *(bool *)value);
				break;
			case Enums::BVEC4:
				setPropb4((Bool4Property)prop, *(bvec4 *)value);
				break;
			case Enums::FLOAT:
				setPropf((FloatProperty)prop, *(float *)value);
				break;
			case Enums::VEC2:
				setPropf2((Float2Property)prop, *(vec2 *)value);
				break;
			case Enums::VEC4:
				setPropf4((Float4Property)prop, *(vec4 *)value);
				break;
			case Enums::MAT4:
				setPropm4((Mat4Property)prop, *(mat4 *)value);
				break;
			case Enums::MAT3:
				setPropm3((Mat3Property)prop, *(mat3 *)value);
				break;
			default:
				assert(false && "Missing Data Type in class attributeValues or Invalid prop");
			}	
		}


		void initArrays(AttribSet Attribs) {
			//Attribs.initAttribInstanceStringArray(m_StringProps);
			Attribs.initAttribInstanceEnumArray(m_EnumProps);
			Attribs.initAttribInstanceIntArray(m_IntProps);
			Attribs.initAttribInstanceUIntArray(m_UIntProps);
			Attribs.initAttribInstanceBoolArray(m_BoolProps);
			Attribs.initAttribInstanceBvec4Array(m_Bool4Props);
			Attribs.initAttribInstanceVec4Array(m_Float4Props);
			Attribs.initAttribInstanceVec2Array(m_Float2Props);
			Attribs.initAttribInstanceFloatArray(m_FloatProps);
			Attribs.initAttribInstanceMat4Array(m_Mat4Props);
			Attribs.initAttribInstanceMat3Array(m_Mat3Props);

			m_Attribs = Attribs;
		}

		AttributeValues() {
		}
		~AttributeValues() {};
	};

};

#endif



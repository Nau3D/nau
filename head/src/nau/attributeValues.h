#ifndef ATTRIBUTEVALUES_H
#define ATTRIBUTEVALUES_H


#include "nau/attribute.h"
#include "nau/math/matrix.h"


#include <map>
#include <string>
#include <assert.h>

#define STRING_PROP(A,B) static const StringProperty A = (StringProperty)B 
#define ENUM_PROP(A,B) static const EnumProperty A = (EnumProperty)B
#define INT_PROP(A,B) static const IntProperty A = (IntProperty)B 
#define INT2_PROP(A,B) static const Int2Property A = (Int2Property)B 
#define INT3_PROP(A,B) static const Int3Property A = (Int3Property)B 
#define UINT_PROP(A,B) static const UIntProperty A = (UIntProperty)B 
#define UINT2_PROP(A,B) static const UInt2Property A = (UInt2Property)B 
#define UINT3_PROP(A,B) static const UInt3Property A = (UInt3Property)B 

#define BOOL_PROP(A,B) static const BoolProperty A = (BoolProperty)B 
#define BOOL4_PROP(A,B) static const Bool4Property A = (Bool4Property)B 
#define FLOAT_PROP(A,B) static const FloatProperty A = (FloatProperty)B 
#define FLOAT2_PROP(A,B) static const Float2Property A = (Float2Property)B 
#define FLOAT4_PROP(A,B) static const Float4Property A = (Float4Property)B 
#define FLOAT3_PROP(A,B) static const Float3Property A = (Float3Property)B 
#define MAT4_PROP(A,B) static const Mat4Property A = (Mat4Property)B 
#define MAT3_PROP(A,B) static const Mat3Property A = (Mat3Property)B 

using namespace nau;

namespace nau {

	class AttributeValues {

	protected:
		AttribSet *m_Attribs;


	public:
		bool isValid(Enums::DataType dt, unsigned int prop, Data *value);
	// ENUM
	protected:
		std::map<int, int> m_EnumProps;

	public:
		typedef enum {} EnumProperty;

		virtual int getPrope(EnumProperty prop);
		virtual bool isValide(EnumProperty prop, int value);
		virtual void setPrope(EnumProperty prop, int value);

		static const int getNextAttrib() { return NextAttrib++;  }

	protected:
		static int NextAttrib;

	// INT
	protected:
		std::map<int,int> m_IntProps;

	public:
		typedef enum {} IntProperty;

		virtual int getPropi(IntProperty prop);
		virtual bool isValidi(IntProperty prop, int value);
		virtual void setPropi(IntProperty prop, int value);

	// INT2
	protected:
		std::map<int, ivec2> m_Int2Props;

	public:
		typedef enum {} Int2Property;

		virtual ivec2& getPropi2(Int2Property prop);
		virtual bool isValidi2(Int2Property prop, ivec2 &value);
		virtual void setPropi2(Int2Property prop, ivec2 &value);

	// INT3
	protected:
		std::map<int, ivec3> m_Int3Props;

	public:
		typedef enum {} Int3Property;

		virtual ivec3& getPropi3(Int3Property prop);
		virtual bool isValidi3(Int3Property prop, ivec3 &value);
		virtual void setPropi3(Int3Property prop, ivec3 &value);
		
	// UINT
	protected:
		std::map<int,unsigned int> m_UIntProps;

	public:
		typedef enum {} UIntProperty;

		virtual unsigned int getPropui(UIntProperty prop);
		virtual bool isValidui(UIntProperty prop, unsigned int value);
		virtual void setPropui(UIntProperty prop, int unsigned value);

	// UINT2
	protected:
		std::map<int, uivec2> m_UInt2Props;

	public:
		typedef enum {} UInt2Property;

		virtual uivec2 &getPropui2(UInt2Property prop);
		virtual bool isValidui2(UInt2Property prop, uivec2 &value);
		virtual void setPropui2(UInt2Property prop, uivec2 &value);

	// UINT3
	protected:
		std::map<int, uivec3> m_UInt3Props;

	public:
		typedef enum {} UInt3Property;

		virtual uivec3 &getPropui3(UInt3Property prop);
		virtual bool isValidui3(UInt3Property prop, uivec3 &value);
		virtual void setPropui3(UInt3Property prop, uivec3 &value);

		
		// BOOL
	protected:
		std::map<int,bool> m_BoolProps;

	public:
		typedef enum {} BoolProperty;

		virtual bool getPropb(BoolProperty prop);
		virtual bool isValidb(BoolProperty prop, bool value);
		virtual void setPropb(BoolProperty prop, bool value);

	// BOOL4
	protected:
		std::map<int, bvec4> m_Bool4Props;

	public:
		typedef enum {} Bool4Property;

		virtual bvec4 &getPropb4(Bool4Property prop);
		virtual bool isValidb4(Bool4Property prop, bvec4 &value);
		virtual void setPropb4(Bool4Property prop, bvec4 &value);

	// FLOAT
	protected:
		std::map<int, float> m_FloatProps;

	public:
		typedef enum {} FloatProperty;

		virtual float getPropf(FloatProperty prop);
		virtual bool isValidf(FloatProperty prop, float f);
		virtual void setPropf(FloatProperty prop, float value);

	// VEC4
	protected:
		std::map<int, vec4> m_Float4Props;

	public:
		typedef enum {} Float4Property;

		virtual vec4 &getPropf4(Float4Property prop);
		virtual bool isValidf4(Float4Property prop, vec4 &f);
		virtual void setPropf4(Float4Property prop, vec4 &value);
		virtual void setPropf4(Float4Property prop, float x, float y, float z, float w);

	// VEC3
	protected:
		std::map<int, vec3> m_Float3Props;

	public:
		typedef enum {} Float3Property;

		virtual vec3 &getPropf3(Float3Property prop);
		virtual bool isValidf3(Float3Property prop, vec3 &f);
		virtual void setPropf3(Float3Property prop, vec3 &value);
		virtual void setPropf3(Float3Property prop, float x, float y, float z);
		
	// VEC2
	protected:
		std::map<int, vec2> m_Float2Props;

	public:
		typedef enum {} Float2Property;

		virtual vec2 &getPropf2(Float2Property prop);
		virtual bool isValidf2(Float2Property prop, vec2 &f);
		virtual void setPropf2(Float2Property prop, vec2 &value);

	// MAT4
	protected:
		std::map<int, mat4> m_Mat4Props;

	public:
		typedef enum {} Mat4Property;

		virtual const mat4 &getPropm4(Mat4Property prop);
		virtual bool isValidm4(Mat4Property prop, mat4 &value);
		virtual void setPropm4(Mat4Property prop, mat4 &value);

	// MAT3
	protected:
		std::map<int, mat3> m_Mat3Props;
		void initArrays(AttribSet  &attribs);

	public:
		typedef enum {} Mat3Property;

		virtual const mat3 &getPropm3(Mat3Property prop);
		virtual bool isValidm3(Mat3Property prop, mat3 &value);
		virtual void setPropm3(Mat3Property prop, mat3 &value);

	// All

		void copy(AttributeValues *to);
		void clearArrays();

		virtual void *getProp(unsigned int prop, Enums::DataType type);
		virtual void setProp(unsigned int prop, Enums::DataType type, Data *value);
		virtual bool isValid(unsigned int prop, Enums::DataType type, Data *value);
		void registerAndInitArrays(AttribSet  &attribs);
		void initArrays();
		AttribSet *getAttribSet();

		AttributeValues();
		AttributeValues(const AttributeValues &mt);

		~AttributeValues();

		const AttributeValues& operator =(const AttributeValues &mt);
	};

};

#endif



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
#define INT4_PROP(A,B) static const Int4Property A = (Int4Property)B

#define UINT_PROP(A,B) static const UIntProperty A = (UIntProperty)B 
#define UINT2_PROP(A,B) static const UInt2Property A = (UInt2Property)B 
#define UINT3_PROP(A,B) static const UInt3Property A = (UInt3Property)B 
#define UINT4_PROP(A,B) static const UInt4Property A = (UInt4Property)B 

#define BOOL_PROP(A,B) static const BoolProperty A = (BoolProperty)B 
#define BOOL2_PROP(A,B) static const Bool2Property A = (Bool2Property)B 
#define BOOL3_PROP(A,B) static const Bool3Property A = (Bool3Property)B 
#define BOOL4_PROP(A,B) static const Bool4Property A = (Bool4Property)B 

#define FLOAT_PROP(A,B) static const FloatProperty A = (FloatProperty)B 
#define FLOAT2_PROP(A,B) static const Float2Property A = (Float2Property)B 
#define FLOAT4_PROP(A,B) static const Float4Property A = (Float4Property)B 
#define FLOAT3_PROP(A,B) static const Float3Property A = (Float3Property)B 
#define MAT4_PROP(A,B) static const Mat4Property A = (Mat4Property)B 
#define MAT3_PROP(A,B) static const Mat3Property A = (Mat3Property)B 
#define MAT2_PROP(A,B) static const Mat2Property A = (Mat2Property)B 

#define DOUBLE_PROP(A,B) static const DoubleProperty A = (DoubleProperty)B 
#define DOUBLE2_PROP(A,B) static const Double2Property A = (Double2Property)B 
#define DOUBLE4_PROP(A,B) static const Double4Property A = (Double4Property)B 
#define DOUBLE3_PROP(A,B) static const Double3Property A = (Double3Property)B 
#define DMAT4_PROP(A,B) static const DMat4Property A = (DMat4Property)B 
#define DMAT3_PROP(A,B) static const DMat3Property A = (DMat3Property)B 
#define DMAT2_PROP(A,B) static const DMat2Property A = (DMat2Property)B 
using namespace nau;

namespace nau {

	class AttributeValues {

	protected:
		AttribSet *m_Attribs;


	public:
		bool isValid(Enums::DataType dt, unsigned int prop, Data *value);

	// STRING
	protected:
		std::map<int, std::string> m_StringProps;
		std::string m_DummyString = "";

	public:
		typedef enum {} StringProperty;


		virtual const std::string &getProps(StringProperty prop);
		virtual void setProps(StringProperty prop, std::string &value);
		virtual bool isValids(StringProperty, std::string value);
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
		
	// INT4
	protected:
		std::map<int, ivec4> m_Int4Props;

	public:
		typedef enum {} Int4Property;

		virtual ivec4& getPropi4(Int4Property prop);
		virtual bool isValidi4(Int4Property prop, ivec4 &value);
		virtual void setPropi4(Int4Property prop, ivec4 &value);

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

		
	// UINT4
	protected:
		std::map<int, uivec4> m_UInt4Props;

	public:
		typedef enum {} UInt4Property;

		virtual uivec4 &getPropui4(UInt4Property prop);
		virtual bool isValidui4(UInt4Property prop, uivec4 &value);
		virtual void setPropui4(UInt4Property prop, uivec4 &value);


	// BOOL
	protected:
		std::map<int,bool> m_BoolProps;

	public:
		typedef enum {} BoolProperty;

		virtual bool getPropb(BoolProperty prop);
		virtual bool isValidb(BoolProperty prop, bool value);
		virtual void setPropb(BoolProperty prop, bool value);

	// BOOL2
	protected:
		std::map<int, bvec2> m_Bool2Props;

	public:
		typedef enum {} Bool2Property;

		virtual bvec2 &getPropb2(Bool2Property prop);
		virtual bool isValidb2(Bool2Property prop, bvec2 &value);
		virtual void setPropb2(Bool2Property prop, bvec2 &value);

	// BOOL3
	protected:
		std::map<int, bvec3> m_Bool3Props;

	public:
		typedef enum {} Bool3Property;

		virtual bvec3 &getPropb3(Bool3Property prop);
		virtual bool isValidb3(Bool3Property prop, bvec3 &value);
		virtual void setPropb3(Bool3Property prop, bvec3 &value);

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

	public:
		typedef enum {} Mat3Property;

		virtual const mat3 &getPropm3(Mat3Property prop);
		virtual bool isValidm3(Mat3Property prop, mat3 &value);
		virtual void setPropm3(Mat3Property prop, mat3 &value);

	// MAT2
	protected:
		std::map<int, mat2> m_Mat2Props;

	public:
		typedef enum {} Mat2Property;

		virtual const mat2 &getPropm2(Mat2Property prop);
		virtual bool isValidm2(Mat2Property prop, mat2 &value);
		virtual void setPropm2(Mat2Property prop, mat2 &value);

	// DOUBLE
	protected:
		std::map<int, double> m_DoubleProps;

	public:
		typedef enum {} DoubleProperty;

		virtual double getPropd(DoubleProperty prop);
		virtual bool isValidd(DoubleProperty prop, double f);
		virtual void setPropd(DoubleProperty prop, double value);

		// DVEC4
	protected:
		std::map<int, dvec4> m_Double4Props;

	public:
		typedef enum {} Double4Property;

		virtual dvec4 &getPropd4(Double4Property prop);
		virtual bool isValidd4(Double4Property prop, dvec4 &f);
		virtual void setPropd4(Double4Property prop, dvec4 &value);
		virtual void setPropd4(Double4Property prop, double x, double y, double z, double w);

		// DVEC3
	protected:
		std::map<int, dvec3> m_Double3Props;

	public:
		typedef enum {} Double3Property;

		virtual dvec3 &getPropd3(Double3Property prop);
		virtual bool isValidd3(Double3Property prop, dvec3 &f);
		virtual void setPropd3(Double3Property prop, dvec3 &value);
		virtual void setPropd3(Double3Property prop, double x, double y, double z);

		// DVEC2
	protected:
		std::map<int, dvec2> m_Double2Props;

	public:
		typedef enum {} Double2Property;

		virtual dvec2 &getPropd2(Double2Property prop);
		virtual bool isValidd2(Double2Property prop, dvec2 &f);
		virtual void setPropd2(Double2Property prop, dvec2 &value);
		

		// DMAT4
	protected:
		std::map<int, dmat4> m_DMat4Props;

	public:
		typedef enum {} DMat4Property;

		virtual const dmat4 &getPropdm4(DMat4Property prop);
		virtual bool isValiddm4(DMat4Property prop, dmat4 &value);
		virtual void setPropdm4(DMat4Property prop, dmat4 &value);

		// DMAT3
	protected:
		std::map<int, dmat3> m_DMat3Props;

	public:
		typedef enum {} DMat3Property;

		virtual const dmat3 &getPropdm3(DMat3Property prop);
		virtual bool isValiddm3(DMat3Property prop, dmat3 &value);
		virtual void setPropdm3(DMat3Property prop, dmat3 &value);

		// DMAT2
	protected:
		std::map<int, dmat2> m_DMat2Props;

	public:
		typedef enum {} DMat2Property;

		virtual const dmat2 &getPropdm2(DMat2Property prop);
		virtual bool isValiddm2(DMat2Property prop, dmat2 &value);
		virtual void setPropdm2(DMat2Property prop, dmat2 &value);

		// All

		void copy(AttributeValues *to);
		void clearArrays();

		virtual void *getProp(unsigned int prop, Enums::DataType type);
		virtual void setProp(unsigned int prop, Enums::DataType type, Data *value);
		virtual bool isValid(unsigned int prop, Enums::DataType type, Data *value);
		void registerAndInitArrays(AttribSet  &attribs);
		void initArrays();
		void initArrays(AttribSet  &attribs);
		AttribSet *getAttribSet();

		AttributeValues();
		AttributeValues(const AttributeValues &mt);

		~AttributeValues();

		const AttributeValues& operator =(const AttributeValues &mt);
	};

};

#endif



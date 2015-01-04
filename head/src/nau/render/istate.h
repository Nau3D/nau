#ifndef __STATE__
#define __STATE__

#include <nau/math/vec4.h>
#include <nau/math/bvec4.h>
#include <nau/attributeValues.h>
#include <nau/attribute.h>

#include <string>
#include <vector>
#include <map>

using namespace nau::math;

namespace nau
{
	namespace render
	{
		class IState: public AttributeValues {

		public:

			ENUM_PROP(DEPTH_FUNC, 0);
			ENUM_PROP(CULL_TYPE, 1);
			ENUM_PROP(ORDER_TYPE, 2);
			ENUM_PROP(BLEND_SRC, 3);
			ENUM_PROP(BLEND_DST, 4);
			ENUM_PROP(BLEND_EQUATION, 5);

			BOOL_PROP(BLEND, 0);
			BOOL_PROP(DEPTH_TEST, 1);
			BOOL_PROP(CULL_FACE, 2);
			BOOL_PROP(COLOR_MASK, 3);
			BOOL_PROP(DEPTH_MASK, 4);

			INT_PROP(ORDER, 0);

			FLOAT4_PROP(BLEND_COLOR, 0);

			BOOL4_PROP(COLOR_MASK_B4, 0);

			//typedef enum  {
			//	, FOG, ALPHA_TEST, ,
			//	,, , COUNT_BOOLPROPERTY} BoolProperty; 

			//typedef enum {FOG_MODE,FOG_COORD_SRC,
			//		, , , 
			//		, , ,ALPHA_FUNC,COUNT_ENUMPROPERTY} EnumProperty;

			//typedef enum {FOG_START, FOG_END, FOG_DENSITY, ALPHA_VALUE,COUNT_FLOATPROPERTY} FloatProperty;

			//typedef enum {,COUNT_INTPROPERTY} IntProperty;

			//typedef enum   {, FOG_COLOR,COUNT_FLOAT4PROPERTY} Float4Property;

			//typedef enum {COLOR_MASK_B4, COUNT_BOOL4PROPERTY} Bool4Property;

			typedef enum  { FRONT_TO_BACK, BACK_TO_FRONT, NONE} OrderType;

			static AttribSet Attribs;

			virtual ~IState(void) {};

			static IState* create();
			virtual void setDefault();

			virtual void setName(std::string name);
			virtual std::string getName();

			virtual IState* clone();
			//virtual void clear();

			virtual void setDiff(IState *defState, IState *pState) = 0;

			//virtual void setProp(BoolProperty ss, bool value);
			//virtual void setProp(EnumProperty prop, int value);
			//virtual void setProp(FloatProperty prop, float value);
			//virtual void setProp(IntProperty prop, int value);
			//virtual void setProp(Float4Property prop, float r, float g, float b, float a);
			//virtual void setProp(Float4Property prop, const vec4& color);
			//virtual void setProp(Bool4Property prop, bool r, bool g, bool b, bool a);
			//virtual void setProp(Bool4Property prop, bool* values); 
			//virtual void setProp(Bool4Property prop, bvec4& values); 
			//// Note: no validation is performed!
			//void setProp(int prop, Enums::DataType type, void *value);

			//virtual bool getPropb(BoolProperty ss);
			//virtual int getPrope(EnumProperty prop);
			//virtual float getPropf(FloatProperty prop);
			//virtual int getPropi(IntProperty prop);
			//virtual const vec4& getProp4f(Float4Property prop);
			//virtual const bvec4& getProp4b(Bool4Property prop);
			//void *getProp(int prop, Enums::DataType type);

			//virtual bool isSetProp(EnumProperty prop);
			//virtual bool isSetProp(FloatProperty prop);
			//virtual bool isSetProp(IntProperty prop);
			//virtual bool isSetProp(Float4Property prop);
			//virtual bool isSetProp(Bool4Property prop);

		protected:
			IState();
			
			static bool Init();
			static bool Inited;

			std::string m_Name;
			vec4 m_defColor;
			bvec4 m_defBoolV;

			//std::map<int, int> m_IntProps;
			//std::map<int, bool> m_EnableProps;
			//std::map<int, float> m_FloatProps;
			//std::map<int, vec4> m_Float4Props;
			//std::map<int, bvec4> m_Bool4Props;
			//std::map<int, int> m_EnumProps;
		};
	};
};

#endif

#ifndef __STATE__
#define __STATE__

#include "nau/math/vec4.h"
//#include "nau/math/bvec4.h"
#include "nau/attributeValues.h"
#include "nau/attribute.h"

#include <string>
#include <vector>
#include <map>

using namespace nau::math;

namespace nau
{
	namespace material
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
			BOOL_PROP(DEPTH_MASK, 4);

			INT_PROP(ORDER, 0);

			FLOAT4_PROP(BLEND_COLOR, 0);

			BOOL4_PROP(COLOR_MASK_B4, 0);

			typedef enum  { FRONT_TO_BACK, BACK_TO_FRONT, NONE} OrderType;

			static AttribSet Attribs;

			virtual ~IState(void) {};

			static IState* create();
			virtual void setDefault();

			virtual void setName(std::string name);
			virtual std::string getName();

			virtual IState* clone();

			virtual void setDiff(IState *defState, IState *pState) = 0;

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

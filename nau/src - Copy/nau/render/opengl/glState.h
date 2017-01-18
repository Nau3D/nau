#ifndef GLSTATE_H
#define GLSTATE_H

#include "nau/material/iState.h"

#include <string>


using namespace nau::material;

namespace nau
{
	namespace render
	{
		class GLState: public IState {

		public:

			GLState();
			~GLState();

			virtual void set();
			void setDiff(IState *defState, IState *pState);

		private:

			static bool Inited;
			static bool InitGL();

			bool difColor(vec4& a, vec4& b);
			bool difBoolVector(bool* a, bool* b);

			float* toFloatPtr(vec4& v);

		};
	};
};

#endif //GLSTATE_H

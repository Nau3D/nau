#ifndef GLSTATE_H
#define GLSTATE_H

#include <nau/render/istate.h>
#include <string>
#include <GL/glew.h>


namespace nau
{
	namespace render
	{
		class GlState: public IState {

		public:

			GlState();
			~GlState();

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

#ifndef GLGLOBALSTATE_H
#define GLGLOBALSTATE_H

#include "nau/render/iGlobalState.h"

#include <string>


using namespace nau::render;

namespace nau
{
	namespace render
	{
		class GLGlobalState : public IGlobalState {

		public:
			GLGlobalState();

		protected:
			std::string getStateValue(unsigned int enumValue, FunctionType type);
			std::string getStateValue(unsigned int enumValue, 
									FunctionType type, 
									unsigned int length);
		};
	};
};

#endif
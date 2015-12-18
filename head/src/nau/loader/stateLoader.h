#ifndef NAU_STATE_LOADER_H
#define NAU_STATE_LOADER_H

#include <nau/render/iGlobalState.h>

#include <tinyxml.h>

#include <string>

using namespace nau::render;

namespace nau
{
	namespace loader
	{
		class StateLoader
		{
		public:
			static void LoadStateXMLFile(std::string file, IGlobalState *state);

		protected:
			static void LoadFunctionEnums(TiXmlHandle &handle, IGlobalState *state);
			static void LoadEnums(TiXmlHandle &hRoot, std::string functionName,
				IGlobalState *state);
		};
	};
};

#endif
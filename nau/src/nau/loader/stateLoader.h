#ifndef NAU_STATE_LOADER_H
#define NAU_STATE_LOADER_H

#include <nau/render/iGlobalState.h>

#include <string>

using namespace nau::render;

#ifdef _WINDLL
#ifdef nau_EXPORTS
#define nau_API __declspec(dllexport)   
#else  
#define nau_API __declspec(dllimport)   
#endif 
#else
#define nau_API
#endif

class TiXmlHandle;

namespace nau
{
	namespace loader
	{
		class StateLoader
		{
		public:
			static nau_API void LoadStateXMLFile(std::string file, IGlobalState *state);

		protected:
			static void LoadFunctionEnums(TiXmlHandle &handle, IGlobalState *state);
			static void LoadEnums(TiXmlHandle &hRoot, std::string functionName,
				IGlobalState *state);
		};
	};
};

#endif
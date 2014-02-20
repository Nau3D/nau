#ifndef CBSLOADER_H
#define CBSLOADER_H

#include <nau/scene/iscene.h>

namespace nau 
{

	namespace loader 
	{
		class CBSLoader
		{
		public:
			static void loadScene (nau::scene::IScene *aScene, std::string &aFilename);
			static void writeScene (nau::scene::IScene *aScene, std::string &aFilename);

		private:
			CBSLoader(void) {};
			~CBSLoader(void) {};
		};
	};
};

#endif //CBSLOADER_H

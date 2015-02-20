#ifndef PATCHLOADER_H
#define PATCHLOADER_H

#include "nau/scene/iscene.h"

namespace nau 
{

	namespace loader 
	{
		// Loads Bezier Patches
		class PatchLoader
		{
		public:
			// Load Scene
			static void loadScene (nau::scene::IScene *aScene, std::string &aFilename);
		};
	};
};

#endif 
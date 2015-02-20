#ifndef OBJLOADER_H
#define OBJLOADER_H

#include "nau/scene/iscene.h"

namespace nau 
{

	namespace loader 
	{
		// Wavefront OBJ format loader
		// Uses Nate Robin's GLM implementation as the core
		class OBJLoader
		{
		public:
			// Load Scene
			static void loadScene (nau::scene::IScene *aScene, std::string &aFilename);
			// Write Scene
			static void writeScene (nau::scene::IScene *aScene, std::string &aFilename);

		private:
			// Constructor
			OBJLoader(void) {};
			// Destructor
			~OBJLoader(void) {};
		};
	};
};

#endif //OBJLOADER_H
#ifndef ASSIMPLOADER_H
#define ASSIMPLOADER_H

// assimp include files. These three are usually needed.
#include "assimp/Importer.hpp"	//OO version Header!
#include "assimp/PostProcess.h"
#include "assimp/Scene.h"

#include "nau/scene/iscene.h"

namespace nau 
{

	namespace loader 
	{
		// Assimp Loader Interface
		class AssimpLoader
		{
		public:
			// Load Scene
			static void loadScene (nau::scene::IScene *aScene, std::string &aFilename, std::string &params = std::string("") );
			// Write Scene
			static void writeScene (nau::scene::IScene *aScene, std::string &aFilename);

		private:

			enum {
				XYZ,
				XZ_Y
			};

			// Constructor
			AssimpLoader(void) {};
			// Destructor
			~AssimpLoader(void) {};

			static Assimp::Importer importer;

			static std::vector<nau::math::vec4>* readGL3FArray(float* a, unsigned int arraysize, unsigned int order, float w);
			static std::vector<nau::math::vec4>* readGL2FArray(float* a, unsigned int arraysize);
			static void color4_to_float4(const  aiColor4D *c, float f[4]);
			static void set_float4(float f[4], float a, float b, float c, float d);
			static void recursiveWalk (nau::scene::IScene *aScene, std::string &aFilename,
									const  aiScene *sc, const  aiNode* nd,
									mat4 &m, std::map<unsigned int, std::string> meshNameMap);
		};
	};
};

#endif //ASSIMPLOADER_H
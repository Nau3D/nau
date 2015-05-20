#ifndef CBOLOADER_H
#define CBOLOADER_H

#include "nau/scene/iscene.h"
#include "nau/render/vertexdata.h"
#include "nau/material/material.h"
#include "nau/scene/octreeByMatscene.h"

using namespace nau::render;
using namespace nau::material;
using namespace nau::scene;

namespace nau 
{

	namespace loader 
	{
		class CBOLoader
		{
		public:	
			static void loadScene (nau::scene::IScene *aScene, std::string &aFilename, std::string &params);
			static void writeScene (nau::scene::IScene *aScene, std::string &aFilename);

		private:
			CBOLoader(void) {};
			~CBOLoader(void) {};

			static std::string m_FileName;

			static void _writeMaterial(std::string matName, std::string path, std::fstream &f);
			static void _readMaterial(std::string path, std::fstream &f);
			static void _writeVertexData (VertexData& aVertexData, std::fstream &f) ;
			static void _readVertexData (VertexData& aVertexData, std::fstream &f);
			static void _writeIndexData (IndexData& aVertexData, std::fstream &f) ;
			static void _readIndexData (IndexData& aVertexData, std::fstream &f);
			static void _ignoreVertexData (std::fstream &f);
			static void _writeString (const std::string& aString, std::fstream &f);
			static void _readString ( char *buffer, std::fstream &f);
			static void _ignoreString (std::fstream &f);

			static void _writeOctreeByMat(OctreeByMatScene *aScene, std::fstream &f);
			static void _writeOctreeByMatNode(OctreeByMatNode *n, std::fstream &f)	;	
			static void _writeOctreeByMatSceneObject(SceneObject *so, std::fstream &f)	;

			static void _readOctreeByMat(OctreeByMatScene *aScene, std::fstream &f);
			static void _readOctreeByMatNode(OctreeByMatNode *n, std::fstream &f)	;	
			static void _readOctreeByMatSceneObject(SceneObject *so, std::fstream &f)	;
		};
	};
};

#endif //CBOLOADER_H

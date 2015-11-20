#ifndef OGREMESHLOADER_H
#define OGREMESHLOADER_H

#include "nau/system/file.h"
#include "nau/geometry/meshBones.h"
#include "nau/material/materialGroup.h"
#include "nau/material/materialLib.h"
#include "nau/math/vec3.h"
#include "nau/scene/iScene.h"
#include "nau/scene/scenePoses.h"
#include "nau/scene/sceneSkeleton.h"

#include <tinyxml.h>

#include <string>
#include <vector>

using namespace nau::material;
using namespace nau::math;
using namespace nau::render;
using namespace nau::scene;
using namespace nau::system;

namespace nau
{
	namespace loader
	{
		class OgreMeshLoader
		{
		public:
			static void loadScene (IScene* aScene, std::string file) throw (std::string);
			static std::string m_Path;
			static std::string m_MeshFile;
			static std::string m_SkeletonFile;

		private:

			static std::string m_MeshType;

			typedef enum {
				SIMPLE,
				POSE,
				BONES
			}MeshTypes;

			OgreMeshLoader(void);

			static void loadVertexElement(TiXmlElement *pElemVertexAttrib, VertexData::Attr *vertexElem) ;
			static void loadTextureCoordElement(TiXmlElement *pElemVertexAttrib, VertexData::Attr *vertexElem);
			static void loadVertexBuffer(TiXmlElement *pElemVertexBuffer, std::shared_ptr<VertexData> &vertexData);
			static void loadSubMeshes (TiXmlHandle hRoot, IScene *scn, IRenderable *m, std::string meshType);
			static void loadGeometry(TiXmlElement *pElem, std::shared_ptr<VertexData> &vertexData);
			static void loadFaces(TiXmlElement *pElem, std::shared_ptr<MaterialGroup> &mg, unsigned int operationType);
			static IRenderable *loadSharedGeometry (TiXmlHandle hRoot, IScene *scn, std::string meshType);
			static void loadSubMeshNames(TiXmlHandle hRoot, IScene *scn, bool meshSharedGeometry);
			static void loadVertexBuffers(TiXmlElement *pElem, std::shared_ptr<VertexData> &vertexData);
			static void loadPoses(TiXmlHandle hRoot, IScene *scn, bool meshSharedGeometry);
			static void loadPoseAnimations(TiXmlHandle hRoot, ScenePoses *scn);
			static void loadBoneAssignements(TiXmlElement *pElem, nau::geometry::MeshBones *mb);
			static void loadSkeleton(TiXmlHandle hRoot, SceneSkeleton *sk) throw (std::string);
			static void loadSkeletonElements(TiXmlHandle hRoot, SceneSkeleton *sk)  throw (std::string);

			enum {
				TRIANGLE_LIST,
				TRIANGLE_STRIP,
				TRIANGLE_FAN
			};
		};
	};
};

#endif //OGREMESHLOADER_H

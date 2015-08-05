/*

IRenderable contains only the vertex attribute vectors. The index vectors are stored in the material groups.

*/

#ifndef MESH_H
#define MESH_H

#include "nau/render/iRenderer.h"
#include "nau/render/iRenderable.h"
#include "nau/material/materialgroup.h"
#include "nau/resource/resourcemanager.h"

#include <set>
#include <string>


namespace nau
{
	namespace geometry
	{
		class Mesh : public nau::render::IRenderable
		{
		protected:
			nau::geometry::VertexData* m_VertexData;
			nau::geometry::IndexData* m_IndexData;
			std::vector<nau::material::MaterialGroup*> m_vMaterialGroups;
			unsigned int m_DrawPrimitive;
			unsigned int m_RealDrawPrimitive;
			std::string m_Name;
			int m_NumberOfPrimitives;
			std::vector<unsigned int> m_UnifiedIndex;
			int m_VerticesPerPatch = 0;
			void createUnifiedIndexVector();
			void prepareIndexData(); 
			Mesh(void);

		public:
			friend class nau::resource::ResourceManager;

			static Mesh *createUnregisteredMesh();
			~Mesh (void);

			void setName (std::string name);
			std::string& getName (void);

			unsigned int getDrawingPrimitive(); 
			unsigned int getRealDrawingPrimitive();
			void setDrawingPrimitive(unsigned int aDrawingPrimitive);

			void prepareTriangleIDs(unsigned int sceneObjectID);
			void unitize(vec3 &center, vec3 &min, vec3 &max);

			void getMaterialNames(std::set<std::string> *nameList);
			void addMaterialGroup (nau::material::MaterialGroup*, int offset = 0);
			void addMaterialGroup (nau::material::MaterialGroup* materialGroup, 
				nau::render::IRenderable *aRenderable); 
			std::vector<nau::material::MaterialGroup*>& getMaterialGroups (void);

			nau::geometry::VertexData& getVertexData (void);
			nau::geometry::IndexData& getIndexData(void);

			int getNumberOfVertices (void);
			void setNumberOfVerticesPerPatch(int i);
			int getnumberOfVerticesPerPatch(void);

			std::string getType (void);
			void resetCompilationFlags();

			void merge (nau::render::IRenderable *aRenderable);
		};
	};
};

#endif

/*

This IRenderable contains only the attribute vectors. The index vectors are stored in the material groups.

*/

#ifndef MESH_H
#define MESH_H

#include <set>
#include <string>

#include <nau/render/irenderer.h>
#include <nau/render/irenderable.h>
#include <nau/material/imaterialgroup.h>
#include <nau/resource/resourcemanager.h>


namespace nau
{
	namespace geometry
	{
		class Mesh : public nau::render::IRenderable
		{
		protected:
			nau::render::VertexData* m_pVertexData;
			nau::render::IndexData* m_IndexData;
			std::vector<nau::material::IMaterialGroup*> m_vMaterialGroups;
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
			void unitize(float min, float max);

			void getMaterialNames(std::set<std::string> *nameList);
			void addMaterialGroup (nau::material::IMaterialGroup*, int offset = 0);
			void addMaterialGroup (nau::material::IMaterialGroup* materialGroup, 
				nau::render::IRenderable *aRenderable); 
			std::vector<nau::material::IMaterialGroup*>& getMaterialGroups (void);

			nau::render::VertexData& getVertexData (void);
			nau::render::IndexData& getIndexData(void);

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

#ifndef MESH_H
#define MESH_H

#include <nau/render/irenderer.h>
#include <nau/render/irenderable.h>
#include <nau/material/imaterialgroup.h>

#include <nau/resource/resourcemanager.h>


#include <string>

namespace nau
{
	namespace geometry
	{
		class Mesh : public nau::render::IRenderable
		{
		protected:
			nau::render::VertexData* m_pVertexData;
			std::vector<nau::material::IMaterialGroup*> m_vMaterialGroups;
			unsigned int m_DrawPrimitive;
			unsigned int m_RealDrawPrimitive;
			std::string m_Name;
			int m_NumberOfPrimitives;
			std::vector<unsigned int> m_UnifiedIndex;
			int m_VerticesPerPatch = 0;
			void createUnifiedIndexVector();
			void prepareIndexData(); 
			void resetCompilationFlags();
			Mesh(void);

		public:
			friend class nau::resource::ResourceManager;

			static Mesh *createUnregisteredMesh();
			virtual ~Mesh (void);

			virtual void setName (std::string name);
			virtual std::string& getName (void);

			unsigned int getDrawingPrimitive(); 
			unsigned int getRealDrawingPrimitive();
			void setDrawingPrimitive(unsigned int aDrawingPrimitive);

			void prepareTriangleIDs(unsigned int sceneObjectID);

			void unitize(float min, float max);

			void addMaterialGroup (nau::material::IMaterialGroup*);
			void addMaterialGroup (nau::material::IMaterialGroup* materialGroup, 
				nau::render::IRenderable *aRenderable); 
			std::vector<nau::material::IMaterialGroup*>& getMaterialGroups (void);
			virtual void getMaterialNames(std::set<std::string> *nameList);

			void merge (nau::render::IRenderable *aRenderable);

			virtual nau::render::VertexData& getVertexData (void);
			virtual int getNumberOfVertices (void);
			void setNumberOfVerticesPerPatch(int i);
			int getnumberOfVerticesPerPatch(void);

			virtual std::string getType (void);

			//int getPrimitiveOffset(void);
			//virtual int getNumberOfPrimitives(void);
		};
	};
};

#endif

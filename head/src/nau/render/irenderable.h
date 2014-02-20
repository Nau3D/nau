#ifndef IRENDERABLE_H
#define IRENDERABLE_H

//#include <nau/render/irenderer.h>
#include <nau/material/imaterialgroup.h>
#include <nau/render/vertexdata.h>
#include <nau/math/vec3.h>

#include <vector>
#include <set>

#include <nau/event/ilistener.h>

using namespace nau::event_;


namespace nau
{
	namespace render 
	{
		class IRenderable: public IListener 
		{
		public:

			virtual void setName (std::string name) = 0;
			virtual std::string& getName () = 0;
			virtual unsigned int getDrawingPrimitive() = 0; 
			virtual unsigned int getRealDrawingPrimitive() = 0;
			virtual void setDrawingPrimitive(unsigned int aDrawingPrimitive) = 0;

			virtual void prepareTriangleIDs(unsigned int sceneObjectID) = 0;
			virtual void unitize(float min, float max) = 0;
			virtual void createUnifiedIndexVector() = 0;

			virtual nau::render::VertexData& getVertexData (void) = 0;
			virtual std::vector<nau::material::IMaterialGroup*>& 
												getMaterialGroups (void) = 0;

			virtual void getMaterialNames(std::set<std::string> *nameList) = 0;
			virtual void addMaterialGroup (nau::material::IMaterialGroup*) = 0;
			virtual void addMaterialGroup (nau::material::IMaterialGroup* materialGroup, 
				nau::render::IRenderable *aRenderable) = 0; 
			virtual int getNumberOfVertices (void) = 0;
			virtual int getNumberOfPrimitives(void) = 0;
			virtual void setNumberOfVerticesPerPrimitive(int i) = 0;
			virtual int getnumberOfVerticesPerPrimitive(void) = 0;
			virtual std::string getType (void) = 0;
			virtual void resetCompilationFlags() = 0;

			virtual ~IRenderable(void) {};
		};
	};
};

#endif //IRENDERABLE_H


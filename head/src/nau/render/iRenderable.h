#ifndef IRENDERABLE_H
#define IRENDERABLE_H

#include "nau/attribute.h"
#include "nau/attributeValues.h"
#include "nau/config.h"
#include "nau/event/ilistener.h"
#include "nau/math/vec3.h"
#include "nau/geometry/indexData.h"
#include "nau/geometry/vertexData.h"

#include <vector>
#include <set>

using namespace nau::event_;

namespace nau
{
	namespace material {

		class MaterialGroup;
	}
	namespace render 
	{
		class IRenderable: public IListener, public AttributeValues
		{
		public:

			typedef enum {
				TRIANGLES = 0,
				TRIANGLE_STRIP,
				TRIANGLE_FAN,
				LINES,
				LINE_LOOP,
				POINTS,
				TRIANGLES_ADJACENCY
//#if NAU_OPENGL_VERSION >= 400
				, PATCHES
//#endif
			} DrawPrimitive;

			ENUM_PROP(PRIMITIVE_TYPE, 0);

			static AttribSet Attribs;

			virtual void setName(std::string name) = 0;
			virtual std::string& getName () = 0;
			virtual unsigned int getDrawingPrimitive() = 0; 
			virtual unsigned int getRealDrawingPrimitive() = 0;
			virtual void setDrawingPrimitive(unsigned int aDrawingPrimitive) = 0;

			virtual void prepareTriangleIDs(unsigned int sceneObjectID) = 0;
			virtual void unitize(vec3 &vCenter, vec3 &vMin, vec3 &vMax) = 0;

			virtual void getMaterialNames(std::set<std::string> *nameList) = 0;
			virtual void addMaterialGroup(std::shared_ptr<nau::material::MaterialGroup> &, int offset=0) = 0;
			virtual void addMaterialGroup(std::shared_ptr<nau::material::MaterialGroup> &,
				nau::render::IRenderable *aRenderable) = 0; 
			virtual std::vector<std::shared_ptr<nau::material::MaterialGroup>>& getMaterialGroups(void) = 0;

			virtual std::shared_ptr<nau::geometry::VertexData>& getVertexData (void) = 0;
			virtual std::shared_ptr<nau::geometry::IndexData>& getIndexData(void) = 0;

			virtual void merge(nau::render::IRenderable *aRenderable) = 0;
			virtual unsigned int getNumberOfVertices(void) = 0;
			virtual void setNumberOfVerticesPerPatch(int i) = 0;
			virtual int getnumberOfVerticesPerPatch(void) = 0;

			virtual std::string getType (void) = 0;
			virtual void resetCompilationFlags() = 0;

			virtual ~IRenderable(void) {};

		protected:
			static bool Init();
			static bool Inited;

			IRenderable();

		};
	};
};

#endif //IRENDERABLE_H


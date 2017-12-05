#ifndef IBOUNDINGVOLUME_H
#define IBOUNDINGVOLUME_H

#include <vector>

#include "nau/math/vec3.h"
#include "nau/math/matrix.h"
#include "nau/geometry/vertexData.h"


using namespace nau::geometry;
using namespace nau::math;

namespace nau
{
	namespace geometry
	{

		class IBoundingVolume
		{
		protected:
			enum BoundingVolumeKind {
			  BOX,
			  SPHERE
			};
        public:
			virtual std::string getClassName() = 0;
			// defines a bounding volume based on two points
			virtual void set(nau::math::vec3 a, nau::math::vec3 b) = 0;
			virtual void calculate (const std::shared_ptr<std::vector<VertexData::Attr>> &vertices) = 0;
			virtual void setTransform (nau::math::mat4 &aTransform) = 0;
			virtual bool intersect (const IBoundingVolume *volume) = 0;
			virtual void compound (const IBoundingVolume *volume) = 0;
			virtual bool isA (BoundingVolumeKind kind) const = 0; /***MARK***/
			virtual std::vector<vec3> &getPoints (void) = 0;

			virtual const nau::math::vec3& getMin (void) const = 0;
			virtual const nau::math::vec3& getMax (void) const = 0;
			virtual const nau::math::vec3& getCenter (void) const = 0;

			virtual ~IBoundingVolume () {}
		};
	};
};
#endif

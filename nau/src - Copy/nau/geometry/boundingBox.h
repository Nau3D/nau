#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H

#include <vector>

#include "nau/geometry/iBoundingVolume.h"
#include "nau/math/matrix.h"
#include "nau/math/vec3.h"
#include "nau/geometry/boundingBoxPrimitive.h"

namespace nau
{
	namespace geometry
	{
		class BoundingBox :
			public nau::geometry::IBoundingVolume
		{
		private:
			enum {
				MIN = 0,
				MAX,
				CENTER
			};
	
			std::vector<nau::math::vec3> m_vPoints,m_vLocalPoints; 

			mat4 m_GeometryTransform; 

		public:

#ifdef NAU_RENDER_FLAGS
			static std::shared_ptr<nau::geometry::BBox> Geometry;
			static nau::geometry::BBox *getGeometry();
#endif
			BoundingBox (void);
			BoundingBox (nau::math::vec3 min, nau::math::vec3 max);
			BoundingBox (const BoundingBox &aBoundingBox);

			void set(nau::math::vec3 min, nau::math::vec3 max);
			void calculate (const std::shared_ptr<std::vector<VertexData::Attr>> &vertices);

			// updates the geometry transform
			void setTransform(nau::math::mat4 &aTransform);
			mat4 &getTransform();


			bool intersect (const IBoundingVolume *volume);
			void compound (const IBoundingVolume *volume);
			
			bool isA (BoundingVolumeKind kind) const;
			std::string getType (void) const;
			std::vector<nau::math::vec3> getPoints (void);
			std::vector<nau::math::vec3>& getNonTransformedPoints (void) ;

			const nau::math::vec3& getMin (void) const;
			const nau::math::vec3& getMax (void) const;

			const nau::math::vec3& getCenter (void) const;
		private:
			void _calculateCenter (void);

		public:
			virtual ~BoundingBox(void);
			void eventReceived(const std::string & sender, const std::string & eventType, 
				const std::shared_ptr<IEventData>& evt);
		};
	};
};

#endif

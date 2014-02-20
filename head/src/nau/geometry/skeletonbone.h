#ifndef SKELETONBONE_H
#define SKELETONBONE_H

#include <nau/math/vec3.h>
#include <nau/math/simpletransform.h>

#include <string>

using namespace nau::math;

namespace nau {

	namespace geometry {
	
		class SkeletonBone {
		
		protected:

			vec3 m_Position, m_RotVector;
			float m_Angle;
			unsigned int m_Id;
			std::string m_Name;
			SimpleTransform m_LocalTransform, m_CompositeTransform;

		public:

			SkeletonBone();
			SkeletonBone(std::string name, unsigned int id, vec3 pos, float angle, vec3 axis);
			~SkeletonBone();

			void setPosition(vec3 pos);
			void setRotation(vec3 axis, float angle);
			void setId(unsigned int i);
			void setName(std::string name);

			unsigned int getID();
			std::string getName();

			SimpleTransform &getFullTransform();
			SimpleTransform &getLocalTransform();


		
		};
	};

};

#endif

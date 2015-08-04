#include "nau/geometry/frustum.h"

#include "nau/geometry/boundingBox.h"
#include "nau/config.h"

using namespace nau::geometry;
using namespace nau::math;
using namespace nau::scene;

Frustum::Frustum(void)
{
}

Frustum::~Frustum(void)
{
}

#ifdef NAU_OPENGL
#define m(row, col) m[col*4+row-5]
#elif NAU_DIRECTX
#endif

void 
Frustum::setFromMatrix (const float *m)
{
	m_Planes[NEARP].setCoefficients(
				 m(3,1) + m(4,1),
				 m(3,2) + m(4,2),
				 m(3,3) + m(4,3),
				 m(3,4) + m(4,4));
	m_Planes[FARP].setCoefficients( 
				-m(3,1) + m(4,1),
				-m(3,2) + m(4,2),
				-m(3,3) + m(4,3),
				-m(3,4) + m(4,4));
	m_Planes[BOTTOM].setCoefficients(
				 m(2,1) + m(4,1),
				 m(2,2) + m(4,2),
				 m(2,3) + m(4,3),
				 m(2,4) + m(4,4));
	m_Planes[TOP].setCoefficients(  
				-m(2,1) + m(4,1),
				-m(2,2) + m(4,2),
				-m(2,3) + m(4,3),
				-m(2,4) + m(4,4));
	m_Planes[LEFT].setCoefficients(  
				 m(1,1) + m(4,1),
				 m(1,2) + m(4,2),
				 m(1,3) + m(4,3),
				 m(1,4) + m(4,4));
	m_Planes[RIGHT].setCoefficients(
				-m(1,1) + m(4,1),
				-m(1,2) + m(4,2),
				-m(1,3) + m(4,3),
				-m(1,4) + m(4,4));
}

#undef m

int
Frustum::isVolumeInside (const IBoundingVolume *aBoundingVolume, bool conservative)
{
	int result = Frustum::INSIDE, out, in;
	int planeCount;

	planeCount = conservative?4:6;

	vec3 bbMin (aBoundingVolume->getMin()), bbMax (aBoundingVolume->getMax());

	for (int i = 0; i < planeCount; i++){
		out = 0;
		in = 0;

		vec3 positiveVec (bbMin);
		vec3 negativeVec (bbMax);

		const vec3& normal = m_Planes[i].getNormal();

		if (normal.x >= 0){
			positiveVec.x = bbMax.x;
			negativeVec.x = bbMin.x;
		}
		if (normal.y >= 0){
			positiveVec.y = bbMax.y;
			negativeVec.y = bbMin.y;
		}
		if (normal.z >= 0){
			positiveVec.z = bbMax.z;
			negativeVec.z = bbMin.z;
		}

		if (m_Planes[i].distance (positiveVec) < 0){
			return Frustum::OUTSIDE;
		} else if (m_Planes[i].distance (negativeVec) < 0){
			result = Frustum::INTERSECT;
		}

	}
	return result;
}




//int
//Frustum::isVolumeInside (IBoundingVolume &aBoundingVolume)
//{
//	int result = Frustum::INSIDE, out, in;
//
//	vec3 bbMin (aBoundingVolume.getMin()), bbMax (aBoundingVolume.getMax());
//
//	for (int i = 0; i < 6; i++){
//		out = 0;
//		in = 0;
//
//		vec3 positiveVec (bbMin);
//		vec3 negativeVec (bbMax);
//
//		const vec3& normal = m_Planes[i].getNormal();
//
//		if (normal.x >= 0){
//			positiveVec.x = bbMax.x;
//			negativeVec.x = bbMin.x;
//		}
//		if (normal.y >= 0){
//			positiveVec.y = bbMax.y;
//			negativeVec.y = bbMin.y;
//		}
//		if (normal.z >= 0){
//			positiveVec.z = bbMax.z;
//			negativeVec.z = bbMin.z;
//		}
//
//		if (m_Planes[i].distance (positiveVec) < 0){
//			return Frustum::OUTSIDE;
//		} else if (m_Planes[i].distance (negativeVec) < 0){
//			result = Frustum::INTERSECT;
//		}
//
//	}
//	return result;
//}

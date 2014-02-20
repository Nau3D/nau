#include <nau/geometry/boundingbox.h>


#define _USE_MATH_DEFINES
#include <cmath>


using namespace nau::math;
using namespace nau::geometry;

float MAXFLOAT = 0xffffff;

#ifdef NAU_RENDER_FLAGS

BBox *BoundingBox::Geometry = NULL;

#endif

BoundingBox::BoundingBox(void):
	m_vPoints(3),
	m_vLocalPoints(3) 
{
	m_vPoints[MIN].set ((float)MAXFLOAT, (float)MAXFLOAT, (float)MAXFLOAT);
	m_vPoints[MAX].set ((float)-MAXFLOAT, (float)-MAXFLOAT, (float)-MAXFLOAT);

	m_vLocalPoints[MIN].set ((float)MAXFLOAT, (float)MAXFLOAT, (float)MAXFLOAT);
	m_vLocalPoints[MAX].set ((float)-MAXFLOAT, (float)-MAXFLOAT, (float)-MAXFLOAT);

	m_GeometryTransform.setScale(-1.0f);

#ifdef NAU_RENDER_FLAGS
	if (Geometry == NULL) {
		Geometry = new BBox();
		Geometry->setDrawingPrimitive(IRenderer::LINE_LOOP);
	}
#endif
}

BoundingBox::BoundingBox (vec3 min, vec3 max):
	m_vPoints(3),
	m_vLocalPoints(3)
{
	set(min,max);
}


BoundingBox::BoundingBox (const BoundingBox &aBoundingBox):
	m_vPoints(3),
	m_vLocalPoints(3)
{
	m_vPoints[MIN] = aBoundingBox.m_vPoints[MIN];
	m_vPoints[MAX] = aBoundingBox.m_vPoints[MAX];
	m_vPoints[CENTER] = aBoundingBox.m_vPoints[CENTER];

	m_vLocalPoints[MIN] = aBoundingBox.m_vLocalPoints[MIN];
	m_vLocalPoints[MAX] = aBoundingBox.m_vLocalPoints[MAX];

	m_GeometryTransform.clone((ITransform *)(&(aBoundingBox.m_GeometryTransform)));
}


BoundingBox::~BoundingBox(void)
{
}


BBox *
BoundingBox::getGeometry()
{
	return Geometry;
}


ITransform *
BoundingBox::getTransform()
{
	return &m_GeometryTransform;
}


void BoundingBox::set(vec3 min, vec3 max) {

	m_vPoints[MIN].set (min.x, min.y, min.z);
	m_vPoints[MAX].set (max.x, max.y, max.z);

	m_vLocalPoints[MIN].set (min.x, min.y, min.z);
	m_vLocalPoints[MAX].set (max.x, max.y, max.z);

	_calculateCenter();

	m_GeometryTransform.setTranslation(m_vPoints[CENTER]);
	m_GeometryTransform.scale(	
						0.5 * (m_vPoints[MAX].x - m_vPoints[MIN].x),
						0.5 * (m_vPoints[MAX].y - m_vPoints[MIN].y),
						0.5 * (m_vPoints[MAX].z - m_vPoints[MIN].z));
}


void 
BoundingBox::calculate (const std::vector<VertexData::Attr> &vertices)
{
	m_vPoints[MIN].set ((float)MAXFLOAT, (float)MAXFLOAT, (float)MAXFLOAT);
	m_vPoints[MAX].set ((float)-MAXFLOAT, (float)-MAXFLOAT, (float)-MAXFLOAT);

	std::vector<vec4>::const_iterator verticesIter;

	verticesIter = vertices.begin();

	for ( ; verticesIter != vertices.end(); ++verticesIter) {

		const vec4 &aVec = (*verticesIter);

		if (m_vPoints[MIN].x > aVec.x) {
			m_vPoints[MIN].x = aVec.x;
		}
		if (m_vPoints[MIN].y > aVec.y) {
			m_vPoints[MIN].y = aVec.y;
		}
		if (m_vPoints[MIN].z > aVec.z) {
			m_vPoints[MIN].z = aVec.z;
		}

		if (m_vPoints[MAX].x < aVec.x) {
			m_vPoints[MAX].x = aVec.x;
		}
		if (m_vPoints[MAX].y < aVec.y) {
			m_vPoints[MAX].y = aVec.y;
		}
		if (m_vPoints[MAX].z < aVec.z) {
			m_vPoints[MAX].z = aVec.z;
		}
	}
	_calculateCenter();

	m_vLocalPoints[MIN] = m_vPoints[MIN];
	m_vLocalPoints[MAX] = m_vPoints[MAX];

	m_GeometryTransform.setTranslation(m_vPoints[CENTER]);
	m_GeometryTransform.scale(	
						0.5 * (m_vPoints[MAX].x - m_vPoints[MIN].x),
						0.5 * (m_vPoints[MAX].y - m_vPoints[MIN].y),
						0.5 * (m_vPoints[MAX].z - m_vPoints[MIN].z));

}


void
BoundingBox::setTransform (ITransform &aTransform)
{
	mat4 m = aTransform.getMat44();
	m_GeometryTransform.setMat44(m);

	std::vector<vec4> vertices (8);

	vertices[0].set (m_vLocalPoints[MIN].x, m_vLocalPoints[MIN].y, m_vLocalPoints[MIN].z);
	vertices[1].set (m_vLocalPoints[MAX].x, m_vLocalPoints[MIN].y, m_vLocalPoints[MIN].z);
	vertices[2].set (m_vLocalPoints[MAX].x, m_vLocalPoints[MIN].y, m_vLocalPoints[MAX].z);
	vertices[3].set (m_vLocalPoints[MIN].x, m_vLocalPoints[MIN].y, m_vLocalPoints[MAX].z);

	vertices[4].set (m_vLocalPoints[MIN].x, m_vLocalPoints[MAX].y, m_vLocalPoints[MIN].z);
	vertices[5].set (m_vLocalPoints[MAX].x, m_vLocalPoints[MAX].y, m_vLocalPoints[MIN].z);
	vertices[6].set (m_vLocalPoints[MAX].x, m_vLocalPoints[MAX].y, m_vLocalPoints[MAX].z);
	vertices[7].set (m_vLocalPoints[MIN].x, m_vLocalPoints[MAX].y, m_vLocalPoints[MAX].z);

	//vertices[0].set (m_vPoints[MIN].x, m_vPoints[MIN].y, m_vPoints[MIN].z);
	//vertices[1].set (m_vPoints[MAX].x, m_vPoints[MIN].y, m_vPoints[MIN].z);
	//vertices[2].set (m_vPoints[MAX].x, m_vPoints[MIN].y, m_vPoints[MAX].z);
	//vertices[3].set (m_vPoints[MIN].x, m_vPoints[MIN].y, m_vPoints[MAX].z);

	//vertices[4].set (m_vPoints[MIN].x, m_vPoints[MAX].y, m_vPoints[MIN].z);
	//vertices[5].set (m_vPoints[MAX].x, m_vPoints[MAX].y, m_vPoints[MIN].z);
	//vertices[6].set (m_vPoints[MAX].x, m_vPoints[MAX].y, m_vPoints[MAX].z);
	//vertices[7].set (m_vPoints[MIN].x, m_vPoints[MAX].y, m_vPoints[MAX].z);

	for (int i = 0; i < 8; i++) {
		aTransform.getMat44().transform (vertices[i]);
	}

	//aTransform.getMat44().transform (m_vPoints[MIN]);
	//aTransform.getMat44().transform (m_vPoints[MAX]);
	//aTransform.getMat44().transform (m_vPoints[CENTER]);

	//std::vector<vec3> vertices(3);

	//for (int i = MIN; i <= CENTER; i++) {
	//	vertices[i] = m_vPoints[i];
	//}

	// Need to preserve local points
	vec3 auxMin, auxMax;
	auxMin = m_vLocalPoints[MIN];
	auxMax = m_vLocalPoints[MAX];

	calculate (vertices);

	m_vLocalPoints[MIN] = auxMin;
	m_vLocalPoints[MAX] = auxMax;
}


bool 
BoundingBox::intersect (const IBoundingVolume *volume)
{
	return true; /***MARK***/
}


void 
BoundingBox::compound (const IBoundingVolume  *volume)
{
	if (true == volume->isA (BOX)) {
		const BoundingBox *aBox = static_cast<const BoundingBox*> (volume);

		// MERGING POINTS
		const vec3 &min = aBox->getMin();
		const vec3 &max = aBox->getMax();

		if (m_vPoints[MIN].x > min.x) {
			m_vPoints[MIN].x = min.x;
		}
		if (m_vPoints[MIN].y > min.y) {
			m_vPoints[MIN].y = min.y;
		}
		if (m_vPoints[MIN].z > min.z) {
			m_vPoints[MIN].z = min.z;
		}

		if (m_vPoints[MAX].x < max.x) {
			m_vPoints[MAX].x = max.x;
		}
		if (m_vPoints[MAX].y < max.y) {
			m_vPoints[MAX].y = max.y;
		}
		if (m_vPoints[MAX].z < max.z) {
			m_vPoints[MAX].z = max.z;
		}
		
		// MERGING LOCAL POINTS
		const vec3 &minLocal = aBox->m_vLocalPoints[MIN];
		const vec3 &maxLocal = aBox->m_vLocalPoints[MAX];

		if (m_vLocalPoints[MIN].x > minLocal.x) {
			m_vLocalPoints[MIN].x = minLocal.x;
		}
		if (m_vLocalPoints[MIN].y > minLocal.y) {
			m_vLocalPoints[MIN].y = minLocal.y;
		}
		if (m_vLocalPoints[MIN].z > minLocal.z) {
			m_vLocalPoints[MIN].z = minLocal.z;
		}

		if (m_vLocalPoints[MAX].x < maxLocal.x) {
			m_vLocalPoints[MAX].x = maxLocal.x;
		}
		if (m_vLocalPoints[MAX].y < maxLocal.y) {
			m_vLocalPoints[MAX].y = maxLocal.y;
		}
		if (m_vLocalPoints[MAX].z < maxLocal.z) {
			m_vLocalPoints[MAX].z = maxLocal.z;
		}

	}
	_calculateCenter();
	
	m_GeometryTransform.setTranslation(m_vPoints[CENTER]);
	m_GeometryTransform.scale(	0.5 * (m_vPoints[MAX].x - m_vPoints[MIN].x),
						0.5 * (m_vPoints[MAX].y - m_vPoints[MIN].y),
						0.5 * (m_vPoints[MAX].z - m_vPoints[MIN].z));

}


void
BoundingBox::_calculateCenter (void)
{
	m_vPoints[CENTER].x = (m_vPoints[MAX].x + m_vPoints[MIN].x) * 0.5f;
	m_vPoints[CENTER].y = (m_vPoints[MAX].y + m_vPoints[MIN].y) * 0.5f;
	m_vPoints[CENTER].z = (m_vPoints[MAX].z + m_vPoints[MIN].z) * 0.5f;
}


bool 
BoundingBox::isA (BoundingVolumeKind kind) const
{
	if (BOX == kind) {
		return true;
	}
	return false;
}


std::string 
BoundingBox::getType (void) const
{
	return "BoundingBox";
}


std::vector<vec3>& 
BoundingBox::getPoints (void) const
{ 
	return m_vPoints;
}


std::vector<vec3>& 
BoundingBox::getNonTransformedPoints (void) const
{ 
	return m_vLocalPoints;
}


const vec3&
BoundingBox::getMin (void) const
{
	return m_vPoints[MIN];
}


const vec3&
BoundingBox::getMax (void) const
{
	return m_vPoints[MAX];
}


const vec3& 
BoundingBox::getCenter (void) const
{
	return m_vPoints[CENTER];
}

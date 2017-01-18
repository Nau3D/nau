#include "nau/scene/scenePoseObject.h"
#include "nau/geometry/boundingBox.h"
#include "nau/math/matrix.h"
#include "nau/geometry/meshWithPose.h"

using namespace nau::scene;
using namespace nau::render;
using namespace nau::geometry;
using namespace nau::math;

ScenePoseObject::ScenePoseObject(void) : SceneObject()
{
	m_StaticCondition = false;
}

ScenePoseObject::~ScenePoseObject(void) {

}

bool
ScenePoseObject::isStatic() {

	return false;
}


void 
ScenePoseObject::setStaticCondition(bool aCondition) {

}


IBoundingVolume*
ScenePoseObject::getBoundingVolume ()
{
	if (0 == m_BoundingVolume) {
		calculateBoundingVolume();
		m_BoundingVolume->setTransform (m_Transform);
	}
	//m_BoundingVolume->transform (*m_Transform);
	return (m_BoundingVolume);
}


// When one wants to manually set the bounding volume
// for instance for a node in an octree
// has no effect for pose based objects
void
ScenePoseObject::setBoundingVolume (IBoundingVolume *b)
{
}

const nau::math::mat4 & 
nau::scene::ScenePoseObject::getTransform() {

	return m_Transform;// TODO: insert return statement here
}



// This is a NOP for Pose based objects
void 
ScenePoseObject::burnTransform (void)
{

}


std::string 
ScenePoseObject::getType (void)
{
	return "PoseObject";
}


void
ScenePoseObject::calculateBoundingVolume (void)
{
	if (0 != m_BoundingVolume) {
		delete m_BoundingVolume;
	}
	m_BoundingVolume = new BoundingBox; /***MARK***/

	std::shared_ptr<MeshPose> &mp = std::dynamic_pointer_cast<MeshPose>(m_Renderable);

	mp->setReferencePose();
	m_BoundingVolume->calculate (m_Renderable->getVertexData()->getDataOf (VertexData::GetAttribIndex(std::string("position"))));
	
	IBoundingVolume *bv = new BoundingBox;

	unsigned int numPoses = mp->getNumberOfPoses();
	for (unsigned int i = 0; i < numPoses; i++) {

		mp->setPose(i);
		bv->calculate( m_Renderable->getVertexData()->getDataOf (VertexData::GetAttribIndex(std::string("position"))));
		m_BoundingVolume->compound(bv);
	}
	mp->setReferencePose();
}


void 
ScenePoseObject::writeSpecificData (std::fstream &f)
{
	return;
}


void 
ScenePoseObject::readSpecificData (std::fstream &f)
{
	return;
}




// Does not apply
void ScenePoseObject::unitize(float min, float max) {
	
}
#include <nau/geometry/poseoffset.h>

using namespace nau::geometry;

PoseOffset::PoseOffset(unsigned int aSize) : 
	m_vOffset(aSize), 
	m_Size(aSize)
{}

PoseOffset::~PoseOffset() 
{

}


void
PoseOffset::addPoseOffset(int index, float x, float y, float z)
{
	if (index >= m_Size)
		return;
	else
		m_vOffset.at(index).set(x, y, z);
}

std::vector<vec3> 
PoseOffset::getOffsets() 
{
	return m_vOffset;
}

#include <nau/geometry/meshposekeyframe.h>

using namespace nau::geometry;


MeshPoseKeyFrame::MeshPoseKeyFrame(): m_PoseIndex(0)
{}


MeshPoseKeyFrame::~MeshPoseKeyFrame()
{}


void
MeshPoseKeyFrame::setPoseIndex(unsigned int anIndex)
{
	m_PoseIndex = anIndex;	
}


unsigned int
MeshPoseKeyFrame::getPoseIndex() 
{
	return m_PoseIndex;
}


void 
MeshPoseKeyFrame::addInfluence(float aTime, float aInfluence)
{
	std::pair<float, float> influence(aTime,aInfluence);

	m_Influences.push_back(influence);
}


// Interpolation function for a pose
float 
MeshPoseKeyFrame::getInfluence(float aTime) 
{
	return 0.0;
}
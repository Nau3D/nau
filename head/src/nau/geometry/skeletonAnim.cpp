#include "nau/geometry/skeletonAnim.h"

using namespace nau::geometry;

SkeletonAnim::SkeletonAnim():
	m_Length(0),
	m_Tracks()
{}

SkeletonAnim::~SkeletonAnim()
{}

std::string
SkeletonAnim::getType()
{
	return "SkeletonAnim";
}

void
SkeletonAnim::setLength(float length)
{
	m_Length = length;
}

float 
SkeletonAnim::getLength()
{
	return m_Length;
}

// creates an empty track
void 
SkeletonAnim::addTrack(unsigned int boneID) 
{
	m_Tracks[boneID] = Track();
}

// insert keyframes, ordered by time
void
SkeletonAnim::addKeyFrame(unsigned int boneID, float aTime, nau::math::vec3 pos, nau::math::vec3 axis, float angle)
{
	if (m_Tracks.count(boneID) == 0) 
		addTrack(boneID);
		
	Keyframe kf;

	kf.m_Angle = angle;
	kf.m_Axis.set(axis.x, axis.y, axis.z);
	kf.m_Position.set(pos.x, pos.y, pos.z);

	std::pair<float, Keyframe> p(aTime,kf);

	Track::iterator iter;
	iter = m_Tracks[boneID].begin();

	while (iter != m_Tracks[boneID].end() && (*iter).first < aTime)
		++iter;

	// aTime is greater than any other time
	if ( iter == m_Tracks[boneID].end()) {


		m_Tracks[boneID].push_back(p);
		//(m_Tracks[meshIndex].back().second)[poseIndex] = anInfluence;
	}
	// aTime does not exist and is before the iterator
	else if ((*iter).first > aTime) {

		m_Tracks[boneID].insert(iter, p);
	}
}


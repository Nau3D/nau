#include "nau/geometry/meshposeanim.h"

using namespace nau::geometry;

MeshPoseAnim::MeshPoseAnim():
		m_Length(0),
		m_Tracks()

{}


MeshPoseAnim::~MeshPoseAnim() 
{
	m_Tracks.clear();
}


std::string 
MeshPoseAnim::getType()
{
	return("MeshPoseAnim");
}


void
MeshPoseAnim::setLength(float aLength)
{
	m_Length = aLength;
}


float
MeshPoseAnim::getLength()
{
	return m_Length;
}


void
MeshPoseAnim::addTrack(unsigned int aTrack)
{
	m_Tracks[aTrack] = KeyFrameVector();
}


//std::map<unsigned int, MeshPoseAnim::KeyFrameVector> *
//MeshPoseAnim::getTracks()
//{
//	return &m_Tracks;
//}


void
MeshPoseAnim::setKeyFrame(unsigned int meshIndex, float time) 
{
	KeyFrame kf(time, PoseInfluenceMap()) ;

	m_Tracks[meshIndex].push_back(kf); 
}


//useless?
//void 
//MeshPoseAnim::setPoseIndex(unsigned int meshIndex, unsigned int poseIndex)
//{
//	m_Tracks[meshIndex][poseIndex] = std::vector<std::pair<float,float>>();
//}


// Adds the keyframe data, so that the KeyFrame vector is in ascending order
void 
MeshPoseAnim::addInfluence(unsigned int meshIndex, unsigned int poseIndex, float aTime, float anInfluence)
{
	KeyFrameVector::iterator iter;

	if (m_Tracks.count(meshIndex) == 0) 
		addTrack(meshIndex);
		
	iter = m_Tracks[meshIndex].begin();

	while (iter != m_Tracks[meshIndex].end() && (*iter).first < aTime)
		++iter;

	// aTime is greater than any other time
	if ( iter == m_Tracks[meshIndex].end()) {

		KeyFrameVector kfv;
		PoseInfluenceMap pim;
		pim[poseIndex] = anInfluence;
		KeyFrame kf( aTime, pim );
		m_Tracks[meshIndex].push_back(kf);
		//(m_Tracks[meshIndex].back().second)[poseIndex] = anInfluence;
	}
	// aTime does not exist and is before the iterator
	else if ((*iter).first > aTime) {

		PoseInfluenceMap pim;
		pim[poseIndex] = anInfluence;
		m_Tracks[meshIndex].insert(iter, KeyFrame(aTime, pim));
	}
	// There is already a keyframe with aTime
	else {
		((*iter).second)[poseIndex] = anInfluence;
	}
}



std::map<unsigned int, float> *
MeshPoseAnim::getInfluences(int meshIndex, float time) 
{
	// THIS IS WHERE THE INFLUENCES FOR MESH meshIndex, FOR A PARTICULAR TIME
	// ARE COLLECTED

	// The map returns for each pose, its influence
	std::map<unsigned int, float> *infMap = new std::map<unsigned int, float>();

	if (m_Tracks.count(meshIndex) == 0)
		return infMap;


	float prevTime, nextTime;
	PoseInfluenceMap piPrevMap, piNextMap;

	MeshPoseAnim::KeyFrameVector kfVec;

	kfVec = m_Tracks[meshIndex];

	KeyFrameVector::iterator kfvNextIter, kfvPrevIter;
	PoseInfluenceMap::iterator pimIter;

	kfvNextIter = kfVec.begin();
	kfvPrevIter = kfvNextIter;

	while (kfvNextIter != kfVec.end() && (*kfvNextIter).first < time) {
		kfvPrevIter = kfvNextIter;
		kfvNextIter++;
	}

	float prevInf, nextInf;
	// time is before the first keyframe
	if (kfvNextIter == kfvPrevIter) {

		nextTime = (*kfvNextIter).first;
		for (pimIter = piNextMap.begin(); pimIter != piNextMap.end(); pimIter++) {

			nextInf =  (*pimIter).second;
			(*infMap)[(*pimIter).first] = (time)/(nextTime) * nextInf;
		}
	}
	// time is after the last keyframe
	// note: this should not happen
	else if (kfvNextIter == kfVec.end()) {

	}
	// time is between keyframes
	else {

		prevTime = (*kfvPrevIter).first;
		nextTime = (*kfvNextIter).first;

		piPrevMap = (*kfvPrevIter).second;
		piNextMap = (*kfvNextIter).second;

		// start with the previous keyframe 
		for (pimIter = piPrevMap.begin(); pimIter != piPrevMap.end(); pimIter++) {

			int poseIndex = (*pimIter).first;
			prevInf = (*pimIter).second;

			// pose is present in both frames
			if (piNextMap.count((*pimIter).first) > 0) {
			
				nextInf = piNextMap[poseIndex];
			}
			// pose is present only in the previous frame
			else {
				nextInf = 0.0f;
			}

			(*infMap)[(*pimIter).first] = prevInf + ((time - prevTime)/(nextTime - prevTime) * (nextInf - prevInf));

		}
		// for each pose in the next keyframe
		// going to add those poses only present in the next key frame
		for (pimIter = piNextMap.begin(); pimIter != piNextMap.end(); pimIter++) {
		
			// pose is only present in the last keyframe
			if (infMap->count((*pimIter).first) == 0) {
			
				nextInf = (*pimIter).second;
				prevInf = 0;

				(*infMap)[(*pimIter).first] = prevInf + ((time - prevTime)/(nextTime - prevTime) * (nextInf - prevInf));
			}
		}
	}
	return (infMap);
}



//	MeshPoseAnim::keyFrame kf;
//	float influence,prevInfluence,nextInfluence;
//	float prevTime, nextTime;
//
//	kf = m_Tracks[meshIndex];
//	keyFrame::iterator iter = kf.begin();
//
//	// for each pose
//	for ( ; iter != kf.end(); ++iter) {
//	
//		// compute proper influence using linear interpolation
//		influence = 0;
//		prevInfluence = 0;
//		nextInfluence = 0;
//		// first locate the time
//		std::vector<std::pair<float,float>>::iterator timeNextIter, timePrevIter;
//		timeNextIter = (*iter).second.begin();
//		timePrevIter = (*iter).second.begin();
//		for ( ; timeNextIter != (*iter).second.end() &&(*timeNextIter).first < time ; timeNextIter++) {
//
//			timePrevIter = timeNextIter;
//		}
//
//		// there is no keyframe with a greater time
//		if (timeNextIter == (*iter).second.end()) {
//		
//			influence = (*timePrevIter).second;
//		}
//
//		// there is no keyframe with a shorther time
//		else if (timeNextIter == timePrevIter) {
//			
//			nextTime = (*timeNextIter).first;
//			nextInfluence = (*timePrevIter).second;
//			influence = nextInfluence * ( time / nextTime);
//		}
//
//		// time is between prev and next
//		else {
//		
//			nextInfluence = (*timeNextIter).second;
//			prevInfluence = (*timePrevIter).second;
//			nextTime = (*timeNextIter).first;
//			prevTime = (*timePrevIter).first;
//			influence = prevInfluence + (nextInfluence - prevInfluence) * (( nextTime - time) / (nextTime - prevTime));
//
//		}
//
//		(*infMap)[(*iter).first] = influence;
//
//	}
//	return infMap;
//}
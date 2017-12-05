#ifndef MESHPOSEANIM_H
#define MESHPOSEANIM_H

#include <string>
#include <vector>
#include <map>

//#include "nau/geometry/meshposekeyframe.h"
#include "nau.h"

using namespace nau::geometry;

namespace nau {

	namespace geometry {

		class MeshPoseAnim
		{
			public:

				// map: poseIndex -> influence
				typedef std::map<int, float> PoseInfluenceMap;

				// (time, PoseInfluenceVector)
				typedef std::pair<float, PoseInfluenceMap> KeyFrame;

				// vector of KeyFrame
				// time must be in ascending order 
				typedef  std::vector <KeyFrame> KeyFrameVector;

				typedef std::map<unsigned int, KeyFrameVector> Tracks;

				MeshPoseAnim(void);
				~MeshPoseAnim(void);
			
				std::string getClassName (void);

				//void setName(std::string aName);
				//std::string getName();

				void setLength(float aLength);
				float getLength();

				void addTrack(unsigned int meshIndex);
				//std::map<unsigned int, KeyFrameVector> *getTracks();

				//void setPoseIndex(unsigned int meshIndex, unsigned int poseIndex);
				void setKeyFrame(unsigned int meshIndex, float time) ;
				void addInfluence(unsigned int meshIndex, unsigned int poseIndex, 
						float aTime, float anInfluence);

				map<unsigned int, float>* getInfluences(int meshIndex, float time);
				
			private:
	
//				std::string m_Name;
				float m_Length;
				// map from mesh index to keyframes
				Tracks m_Tracks;		
				//std::map<unsigned int, std::vector<MeshPoseKeyFrame *>> m_Tracks;		
				
		
		
		};
	};
};



#endif

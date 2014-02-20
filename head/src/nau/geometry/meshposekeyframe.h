// Stores keyframes for a particular pose

#ifndef MESHPOSEKEYFRAME_H
#define MESHPOSEKEYFRAME_H

#include <vector>
#include <utility>

namespace nau {

	namespace geometry {

		class MeshPoseKeyFrame
		{
			public:
				MeshPoseKeyFrame(void);
				~MeshPoseKeyFrame(void);
			
				std::string getType (void);

				//void setName(std::string aName);
				//std::string getName();

				void setPoseIndex(unsigned int anIndex);
				unsigned int getPoseIndex();

				void addInfluence(float time, float influence);

				// this is where the interpolation takes place
				float getInfluence(float aTime);

			private:
	
				//float m_FrameTime;
				//// vector with pose index and influence
				//std::vector<std::pair<unsigned int, float>> m_Influences;	

				// OR

				unsigned int m_PoseIndex;
				// vector with time and influence
				std::vector<std::pair<float, float>> m_Influences;
				
		};
	};
};



#endif

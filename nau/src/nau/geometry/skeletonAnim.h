#ifndef MESHSKELETONANIM_H
#define MESHSKELETONANIM_H

#include "nau.h"

using namespace nau::math;

namespace nau {

	namespace geometry {
	
		class SkeletonAnim {
		
		public:

			struct Keyframe {
				vec3 m_Position;
				vec3 m_Axis;
				float m_Angle;
			};
			
			// time for each keyframe
			// it must be ordered!
			typedef std::vector<std::pair<float,Keyframe>> Track;
		
			// map with the tracks for each bone
			typedef std::map<unsigned int, Track> TrackVector;

			SkeletonAnim();
			~SkeletonAnim();
				
			std::string getClassName(void);

			void setLength(float lenght);
			float getLength();

			void addTrack(unsigned int boneID);
			void addKeyFrame(unsigned int boneID, float time, vec3 pos, vec3 axis, float angle);

		private:

			float m_Length;

			TrackVector m_Tracks;

		};
	
	};



};



#endif
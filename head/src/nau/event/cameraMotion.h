#ifndef CAMERAMOTION_H
#define CAMERAMOTION_H

#include <string>

namespace nau
{
	namespace event_
	{
		class CameraMotion
		{
		public:
			std::string directionType;
			float velocity;
			
			CameraMotion(std::string directionType, float velocity);
			CameraMotion(const CameraMotion &c);
			CameraMotion(void);
			~CameraMotion(void);
			
			void setCameraMotion(std::string directionType, float velocity);
			void setVelocity(float velocity);
			void setDirection(std::string directionType);
			
			float getVelocity(void);
			std::string getDirection(void);			
		};
	};
};

#endif
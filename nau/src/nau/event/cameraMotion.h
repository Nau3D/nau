#ifndef CAMERAMOTION_H
#define CAMERAMOTION_H

#include <string>

#ifdef _WINDLL
#ifdef nau_EXPORTS
#define nau_API __declspec(dllexport)   
#else  
#define nau_API __declspec(dllimport)   
#endif 
#else
#define nau_API
#endif


namespace nau
{
	namespace event_
	{
		class CameraMotion
		{
		public:
			std::string directionType;
			float velocity;
			
			nau_API CameraMotion(std::string directionType, float velocity);
			nau_API CameraMotion(const CameraMotion &c);
			nau_API CameraMotion(void);
			nau_API ~CameraMotion(void);
			
			nau_API void setCameraMotion(std::string directionType, float velocity);
			nau_API void setVelocity(float velocity);
			nau_API void setDirection(std::string directionType);
			
			nau_API float getVelocity(void);
			nau_API std::string getDirection(void);
		};
	};
};

#endif
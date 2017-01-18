#ifndef CAMERAORIENTATION_H
#define CAMERAORIENTATION_H


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
		class CameraOrientation
		{
		public:
			float alpha;
			float beta;
			float newX;
			float newY;
			float oldX;
			float oldY; 
			float scaleFactor;
			
			nau_API CameraOrientation(float alpha, float beta);
			nau_API CameraOrientation(const CameraOrientation &c);
			nau_API CameraOrientation(void);
			nau_API ~CameraOrientation(void);
			
			nau_API void setCameraOrientation(float alpha, float beta, float newX, float newY, float oldX,  float oldY,  float scaleFactor);
			nau_API void setAlpha(float alpha);
			nau_API void setBeta(float beta);
			nau_API void setNewX(float newX);
			nau_API void setNewY(float newY);
			nau_API void setOldX(float oldX);
			nau_API void setOldY(float oldY);
			nau_API void setScaleFactor(float scaleFactor);
			

			nau_API float getAlpha(void);
			nau_API float getBeta(void);
			nau_API float getNewX(void);
			nau_API float getNewY(void);
			nau_API float getOldX(void);
			nau_API float getOldY(void);
			nau_API float getScaleFactor(void);

		};
	};
};

#endif
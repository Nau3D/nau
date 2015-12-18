#ifndef CAMERAORIENTATION_H
#define CAMERAORIENTATION_H



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
			
			CameraOrientation(float alpha, float beta);
			CameraOrientation(const CameraOrientation &c);
			CameraOrientation(void);
			~CameraOrientation(void);
			
			void setCameraOrientation(float alpha, float beta, float newX, float newY, float oldX,  float oldY,  float scaleFactor);
			void setAlpha(float alpha);
			void setBeta(float beta);
			void setNewX(float newX);
			void setNewY(float newY);
			void setOldX(float oldX);
			void setOldY(float oldY);
			void setScaleFactor(float scaleFactor);
			

			float getAlpha(void);
			float getBeta(void);
			float getNewX(void);
			float getNewY(void);
			float getOldX(void);
			float getOldY(void);
			float getScaleFactor(void);

		};
	};
};

#endif
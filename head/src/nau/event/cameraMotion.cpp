#include "nau/event/cameraMotion.h"

using namespace nau::event_;

CameraMotion::CameraMotion(std::string directionType, float velocity)
{
	this->directionType=directionType;
	this->velocity=velocity;	
}

CameraMotion::CameraMotion(const CameraMotion &c)
{
	directionType=c.directionType;
	velocity=c.velocity;
}



CameraMotion::CameraMotion(void)
{
	directionType="FORWARD";
	velocity=0;
}

CameraMotion::~CameraMotion(void)
{
}

void CameraMotion::setCameraMotion(std::string directionType, float velocity)
{
	this->directionType=directionType;
	this->velocity=velocity;	
}



void CameraMotion::setVelocity(float velocity)
{
	this->velocity=velocity;
}

void CameraMotion::setDirection(std::string directionType)
{
	this->directionType=directionType;
}

float CameraMotion::getVelocity(void)
{
	return velocity;
}
		
std::string CameraMotion::getDirection(void)
{
	return directionType;
}
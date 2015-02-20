#include "nau/event/cameraOrientation.h"

using namespace nau::event_;

CameraOrientation::CameraOrientation(float alpha, float beta)
{
	this->alpha=alpha;
	this->beta=beta;
	
}

CameraOrientation::CameraOrientation(const CameraOrientation &c)
{
	alpha=c.alpha;
	beta=c.beta;
	newX=c.newX;
	newY=c.newY;
	oldX=c.oldX;
	oldY=c.oldY;
	scaleFactor=c.scaleFactor;
}



CameraOrientation::CameraOrientation(void)
{
	alpha=0;
	beta=0;
	newX=0;
	newY=0;
	oldX=0;
	oldY=0;
	scaleFactor=0;
}

CameraOrientation::~CameraOrientation(void)
{
}

void CameraOrientation::setCameraOrientation(float alpha, float beta, float newX, float newY, float oldX,  float oldY,  float scaleFactor)
{
	this->alpha=alpha;
	this->beta=beta;
	this->newX=newX;
	this->newY=newY;
	this->oldX=oldX;
	this->oldY=oldY;
	this->scaleFactor=scaleFactor;
}
void CameraOrientation::setAlpha(float alpha)
{
	this->alpha=alpha;
}

void CameraOrientation::setBeta(float beta)
{
	this->beta=beta;
}

void CameraOrientation::setNewX(float newX)
{
	this->newX=newX;
}

void CameraOrientation::setNewY(float newY)
{
	this->newY=newY;
}

void CameraOrientation::setOldX(float oldX)
{
	this->oldX=oldX;
}

void CameraOrientation::setOldY(float oldY)
{
	this->oldY=oldY;
}

void CameraOrientation::setScaleFactor(float scaleFactor)
{
	this->scaleFactor=scaleFactor;
}


float CameraOrientation::getAlpha(void)
{
	return alpha;
}
		
float CameraOrientation::getBeta(void)
{
	return beta;
}

float CameraOrientation::getNewX(void)
{
	return newX;
}

float CameraOrientation::getNewY(void)
{
	return newY;
}

float CameraOrientation::getOldX(void)
{
	return oldX;
}

float CameraOrientation::getOldY(void)
{
	return oldY;
}

float CameraOrientation::getScaleFactor(void)
{
	return scaleFactor;
}

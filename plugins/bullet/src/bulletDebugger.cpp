#include "bulletDebugger.h"

void BulletDebugger::drawLine(const btVector3 & from, const btVector3 & to, const btVector3 & fromColor, const btVector3 & toColor) {
	points->push_back(from.getX());
	points->push_back(from.getY());
	points->push_back(from.getZ());
	points->push_back(1.0f);
	points->push_back(to.getX());
	points->push_back(to.getY());
	points->push_back(to.getZ());
	points->push_back(1.0f);
}

void BulletDebugger::drawLine(const btVector3 & from, const btVector3 & to, const btVector3 & color) {
	points->push_back(from.getX());
	points->push_back(from.getY());
	points->push_back(from.getZ());
	points->push_back(1.0f);
	points->push_back(to.getX());
	points->push_back(to.getY());
	points->push_back(to.getZ());
	points->push_back(1.0f);
}

//void BulletDebugger::drawSphere(const btVector3 & p, btScalar radius, const btVector3 & color)
//{
//}
//
//void BulletDebugger::drawTriangle(const btVector3 & a, const btVector3 & b, const btVector3 & c, const btVector3 & color, btScalar alpha)
//{
//}

void BulletDebugger::drawContactPoint(const btVector3 & PointOnB, const btVector3 & normalOnB, btScalar distance, int lifeTime, const btVector3 & color) {
}

void BulletDebugger::reportErrorWarning(const char * warningString) {
}

void BulletDebugger::draw3dText(const btVector3 & location, const char * textString) {
}

void BulletDebugger::setDebugMode(int debugMode) {
	m_debugMode = debugMode;
}

std::vector<float> * BulletDebugger::getDebugPoints() {
	return points;
}


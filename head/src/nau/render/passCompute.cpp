#include "nau/render/passCompute.h"

#include "nau.h"
#include "nau/debug/profile.h"

using namespace nau::material;
using namespace nau::scene;
using namespace nau::render;
using namespace nau::geometry;


PassCompute::PassCompute(const std::string &passName) : Pass(passName),
m_Mat(0), m_DimX(1), m_DimY(1), m_DimZ(1),
m_BufferX(0), m_BufferY(0), m_BufferZ(0),
m_OffsetX(0), m_OffsetY(0), m_OffsetZ(0) {

	m_ClassName = "compute";
}


void
PassCompute::eventReceived(const std::string &sender, const std::string &eventType, IEventData *evtData) {

}


PassCompute::~PassCompute(){

}


void
PassCompute::prepare (void) {

	m_Mat->prepare();

	if (m_BufferX) {
		m_BufferX->getData(m_OffsetX, 4, &m_DimX);
	}
	if (m_BufferY) {
		m_BufferY->getData(m_OffsetY, 4, &m_DimY);
	}
	if (m_BufferZ) {
		m_BufferZ->getData(m_OffsetZ, 4, &m_DimZ);
	}
}


void
PassCompute::restore (void) {

	m_Mat->restore();
}


void
PassCompute::doPass (void) {

	RENDERER->dispatchCompute(m_DimX, m_DimY, m_DimZ);
}


void 
PassCompute::setMaterialName(const std::string &lName,const std::string &mName) {

	m_Mat = MATERIALLIBMANAGER->getMaterial(lName, mName);
	m_MaterialMap[mName] = MaterialID(lName, mName);
}


Material *
PassCompute::getMaterial() {

	return m_Mat;
}


void
PassCompute::setDimension(int dimX, int dimY, int dimZ) {

	m_DimX = dimX;
	m_DimY = dimY;
	m_DimZ = dimZ;
}


void
PassCompute::setDimFromBuffer(IBuffer  *buffNameX, unsigned int offX,
								IBuffer  *buffNameY, unsigned int offY, 
								IBuffer  *buffNameZ, unsigned int offZ ) {

	m_BufferX = buffNameX;
	m_BufferY = buffNameY;
	m_BufferZ = buffNameZ;

	m_OffsetX = offX;
	m_OffsetY = offY;
	m_OffsetZ = offZ;
}
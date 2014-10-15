#include <nau/render/passCompute.h>

#include <GL/glew.h>
#include <sstream>
#include <algorithm>

#include <nau.h>
#include <nau/debug/profile.h>

using namespace nau::material;
using namespace nau::scene;
using namespace nau::render;
using namespace nau::geometry;


PassCompute::PassCompute (const std::string &passName) : Pass(passName),
m_DimX(1), m_DimY(1), m_DimZ(1), m_Mat(0), m_AtomicX(-1), m_AtomicY(-1), m_AtomicZ(-1)
{
	m_ClassName = "compute";
}


void
PassCompute::eventReceived(const std::string &sender, const std::string &eventType, IEventData *evtData) 
{
}



PassCompute::~PassCompute()
{
}


void
PassCompute::prepare (void)
{
	unsigned int * values;

	m_Mat->prepare();

	if (m_AtomicX >= 0 || m_AtomicY >= 0 || m_AtomicZ >= 0) {

		values = RENDERER->getAtomicCounterValues();

		if (m_AtomicX >= 0)
			m_DimX = values[m_AtomicX];
		if (m_AtomicY >= 0)
			m_DimY = values[m_AtomicY];
		if (m_AtomicZ >= 0)
			m_DimZ = values[m_AtomicZ];
	}
		 
}


void
PassCompute::restore (void)
{
	m_Mat->restore();
}


void
PassCompute::doPass (void)
{	
	glDispatchCompute(m_DimX, m_DimY, m_DimZ);
}


void 
PassCompute::setMaterialName(const std::string &lName,const std::string &mName) 
{
	m_Mat = MATERIALLIBMANAGER->getMaterial(lName, mName);
	m_MaterialMap[mName] = MaterialID(lName, mName);
}


void
PassCompute::setDimension(int dimX, int dimY, int dimZ)
{
	m_DimX = dimX;
	m_DimY = dimY;
	m_DimZ = dimZ;
}


void
PassCompute::setAtomics(int atomicX, int atomicY, int atomicZ)
{
	m_AtomicX = atomicX;
	m_AtomicY = atomicY;
	m_AtomicZ = atomicZ;
}
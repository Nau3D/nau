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
	m_DimX(1), m_DimY(1), m_DimZ(1), m_Mat(0)
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
	m_Mat->prepare();
}


void
PassCompute::restore (void)
{
	m_Mat->restore();
}


bool 
PassCompute::renderTest (void)
{
	return true;
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

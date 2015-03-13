#include "nau/render/optix/optixMaterialLib.h"
#include "nau/render/optix/optixMaterial.h"

#include <vector>

#include "nau.h"

using namespace nau::render::optixRender;


void 
OptixMaterialLib::setContext(optix::Context &aContext) {

	m_Context = aContext;
}


void 
OptixMaterialLib::setTextureLib(OptixTextureLib *otl) {

	m_TextureLib = otl;
}


void
OptixMaterialLib::setMaterialProgram(MaterialPrograms aProgram, int rayType, std::string filename, std::string proc) {

	try {
		switch (aProgram) {
			case CLOSEST_HIT:
				m_ClosestHitProgram[rayType] = m_Context->createProgramFromPTXFile(filename, proc);
				break;
			case ANY_HIT:
				m_AnyHitProgram[rayType] = m_Context->createProgramFromPTXFile(filename, proc);
				break;
			case MISS:
				m_MissProgram[rayType] = m_Context->createProgramFromPTXFile(filename, proc);
		}
	}
	catch(optix::Exception& e) {
		NAU_THROW("Optix Creating Material Program Error (%s - %s) [%s]", filename.c_str(), proc.c_str(), e.getErrorString().c_str());
	}
}


void
OptixMaterialLib::setMaterialProgram(std::string name, MaterialPrograms aProgram, int rayType, std::string filename, std::string proc) {

	OptixMaterial omat;

	if (m_MaterialProcLib.count(name) == 0)
		m_MaterialProcLib[name] = omat;

	m_MaterialProcLib[name].setMaterialProgram(aProgram, rayType, filename, proc);
}


std::string
OptixMaterialLib::isDefined(std::string matName) {

	// the lib contains exactly the material we are looking for
	if (m_MaterialProcLib.count(matName))
		return matName;

	// otherwise look for wildcards (* at the end)
	std::map<std::string, nau::render::optixRender::OptixMaterial>::iterator iter;
	std::string aux;

	iter = m_MaterialProcLib.begin();
	while (iter != m_MaterialProcLib.end()) {

		if ((iter->first)[iter->first.size()-1] == '*') {
			aux = (iter->first).substr(0,iter->first.size()-1);
			
			if (matName.find(aux) != std::string::npos)
				return iter->first;
		}
		
		iter++;
	}
	return "";
}


void 
OptixMaterialLib::addMaterialAttribute(std::string name, nau::material::ProgramValue &p) {

	o_MatAttribute[name] = p;
}


void 
OptixMaterialLib::addMaterial(nau::material::MaterialID aMat) {

	std::string fullMatName = aMat.getLibName() + ":" + aMat.getMaterialName();
	if (m_MaterialLib.count(fullMatName))
		return;

	Material *mat = MATERIALLIBMANAGER->getMaterial(aMat);
	mat->prepareNoShaders();
	optix::Material omat;
	try {
		omat = m_Context->createMaterial();
		unsigned int texCount = m_TextureLib->addTextures(mat);
		
//		if (m_MaterialProcLib.count(fullMatName))
		std::string aux = isDefined(fullMatName);
		if (aux != "")
			m_MaterialProcLib[aux].applyMaterialPrograms(omat);
		else { // apply default programs
			std::map<int, optix::Program>::iterator iter;

			for (iter = m_ClosestHitProgram.begin(); iter != m_ClosestHitProgram.end(); ++iter) {
				omat->setClosestHitProgram(iter->first, iter->second);
			}
			for (iter = m_AnyHitProgram.begin(); iter != m_AnyHitProgram.end(); ++iter) {
				omat->setAnyHitProgram(iter->first, iter->second);
			}
		}

		std::map<std::string, nau::material::ProgramValue>::iterator iter;

		iter = o_MatAttribute.begin();
		for ( ; iter != o_MatAttribute.end(); ++iter) {
			if (iter->second.getValues() != NULL) {
				switch (iter->second.getValueType()) {
							
					case Enums::UINT:
						omat[iter->first]->set1uiv((unsigned int *)iter->second.getValues());
						break;
					case Enums::INT:
					case Enums::BOOL:
					case Enums::ENUM:
						omat[iter->first]->set1iv((int *)iter->second.getValues());
						break;
					case Enums::IVEC2:
					case Enums::BVEC2:
						omat[iter->first]->set2iv((int *)iter->second.getValues());
						break;
					case Enums::IVEC3:
					case Enums::BVEC3:
						omat[iter->first]->set3iv((int *)iter->second.getValues());
						break;
					case Enums::IVEC4:
					case Enums::BVEC4:
						omat[iter->first]->set4iv((int *)iter->second.getValues());
						break;

					case Enums::FLOAT:
						omat[iter->first]->set1fv((float *)iter->second.getValues());
						break;
					case Enums::VEC2:
						omat[iter->first]->set2fv((float *)iter->second.getValues());
						break;
					case Enums::VEC3:
						omat[iter->first]->set3fv((float *)iter->second.getValues());
						break;
					case Enums::VEC4:
						omat[iter->first]->set4fv((float *)iter->second.getValues());
						break;
					case Enums::MAT2:
						omat[iter->first]->setMatrix2x2fv(false,(float *)iter->second.getValues());
						break;
					case Enums::MAT3:
						omat[iter->first]->setMatrix3x3fv(false,(float *)iter->second.getValues());
						break;
					case Enums::MAT4:
						omat[iter->first]->setMatrix4x4fv(false,(float *)iter->second.getValues());
						break;
					default:
						assert(false && "Missing type in OptixMaterialLib.cpp");
						continue;
				}
			}
		}
	}
	catch(optix::Exception& e) {
		NAU_THROW("Optix Creating Material Error (%s:%s) [%s]", 
			aMat.getLibName().c_str(), aMat.getMaterialName().c_str(), e.getErrorString().c_str());
	}

	m_MaterialLib[fullMatName] = omat;

	mat->restoreNoShaders();
}


optix::Material &
OptixMaterialLib::getMaterial(nau::material::MaterialID aMat) {

	std::string fullMatName = aMat.getLibName() + ":" + aMat.getMaterialName();
	if (!m_MaterialLib.count(fullMatName)) 
		addMaterial(aMat);

	return m_MaterialLib[fullMatName];
}


void
OptixMaterialLib::applyMaterial(optix::GeometryInstance gi, nau::material::MaterialID matID) {

	std::string s = matID.getLibName() + ":" + matID.getMaterialName();
	gi->setMaterial(0, m_MaterialLib[s]);

	nau::material::Material *mat = MATERIALLIBMANAGER->getMaterial(matID);
	m_TextureLib->applyTextures(gi, mat);
}


void 
OptixMaterialLib::applyMissPrograms() {

	std::map<int, optix::Program>::iterator iter;

	for (iter = m_MissProgram.begin(); iter != m_MissProgram.end(); ++iter) {
		m_Context->setMissProgram(iter->first, iter->second);
	}
}
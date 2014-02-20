#include <nau/material/materiallib.h>

using namespace nau::material;

MaterialLib::MaterialLib (std::string libName) : 
  m_MaterialLib(),
  m_LibName (libName) 
{
}

MaterialLib::~MaterialLib()
{
   //dtor
	
}


std::string 
MaterialLib::getName()
{
	return(m_LibName);
}


void
MaterialLib::clear()
{
	while (!m_MaterialLib.empty()){
	
		m_MaterialLib.erase(m_MaterialLib.begin());
	}
}

bool 
MaterialLib::hasMaterial (std::string MaterialName)
{
  std::map<std::string, Material*>::const_iterator mat = m_MaterialLib.find (MaterialName);
  
  return (m_MaterialLib.end() != mat);
}

Material*
MaterialLib::getMaterial (std::string MaterialName) 
{
  std::map<std::string, Material*>::const_iterator mat = m_MaterialLib.find (MaterialName);

  if (m_MaterialLib.end() == mat) {
    return &p_Default;
  }

  return (mat->second);
}


std::vector<std::string>* 
MaterialLib::getMaterialNames(std::string aName) 
{
	int len = aName.size();
	if (aName[len-1] == '*')
		len--;

	std::map<std::string, Material*>::iterator iter;
	std::vector<std::string> *names = new std::vector<std::string>;

	for(iter = m_MaterialLib.begin(); iter != m_MaterialLib.end(); ++iter) {
		if (0 == aName.substr(0,len).compare(iter->first.substr(0,len)))
			names->push_back(iter->first);   
  }
	
  return names;
}


std::vector<std::string>* 
MaterialLib::getMaterialNames() 
{

  std::map<std::string, Material*>::iterator iter;
  std::vector<std::string> *names = new std::vector<std::string>;

  for(iter = m_MaterialLib.begin(); iter != m_MaterialLib.end(); ++iter) {
    names->push_back(iter->first);   
  }
	
  return names;
}

void 
MaterialLib::addMaterial (nau::material::Material* aMaterial)
{

  std::string MatName(aMaterial->getName());
  Material *pCurrentMat = getMaterial (MatName);
  
  if (pCurrentMat != 0 && pCurrentMat != aMaterial) {

  }
	
  if (pCurrentMat->getName() != aMaterial->getName()) { 
    m_MaterialLib[MatName] = aMaterial;
  } 
  else if (pCurrentMat != aMaterial) {
    // TODO: Add to log
    //std::cout << "Matlib: adding different valid pointers for same name, possible memleak!" << std::endl;
  }
 }

/*
---------------------------------------------- CUT HERE (OLD CODE) ----------------------------------------------------------
*/


//void CMaterialLib::save(std::string path) {
//
//	std::map<std::string, CMaterial*>::iterator iter;
//	std::string libfilename = path + "/" + m_filename + ".spl";
//	std::ofstream outf(libfilename.c_str());
//
//	outf << "MATERIAL LIB= " << m_filename << "\n";
//	
//    for(iter = lib.begin(); iter != lib.end(); ++iter)
//    {
//		outf << "MATERIAL\n";
//		lib[iter->first]->save(outf,CFilename::GetPath(libfilename));   
//    }
//	outf << "END LIB";
//
//	outf.close();
//
//}
//
//void CMaterialLib::save(std::ofstream &outf, std::string path) {
//
//	std::map<std::string, CMaterial*>::iterator iter;
//	std::string libfilename = path + "/" + m_filename + ".spl";
//
//    for(iter = lib.begin(); iter != lib.end(); ++iter)
//    {
//		outf << "MATERIAL\n";
//		lib[iter->first]->save(outf,CFilename::GetPath(libfilename));   
//    }
//}


//void CMaterialLib::load(std::string &filename) {
//
//	std::ifstream inf(filename.c_str());
//	CMaterial *mat;
//	char line[256];
//	std::string text;
//
//	inf.getline(line,256);
//	text = line;
//
//	m_filename = text.substr(14,100);
//
//	inf.getline(line,256);
//	while (text != "END LIB") {
//
//		text = line;
//		mat = new CMaterial();
//		mat->load(inf,CFilename::GetPath(filename));
//        lib[mat->getName()] = mat;  
//		inf.getline(line,256);
//		text = line;
//    }
//	inf.close();
//}

//void CMaterialLib::add(CMaterial *mat) {
//
//		lib[mat->getName()] = mat;
//}

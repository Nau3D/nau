#include "nau/material/texturemat.h"

#include "nau.h"

using namespace nau::material;
using namespace nau::render;


TextureMat::TextureMat() 
{
  clear();
}


TextureMat::~TextureMat() 
{
	
}


TextureMat *
TextureMat::clone() {

	TextureMat *tm = new TextureMat;

	for (int i = 0 ; i < 8 ; i++) {
	
		tm->m_Textures[i] = m_Textures[i]; 
		tm->m_Samplers[i] = m_Samplers[i];
	}

	return(tm);
}


// unit must be in [0,7]
Texture*
TextureMat::getTexture(int unit)
{

	return m_Textures[unit];
}


// unit must be in [0,7]
TextureSampler*
TextureMat::getTextureSampler(int unit)
{

	return m_Samplers[unit];
}


std::vector<std::string> * 
TextureMat::getTextureNames(){

	std::vector<std::string> *names = new std::vector<std::string>; 

	for( int i = 0 ; i < 8 ; i++ ) {
		if (m_Textures[i] != 0)
			names->push_back(m_Textures[i]->getLabel()); 
    }
	return names;
}


std::vector<int> * 
TextureMat::getTextureUnits(){

	std::vector<int> *units = new std::vector<int>; 

	for( int i = 0 ; i < 8 ; i++ ) {
		if (m_Textures[i] != 0)
			units->push_back(i); 
    }
	return units;
}


void 
TextureMat::prepare(IState *state)
{
	for (int i = 0; i < 8; i++) {
		if (0 != m_Textures[i]){
		//	m_Textures[i]->prepare(i,state);
			m_Textures[i]->prepare(i, m_Samplers[i]);
		//	m_Textures[i]->setUnit (i);
		//	RENDERER->setActiveTextureUnit ((IRenderer::TextureUnit)(IRenderer::TEXTURE_UNIT0 + i));
		//	m_Textures[i]->bind();
			
		}
	}
	//RENDERER->setActiveTextureUnit (IRenderer::TEXTURE_UNIT0);
}


void 
TextureMat::restore(IState *state) 
{
	for (int i = 0; i < 8; i++) {
		if (m_Textures[i] != NULL) {
			m_Textures[i]->restore(i);//,state);
			//RENDERER->setActiveTextureUnit ((IRenderer::TextureUnit)(IRenderer::TEXTURE_UNIT0 + i));
			//m_Textures[i]->unbind();
		}
	}
	RENDERER->setActiveTextureUnit (0);
}


void
TextureMat::clear()
{
  for (int i = 0; i < 8; i++) {
    m_Textures[i] = NULL;
  }
}


void 
TextureMat::setTexture(int unit, Texture *t) 
{
	m_Textures[unit] = t;
	m_Samplers[unit] = TextureSampler::create(t);
//	m_Samplers[unit]->setMipmap(t->getMipmap());
}


void 
TextureMat::unset(int unit) 
{

	m_Textures[unit] = NULL;
	m_Samplers[unit] = 0;
}


//void 
//TextureMat::save(std::ofstream &f, std::string path) {
//
//	int i = 0;
//	while (i < 8 && m_textures[i] == NULL) i++;
//
//	if (i < 8) {
//
//		f << "TEXTURES\n";
//
//		for (i = 0; i < 8; i++) 
//			if (m_Textures[i] != NULL) {
//				f << "TEXTURE UNIT= " << i << "\n";
//
//				if (m_textures[i]->m_filename != NULL) {
//					f << "FILE= "<< CFilename::GetRelativeFileName(path, m_textures[i]->m_filename).c_str() << "\n";
//					m_textures[i]->save(f,path);
//				}
//				else
//					f << "LABEL= " << m_textures[i]->m_label.c_str() << "\n";
//			}
//
//		f << "END TEXTURES\n";
//	}
//}
//
//void 
//TextureMat::load(std::ifstream &inf, std::string libPath) {
//
//	char line[256];
//	int unit,pass,rt,id;
//	std::string text,label;
//
//	inf.getline(line,256);
//	text = line;
//
//
//	while(text != "END TEXTURES") {
//
//		sscanf(line,"TEXTURE UNIT= %d",&unit);
//
//		inf.getline(line,256);
//		text = line;
//		if (text.substr(0,4) == "FILE") {
//
//			text = libPath + "/" + text.substr(6);
//			id = CTextureManager::Instance()->addTexture(text);
//			CTextureDevIL *tex = CTextureManager::Instance()->getTexture(id);
//			m_textures[unit] = tex;		
//			tex->load(inf);
//		}
//		else { // it is a label, i.e. a RT for a PASS
//
//			sscanf(line,"LABEL= Pass #%d: RT %d",&pass,&rt);
//			label = text.substr(7,100);
////			id = CProject::Instance()->m_passes[pass]->m_texId[rt];
//			CTextureDevIL *tex = &CProject::Instance()->m_passes[pass]->m_renderTargets[rt];
//			m_textures[unit] = tex;		
//		}
//		inf.getline(line,256);
//		text = line;
//
//	}
//		
//}

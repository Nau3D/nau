#include "nau/material/texturemat.h"

#include "nau.h"

using namespace nau::material;
using namespace nau::render;


TextureMat::TextureMat() { 

  clear();
}


TextureMat::~TextureMat() {
	
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
TextureMat::getTexture(int unit) {

	return m_Textures[unit];
}


// unit must be in [0,7]
TextureSampler*
TextureMat::getTextureSampler(int unit) {

	return m_Samplers[unit];
}


std::vector<std::string> * 
TextureMat::getTextureNames() {

	std::vector<std::string> *names = new std::vector<std::string>; 

	for( int i = 0 ; i < 8 ; i++ ) {
		if (m_Textures[i] != 0)
			names->push_back(m_Textures[i]->getLabel()); 
    }
	return names;
}


std::vector<int> * 
TextureMat::getTextureUnits() {

	std::vector<int> *units = new std::vector<int>; 

	for( int i = 0 ; i < 8 ; i++ ) {
		if (m_Textures[i] != 0)
			units->push_back(i); 
    }
	return units;
}


void 
TextureMat::prepare(IState *state) {

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
TextureMat::restore(IState *state) {

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
TextureMat::clear() {

  for (int i = 0; i < 8; i++) {
    m_Textures[i] = NULL;
  }
}


void 
TextureMat::setTexture(int unit, Texture *t) {

	m_Textures[unit] = t;
	m_Samplers[unit] = TextureSampler::create(t);
//	m_Samplers[unit]->setMipmap(t->getMipmap());
}


void 
TextureMat::unset(int unit) {

	m_Textures[unit] = NULL;
	m_Samplers[unit] = 0;
}



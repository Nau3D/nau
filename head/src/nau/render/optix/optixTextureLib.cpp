#include "nau/render/optix/optixTextureLib.h"

#include <vector>

#include "nau.h"

#include <glbinding/gl/gl.h>
using namespace gl;
//#include <GL/glew.h>


using namespace nau::render::optixRender;


void 
OptixTextureLib::setContext(optix::Context &aContext) {

	m_Context = aContext;
}


unsigned int
OptixTextureLib::addTextures(std::shared_ptr<nau::material::Material> &m) {

	unsigned int s, count = 0;
	unsigned int mode, mode1;
	ITexture *t;
	GLenum e = glGetError();
	try {
	  for (unsigned int i = 0; i < 8; ++i) {
	
		t = m->getTexture(i);

		if (t != NULL && !m_TextureLib.count(t->getPropi(ITexture::ID))) {
	
			s = t->getPropi(ITexture::ID);
		
			m_TextureLib[s] = m_Context->createTextureSamplerFromGLImage(s, RT_TARGET_GL_TEXTURE_2D);
			mode = m->getTextureSampler(i)->getPrope(ITextureSampler::WRAP_S);
		//	mode = m->getState()->getTexProp((IState::TextureUnit)(IState::TEXTURE0+i),IState::TEXTURE_WRAP_S);
			m_TextureLib[s]->setWrapMode( 0, (RTwrapmode)translateWrapModeToOptix(mode) );
			mode = m->getTextureSampler(i)->getPrope(ITextureSampler::WRAP_T);
		//	mode = m->getState()->getTexProp((IState::TextureUnit)(IState::TEXTURE0+i),IState::TEXTURE_WRAP_T);
			m_TextureLib[s]->setWrapMode( 1, (RTwrapmode)translateWrapModeToOptix(mode) );
			
			mode = m->getTextureSampler(i)->getPrope(ITextureSampler::MAG_FILTER);
			mode1 = m->getTextureSampler(i)->getPrope(ITextureSampler::MIN_FILTER);
		//	mode = m->getState()->getTexProp((IState::TextureUnit)(IState::TEXTURE0+i), IState::TEXTURE_MAG_FILTER);
		//	mode1 = m->getState()->getTexProp((IState::TextureUnit)(IState::TEXTURE0+i), IState::TEXTURE_MIN_FILTER);
			m_TextureLib[s]->setFilteringModes( (RTfiltermode)translateFilterModeToOptix(mode1), 
												(RTfiltermode)translateFilterModeToOptix(mode), 
												RT_FILTER_NONE );
			m_TextureLib[s]->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
			m_TextureLib[s]->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
			m_TextureLib[s]->setMaxAnisotropy(1.0f);

			count++;
		}
	  }
	}
	catch(optix::Exception& e) {
		NAU_THROW("Optix Error: ITexture Creation: texture %s [%s]", t->getLabel().c_str(), e.getErrorString().c_str());
	}
	return(count);
}

//#include <GL\glew.h>
void 
OptixTextureLib::addTexture(unsigned int glID) {

 // GLuint texId = 0;
 // glGenTextures( 1, &texId );
 // glBindTexture( GL_TEXTURE_2D, texId );

 // GLfloat img[2][2][4];

 // //Create a simple checkerboard texture (from OpenGL Programming Guide)
 // for( int j = 0; j < 2; j++ ) {
 //   for( int i = 0; i < 2; i++ ) {
 //     GLfloat c = ( ( ( i & 0x8 ) == 0 ) ^ ( ( j & 0x8 ) == 0 ) ) * 1.0f;
 //     img[ i ][ j ][ 0 ] = 1.0f;
 //     img[ i ][ j ][ 1 ] = c;
 //     img[ i ][ j ][ 2 ] = c;
 //     img[ i ][ j ][ 3 ] = 1.0f;
 //   }
 // }

 // glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
 // glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
 // glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
 // glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
 // glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, 2, 2, 0, GL_RGBA, GL_FLOAT, img );
 // glBindTexture( GL_TEXTURE_2D, 0 );
	//if (m_TextureLib.count(glID))
	//	return;
	m_TextureLib[glID] = m_Context->createTextureSamplerFromGLImage(glID, RT_TARGET_GL_TEXTURE_2D);

	m_TextureLib[glID]->setWrapMode( 0, RT_WRAP_REPEAT );
	m_TextureLib[glID]->setWrapMode( 1, RT_WRAP_REPEAT );
			
	m_TextureLib[glID]->setFilteringModes( RT_FILTER_NEAREST, 
										RT_FILTER_NEAREST, 
										RT_FILTER_NONE );
	//m_TextureLib[glID]->setFilteringModes( RT_FILTER_NEAREST, 
	//									RT_FILTER_NEAREST, 
	//									RT_FILTER_LINEAR );	
	m_TextureLib[glID]->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
	m_TextureLib[glID]->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
	m_TextureLib[glID]->setMaxAnisotropy(1.0f);
}


optix::TextureSampler &
OptixTextureLib::getTexture(unsigned int glID) {

	return m_TextureLib[glID];
}


void
OptixTextureLib::applyTextures(optix::GeometryInstance gi, std::shared_ptr<nau::material::Material> &mat) {

	std::string s;
	ITexture *t;
	try {
	  for (unsigned int i = 0; i < 8; ++i) {
	
		t = mat->getTexture(i);

		if (t != NULL && m_TextureLib.count(t->getPropi(ITexture::ID))) {
			s = "texi";s[3] = i+48;
			gi[s]->setTextureSampler(m_TextureLib[t->getPropi(ITexture::ID)]);
		}
	  }
	}
	catch(optix::Exception& e) {
		NAU_THROW("Optix Error: Setting Textures for Geometry Instances: texture %s [%s]", t->getLabel().c_str(), e.getErrorString().c_str());
	}
}


int 
OptixTextureLib::translateWrapModeToOptix(int mode) {

	switch (mode) {

	case GL_REPEAT:
		return RT_WRAP_REPEAT;
	case GL_CLAMP_TO_EDGE:
		return RT_WRAP_CLAMP_TO_EDGE;
	default:
		return RT_WRAP_REPEAT;
	}
}


int 
OptixTextureLib::translateFilterModeToOptix(int mode) {

	switch (mode) {

		case GL_LINEAR:
			return RT_FILTER_LINEAR;
		case GL_NEAREST:
			return RT_FILTER_NEAREST;
		default:
			return RT_FILTER_LINEAR;
	}
}
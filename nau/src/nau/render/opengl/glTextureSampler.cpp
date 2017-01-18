#include "nau/render/opengl/glTextureSampler.h"

#include "nau.h"

#include <glbinding/gl/gl.h>
using namespace gl;

bool GLTextureSampler::InitedGL = GLTextureSampler::InitGL();

bool
GLTextureSampler::InitGL() {
	// ENUM
	Attribs.listAdd("WRAP_S", "REPEAT", (int)GL_REPEAT);
	Attribs.listAdd("WRAP_S", "CLAMP_TO_EDGE", (int)GL_CLAMP_TO_EDGE);
	Attribs.listAdd("WRAP_S", "CLAMP_TO_BORDER", (int)GL_CLAMP_TO_BORDER);
	Attribs.listAdd("WRAP_S", "MIRRORED_REPEAT", (int)GL_MIRRORED_REPEAT);
	NauInt ni1((int)GL_REPEAT);
	Attribs.setDefault("WRAP_S", ni1);

	Attribs.listAdd("WRAP_T", "REPEAT", (int)GL_REPEAT);
	Attribs.listAdd("WRAP_T", "CLAMP_TO_EDGE", (int)GL_CLAMP_TO_EDGE);
	Attribs.listAdd("WRAP_T", "CLAMP_TO_BORDER", (int)GL_CLAMP_TO_BORDER);
	Attribs.listAdd("WRAP_T", "MIRRORED_REPEAT", (int)GL_MIRRORED_REPEAT);	
	Attribs.setDefault("WRAP_T", ni1);

	Attribs.listAdd("WRAP_R", "REPEAT", (int)GL_REPEAT);
	Attribs.listAdd("WRAP_R", "CLAMP_TO_EDGE", (int)GL_CLAMP_TO_EDGE);
	Attribs.listAdd("WRAP_R", "CLAMP_TO_BORDER", (int)GL_CLAMP_TO_BORDER);
	Attribs.listAdd("WRAP_R", "MIRRORED_REPEAT", (int)GL_MIRRORED_REPEAT);
	Attribs.setDefault("WRAP_R", ni1);

	Attribs.listAdd("MAG_FILTER", "NEAREST", (int)GL_NEAREST);
	Attribs.listAdd("MAG_FILTER", "LINEAR", (int)GL_LINEAR);
	NauInt ni2((int)GL_LINEAR);
	Attribs.setDefault("MAG_FILTER", ni2);

	Attribs.listAdd("MIN_FILTER", "NEAREST", (int)GL_NEAREST);
	Attribs.listAdd("MIN_FILTER", "LINEAR", (int)GL_LINEAR);
	Attribs.listAdd("MIN_FILTER", "LINEAR_MIPMAP_LINEAR", (int)GL_LINEAR_MIPMAP_LINEAR);
	Attribs.listAdd("MIN_FILTER", "LINEAR_MIPMAP_NEAREST", (int)GL_LINEAR_MIPMAP_NEAREST);
	Attribs.listAdd("MIN_FILTER", "NEAREST_MIPMAP_LINEAR", (int)GL_NEAREST_MIPMAP_LINEAR);
	Attribs.listAdd("MIN_FILTER", "NEAREST_MIPMAP_NEAREST", (int)GL_NEAREST_MIPMAP_NEAREST);
	Attribs.setDefault("MIN_FILTER", ni2);

	Attribs.listAdd("COMPARE_FUNC", "LEQUAL", (int)GL_LEQUAL);
	Attribs.listAdd("COMPARE_FUNC", "GEQUAL", (int)GL_GEQUAL);
	Attribs.listAdd("COMPARE_FUNC", "LESS", (int)GL_LESS);
	Attribs.listAdd("COMPARE_FUNC", "GREATER", (int)GL_GREATER);
	Attribs.listAdd("COMPARE_FUNC", "EQUAL", (int)GL_EQUAL);
	Attribs.listAdd("COMPARE_FUNC", "NOTEQUAL", (int)GL_NOTEQUAL);
	Attribs.listAdd("COMPARE_FUNC", "ALWAYS", (int)GL_ALWAYS);
	Attribs.listAdd("COMPARE_FUNC", "NEVER", (int)GL_NEVER);
	NauInt ni3((int)GL_LEQUAL);
	Attribs.setDefault("COMPARE_FUNC", ni3);

	Attribs.listAdd("COMPARE_MODE", "NONE", (int)GL_NONE);
	Attribs.listAdd("COMPARE_MODE", "COMPARE_REF_TO_TEXTURE", (int)GL_COMPARE_REF_TO_TEXTURE);
	NauInt ni4((int)GL_NONE);
	Attribs.setDefault("COMPARE_MODE", ni4);


	return true;
}

using namespace nau::render;



GLTextureSampler::GLTextureSampler(ITexture *t): ITextureSampler() {

	if (APISupport->apiSupport(IAPISupport::TEXTURE_SAMPLERS))
		glGenSamplers(1, (GLuint *)&(m_IntProps[ID]));

	m_BoolProps[MIPMAP] = t->getPropb(ITexture::MIPMAP);	
	if (m_BoolProps[MIPMAP]) {
		m_EnumProps[MIN_FILTER] = (int)GL_LINEAR_MIPMAP_LINEAR;
	}
	else
		m_EnumProps[MIN_FILTER] = (int)GL_LINEAR;

	update();
}


void 
GLTextureSampler::update() {

	if (APISupport->apiSupport(IAPISupport::TEXTURE_SAMPLERS)) {
		glSamplerParameteri(m_IntProps[ID], GL_TEXTURE_MIN_FILTER, m_EnumProps[MIN_FILTER]);
		glSamplerParameteri(m_IntProps[ID], GL_TEXTURE_MAG_FILTER, m_EnumProps[MAG_FILTER]);
		glSamplerParameteri(m_IntProps[ID], GL_TEXTURE_WRAP_S, m_EnumProps[WRAP_S]);
		glSamplerParameteri(m_IntProps[ID], GL_TEXTURE_WRAP_T, m_EnumProps[WRAP_T]);
		glSamplerParameteri(m_IntProps[ID], GL_TEXTURE_WRAP_R, m_EnumProps[WRAP_R]);
		glSamplerParameteri(m_IntProps[ID], GL_TEXTURE_COMPARE_FUNC, m_EnumProps[COMPARE_FUNC]);
		glSamplerParameteri(m_IntProps[ID], GL_TEXTURE_COMPARE_MODE, m_EnumProps[COMPARE_MODE]);
		glSamplerParameterfv(m_IntProps[ID], GL_TEXTURE_BORDER_COLOR, &m_Float4Props[BORDER_COLOR].x);
	}

}


void 
GLTextureSampler::prepare(unsigned int aUnit, int aDim) {

	if (APISupport->apiSupport(IAPISupport::TEXTURE_SAMPLERS)) {
		glBindSampler(aUnit, m_IntProps[ID]);
	}
	else {
		glTexParameteri((GLenum)aDim, GL_TEXTURE_WRAP_S, m_EnumProps[WRAP_S]);
		glTexParameteri((GLenum)aDim, GL_TEXTURE_WRAP_R, m_EnumProps[WRAP_R]);
		glTexParameteri((GLenum)aDim, GL_TEXTURE_WRAP_T, m_EnumProps[WRAP_R]);

		glTexParameteri((GLenum)aDim, GL_TEXTURE_MIN_FILTER, m_EnumProps[MIN_FILTER]);
		glTexParameteri((GLenum)aDim, GL_TEXTURE_MAG_FILTER, m_EnumProps[MAG_FILTER]);

		glTexParameteri((GLenum)aDim, GL_TEXTURE_COMPARE_FUNC, m_EnumProps[COMPARE_FUNC]);
		glTexParameteri((GLenum)aDim, GL_TEXTURE_COMPARE_MODE, m_EnumProps[COMPARE_MODE]);

		vec4 v = m_Float4Props[BORDER_COLOR];
		glTexParameterfv((GLenum)aDim, GL_TEXTURE_BORDER_COLOR, &(v.x));
	}
}


void 
GLTextureSampler::restore(unsigned int aUnit, int aDim) {

	if (APISupport->apiSupport(IAPISupport::TEXTURE_SAMPLERS)) {
		glBindSampler(aUnit, 0);
	}
	else {

		glTexParameteri((GLenum)aDim, GL_TEXTURE_WRAP_S, (int)GL_REPEAT);
		glTexParameteri((GLenum)aDim, GL_TEXTURE_WRAP_R, (int)GL_REPEAT);
		glTexParameteri((GLenum)aDim, GL_TEXTURE_WRAP_T, (int)GL_REPEAT);

		glTexParameteri((GLenum)aDim, GL_TEXTURE_MIN_FILTER, (int)GL_LINEAR);
		glTexParameteri((GLenum)aDim, GL_TEXTURE_MAG_FILTER, (int)GL_LINEAR);

		glTexParameteri((GLenum)aDim, GL_TEXTURE_COMPARE_FUNC, (int)GL_LEQUAL);
		glTexParameteri((GLenum)aDim, GL_TEXTURE_COMPARE_MODE, (int)GL_NONE);

		vec4 v(0.0, 0.0, 0.0, 0.0);
		glTexParameterfv((GLenum)aDim, GL_TEXTURE_BORDER_COLOR, &(v.x));
	}

}

// ---------------------------------------------------------------------
//
//								TEXTURE STATES
//
// ---------------------------------------------------------------------


void 
GLTextureSampler::setPrope(EnumProperty prop, int value) {

	GLenum v2 = (GLenum)value;

	if (prop == MIN_FILTER && m_BoolProps[MIPMAP] == true) { 
		if ((GLenum)value == GL_NEAREST)
			v2 = GL_NEAREST_MIPMAP_NEAREST;
		else if ((GLenum)value == GL_LINEAR)
			v2 = GL_LINEAR_MIPMAP_LINEAR;
	}
	else if (prop == MIN_FILTER && m_BoolProps[MIPMAP] == false) {
		if ((GLenum)value == GL_NEAREST_MIPMAP_NEAREST)
			v2 = GL_NEAREST;
		else if ((GLenum)value == GL_NEAREST_MIPMAP_LINEAR || 
				(GLenum)value == GL_LINEAR_MIPMAP_NEAREST || 
				(GLenum)value == GL_LINEAR_MIPMAP_LINEAR)
			v2 = GL_LINEAR;

	}

	m_EnumProps[prop] = (int)v2;

	if (APISupport->apiSupport(IAPISupport::TEXTURE_SAMPLERS)) {
		switch (prop) {
		case WRAP_S:
			glSamplerParameteri(m_IntProps[ID], GL_TEXTURE_WRAP_S, m_EnumProps[prop]);
			break;
		case WRAP_T:
			glSamplerParameteri(m_IntProps[ID], GL_TEXTURE_WRAP_T, m_EnumProps[prop]);
			break;
		case WRAP_R:
			glSamplerParameteri(m_IntProps[ID], GL_TEXTURE_WRAP_R, m_EnumProps[prop]);
			break;
		case MIN_FILTER:
			glSamplerParameteri(m_IntProps[ID], GL_TEXTURE_MIN_FILTER, m_EnumProps[prop]);
			break;
		case MAG_FILTER:
			glSamplerParameteri(m_IntProps[ID], GL_TEXTURE_MAG_FILTER, m_EnumProps[prop]);
			break;
		case COMPARE_MODE:
			glSamplerParameteri(m_IntProps[ID], GL_TEXTURE_COMPARE_MODE, m_EnumProps[prop]);
			break;
		case COMPARE_FUNC:
			glSamplerParameteri(m_IntProps[ID], GL_TEXTURE_COMPARE_FUNC, m_EnumProps[prop]);
			break;
		}
	}
}


void 
GLTextureSampler::setPropf4(Float4Property prop, vec4 &value) {

	m_Float4Props[prop] = value;
	update();
}

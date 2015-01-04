#include <nau/render/opengl/gltexturesampler.h>

//#include <GL/glew.h>

bool GLTextureSampler::Inited = GLTextureSampler::InitGL();

bool
GLTextureSampler::InitGL() {
	// ENUM
	Attribs.listAdd("WRAP_S", "REPEAT", GL_REPEAT);
	Attribs.listAdd("WRAP_S", "CLAMP_TO_EDGE", GL_CLAMP_TO_EDGE);
	Attribs.listAdd("WRAP_S", "CLAMP_TO_BOREDER", GL_CLAMP_TO_BORDER);
	Attribs.listAdd("WRAP_S", "MIRRORED_REPEAT", GL_MIRRORED_REPEAT);
	Attribs.setDefault("WRAP_S", new int(GL_REPEAT));

	Attribs.listAdd("WRAP_T", "REPEAT", GL_REPEAT);
	Attribs.listAdd("WRAP_T", "CLAMP_TO_EDGE", GL_CLAMP_TO_EDGE);
	Attribs.listAdd("WRAP_T", "CLAMP_TO_BOREDER", GL_CLAMP_TO_BORDER);
	Attribs.listAdd("WRAP_T", "MIRRORED_REPEAT", GL_MIRRORED_REPEAT);
	Attribs.setDefault("WRAP_T", new int(GL_REPEAT));

	Attribs.listAdd("WRAP_R", "REPEAT", GL_REPEAT);
	Attribs.listAdd("WRAP_R", "CLAMP_TO_EDGE", GL_CLAMP_TO_EDGE);
	Attribs.listAdd("WRAP_R", "CLAMP_TO_BOREDER", GL_CLAMP_TO_BORDER);
	Attribs.listAdd("WRAP_R", "MIRRORED_REPEAT", GL_MIRRORED_REPEAT);
	Attribs.setDefault("WRAP_R", new int(GL_REPEAT));

	Attribs.listAdd("MAG_FILTER", "NEAREST", GL_NEAREST);
	Attribs.listAdd("MAG_FILTER", "LINEAR", GL_LINEAR);
	Attribs.setDefault("MAG_FILTER", new int(GL_LINEAR));

	Attribs.listAdd("MIN_FILTER", "NEAREST", GL_NEAREST);
	Attribs.listAdd("MIN_FILTER", "LINEAR", GL_LINEAR);
	Attribs.listAdd("MIN_FILTER", "LINEAR_MIPMAP_LINEAR", GL_LINEAR_MIPMAP_LINEAR);
	Attribs.listAdd("MIN_FILTER", "LINEAR_MIPMAP_NEAREST", GL_LINEAR_MIPMAP_NEAREST);
	Attribs.listAdd("MIN_FILTER", "NEAREST_MIPMAP_LINEAR", GL_NEAREST_MIPMAP_LINEAR);
	Attribs.listAdd("MIN_FILTER", "NEAREST_MIPMAP_NEAREST", GL_NEAREST_MIPMAP_NEAREST);
	Attribs.setDefault("MIN_FILTER", new int(GL_LINEAR));

	Attribs.listAdd("COMPARE_FUNC", "LEQUAL", GL_LEQUAL);
	Attribs.listAdd("COMPARE_FUNC", "GEQUAL", GL_GEQUAL);
	Attribs.listAdd("COMPARE_FUNC", "LESS", GL_LESS);
	Attribs.listAdd("COMPARE_FUNC", "GREATER", GL_GREATER);
	Attribs.listAdd("COMPARE_FUNC", "EQUAL", GL_EQUAL);
	Attribs.listAdd("COMPARE_FUNC", "NOTEQUAL", GL_NOTEQUAL);
	Attribs.listAdd("COMPARE_FUNC", "ALWAYS", GL_ALWAYS);
	Attribs.listAdd("COMPARE_FUNC", "NEVER", GL_NEVER);
	Attribs.setDefault("COMPARE_FUNC", new int(GL_LEQUAL));

	Attribs.listAdd("COMPARE_MODE", "NONE", GL_NONE);
	Attribs.listAdd("COMPARE_MODE", "COMPARE_REF_TO_TEXTURE", GL_COMPARE_REF_TO_TEXTURE);
	Attribs.setDefault("COMPARE_MODE", new int(GL_NONE));


	return true;
}

using namespace nau::render;



GLTextureSampler::GLTextureSampler(Texture *t): TextureSampler() {

	Attribs.initAttribInstanceBoolArray(m_BoolProps);
	Attribs.initAttribInstanceEnumArray(m_EnumProps);
	Attribs.initAttribInstanceIntArray(m_IntProps);
	Attribs.initAttribInstanceVec4Array(m_Float4Props);

#if NAU_OPENGL_VERSION > 320
	glGenSamplers(1, (GLuint *)&(m_IntProps[ID]));
#endif
	m_BoolProps[MIPMAP] = t->getPropb(Texture::MIPMAP);	
	if (m_BoolProps[MIPMAP]) {
		m_EnumProps[MIN_FILTER] = GL_LINEAR_MIPMAP_LINEAR;
	}
	else
		m_EnumProps[MIN_FILTER] = GL_LINEAR;

	//m_EnumProps[MAG_FILTER] = GL_LINEAR;
	//m_EnumProps[WRAP_S] = GL_REPEAT;
	//m_EnumProps[WRAP_T] = GL_REPEAT;
	//m_EnumProps[WRAP_R] = GL_REPEAT;

	//m_EnumProps[COMPARE_MODE] = GL_NONE;
	//m_EnumProps[COMPARE_FUNC] = GL_LEQUAL;

	//float bordercolor[] = {-1.0f, -1.0f, -1.0f, -1.0f};
	//m_Float4Props[BORDER_COLOR].set(bordercolor);

	update();
}


void 
GLTextureSampler::update() {

#if NAU_OPENGL_VERSION > 320
	glSamplerParameteri(m_IntProps[ID], GL_TEXTURE_MIN_FILTER, m_EnumProps[MIN_FILTER]);
	glSamplerParameteri(m_IntProps[ID], GL_TEXTURE_MAG_FILTER, m_EnumProps[MAG_FILTER]);
	glSamplerParameteri(m_IntProps[ID], GL_TEXTURE_WRAP_S, 	m_EnumProps[WRAP_S]);
	glSamplerParameteri(m_IntProps[ID], GL_TEXTURE_WRAP_T, 	m_EnumProps[WRAP_T]);
	glSamplerParameteri(m_IntProps[ID], GL_TEXTURE_WRAP_R,		m_EnumProps[WRAP_R]);
	glSamplerParameteri(m_IntProps[ID], GL_TEXTURE_COMPARE_FUNC, m_EnumProps[COMPARE_FUNC]);
	glSamplerParameteri(m_IntProps[ID], GL_TEXTURE_COMPARE_MODE, m_EnumProps[COMPARE_MODE]);
#endif
}


void 
GLTextureSampler::prepare(unsigned int aUnit, int aDim) {

#if (NAU_OPENGL_VERSION > 320)
	glBindSampler(aUnit, m_IntProps[ID]);
#else
	glTexParameteri(aDim, GL_TEXTURE_WRAP_S, m_EnumProps[WRAP_S]);
	glTexParameteri(aDim, GL_TEXTURE_WRAP_R, m_EnumProps[WRAP_R]);
	glTexParameteri(aDim, GL_TEXTURE_WRAP_T, m_EnumProps[WRAP_R]);

	glTexParameteri(aDim, GL_TEXTURE_MIN_FILTER, m_EnumProps[MIN_FILTER]);
	glTexParameteri(aDim, GL_TEXTURE_MAG_FILTER, m_EnumProps[MAG_FILTER]);

	glTexParameteri(aDim, GL_TEXTURE_COMPARE_FUNC, m_EnumProps[COMPARE_FUNC]);
	glTexParameteri(aDim, GL_TEXTURE_COMPARE_MODE, m_EnumProps[COMPARE_MODE]);

	vec4 v = m_Float4Props[BORDER_COLOR];
	glTexParameterfv(aDim, GL_TEXTURE_BORDER_COLOR,&(v.x));
#endif
}


void 
GLTextureSampler::restore(unsigned int aUnit, int aDim) {

#if (NAU_OPENGL_VERSION > 320)
	glBindSampler(aUnit, 0);
#else

	glTexParameteri(aDim, GL_TEXTURE_WRAP_S ,GL_REPEAT);
	glTexParameteri(aDim, GL_TEXTURE_WRAP_R ,GL_REPEAT);
	glTexParameteri(aDim, GL_TEXTURE_WRAP_T ,GL_REPEAT);

	glTexParameteri(aDim, GL_TEXTURE_MIN_FILTER ,GL_LINEAR);
	glTexParameteri(aDim, GL_TEXTURE_MAG_FILTER ,GL_LINEAR);

	glTexParameteri(aDim, GL_TEXTURE_COMPARE_FUNC ,GL_LEQUAL);
	glTexParameteri(aDim, GL_TEXTURE_COMPARE_MODE ,GL_NONE);

	vec4 v(0.0, 0.0, 0.0, 0.0);
	glTexParameterfv(aDim, GL_TEXTURE_BORDER_COLOR,&(v.x));
#endif

}

// ---------------------------------------------------------------------
//
//								TEXTURE STATES
//
// ---------------------------------------------------------------------


void 
GLTextureSampler::setPrope(EnumProperty prop, int value) {

	int v2 = value;

	if (prop == MIN_FILTER && m_BoolProps[MIPMAP] == true) { 
		if (value == GL_NEAREST)
			v2 = GL_NEAREST_MIPMAP_NEAREST;
		else if (value == GL_LINEAR)
			v2 = GL_LINEAR_MIPMAP_LINEAR;
	}
	else if (prop == MIN_FILTER && m_BoolProps[MIPMAP] == false) {
		if (value == GL_NEAREST_MIPMAP_NEAREST)
			v2 = GL_NEAREST;
		else if (value == GL_NEAREST_MIPMAP_LINEAR || value == GL_LINEAR_MIPMAP_NEAREST || value == GL_LINEAR_MIPMAP_LINEAR)
			v2 = GL_LINEAR;

	}
	else 
		v2 = value;

	m_EnumProps[prop] = v2;

#if NAU_OPENGL_VERSION > 320
	switch(prop) {
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
#endif
}



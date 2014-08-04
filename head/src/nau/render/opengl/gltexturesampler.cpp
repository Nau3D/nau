#include <nau/render/opengl/gltexturesampler.h>

//#include <GL/glew.h>

bool GLTextureSampler::Inited = Init();

bool
GLTextureSampler::Init() {
	// ENUM
	Attribs.add(Attribute(WRAP_S, "WRAP_S", Enums::DataType::ENUM, false, new int(GL_REPEAT)));
	Attribs.listAdd("WRAP_S", "REPEAT", GL_REPEAT);
	Attribs.listAdd("WRAP_S", "CLAMP_TO_EDGE", GL_CLAMP_TO_EDGE);
	Attribs.listAdd("WRAP_S", "CLAMP_TO_BOREDER", GL_CLAMP_TO_BORDER);
	Attribs.listAdd("WRAP_S", "MIRRORED_REPEAT", GL_MIRRORED_REPEAT);
	Attribs.setDefault("WRAP_S", new int(GL_REPEAT));

	Attribs.add(Attribute(WRAP_T, "WRAP_T", Enums::DataType::ENUM, false, new int(GL_REPEAT)));
	Attribs.listAdd("WRAP_T", "REPEAT", GL_REPEAT);
	Attribs.listAdd("WRAP_T", "CLAMP_TO_EDGE", GL_CLAMP_TO_EDGE);
	Attribs.listAdd("WRAP_T", "CLAMP_TO_BOREDER", GL_CLAMP_TO_BORDER);
	Attribs.listAdd("WRAP_T", "MIRRORED_REPEAT", GL_MIRRORED_REPEAT);
	Attribs.setDefault("WRAP_T", new int(GL_REPEAT));

	Attribs.add(Attribute(WRAP_R, "WRAP_R", Enums::DataType::ENUM, false, new int(GL_REPEAT)));
	Attribs.listAdd("WRAP_R", "REPEAT", GL_REPEAT);
	Attribs.listAdd("WRAP_R", "CLAMP_TO_EDGE", GL_CLAMP_TO_EDGE);
	Attribs.listAdd("WRAP_R", "CLAMP_TO_BOREDER", GL_CLAMP_TO_BORDER);
	Attribs.listAdd("WRAP_R", "MIRRORED_REPEAT", GL_MIRRORED_REPEAT);
	Attribs.setDefault("WRAP_R", new int(GL_REPEAT));

	Attribs.add(Attribute(MAG_FILTER, "MAG_FILTER", Enums::DataType::ENUM, false, new int(GL_LINEAR)));
	Attribs.listAdd("MAG_FILTER", "NEAREST", GL_NEAREST);
	Attribs.listAdd("MAG_FILTER", "LINEAR", GL_LINEAR);
	Attribs.setDefault("MAG_FILTER", new int(GL_LINEAR));

	Attribs.add(Attribute(MIN_FILTER, "MIN_FILTER", Enums::DataType::ENUM, false, new int(GL_LINEAR)));
	Attribs.listAdd("MIN_FILTER", "NEAREST", GL_NEAREST);
	Attribs.listAdd("MIN_FILTER", "LINEAR", GL_LINEAR);
	Attribs.listAdd("MIN_FILTER", "LINEAR_MIPMAP_LINEAR", GL_LINEAR_MIPMAP_LINEAR);
	Attribs.listAdd("MIN_FILTER", "LINEAR_MIPMAP_NEAREST", GL_LINEAR_MIPMAP_NEAREST);
	Attribs.listAdd("MIN_FILTER", "NEAREST_MIPMAP_LINEAR", GL_NEAREST_MIPMAP_LINEAR);
	Attribs.listAdd("MIN_FILTER", "NEAREST_MIPMAP_NEAREST", GL_NEAREST_MIPMAP_NEAREST);
	Attribs.setDefault("MIN_FILTER", new int(GL_LINEAR));

	Attribs.add(Attribute(COMPARE_FUNC, "COMPARE_FUNC", Enums::DataType::ENUM, false, new int(GL_LEQUAL)));
	Attribs.listAdd("COMPARE_FUNC", "LEQUAL", GL_LEQUAL);
	Attribs.listAdd("COMPARE_FUNC", "GEQUAL", GL_GEQUAL);
	Attribs.listAdd("COMPARE_FUNC", "LESS", GL_LESS);
	Attribs.listAdd("COMPARE_FUNC", "GREATER", GL_GREATER);
	Attribs.listAdd("COMPARE_FUNC", "EQUAL", GL_EQUAL);
	Attribs.listAdd("COMPARE_FUNC", "NOTEQUAL", GL_NOTEQUAL);
	Attribs.listAdd("COMPARE_FUNC", "ALWAYS", GL_ALWAYS);
	Attribs.listAdd("COMPARE_FUNC", "NEVER", GL_NEVER);
	Attribs.setDefault("COMPARE_FUNC", new int(GL_LEQUAL));

	Attribs.add(Attribute(COMPARE_MODE, "COMPARE_MODE", Enums::DataType::ENUM, false, new int(GL_NONE)));
	Attribs.listAdd("COMPARE_MODE", "NONE", GL_NONE);
	Attribs.listAdd("COMPARE_MODE", "COMPARE_REF_TO_TEXTURE", GL_COMPARE_REF_TO_TEXTURE);
	Attribs.setDefault("COMPARE_MODE", new int(GL_NONE));

	//VEC4
	Attribs.add(Attribute(BORDER_COLOR, "BORDER_COLOR", Enums::DataType::ENUM, false, new vec4(0.0f, 0.0f, 0.0f, 0.0f)));
	//INT
	Attribs.add(Attribute(ID, "ID", Enums::DataType::INT, true, new int(0)));
	//BOOL
	Attribs.add(Attribute(MIPMAP, "MIPMAP", Enums::DataType::BOOL, true, new bool(false)));

	return true;
}

using namespace nau::render;



GLTextureSampler::GLTextureSampler(Texture *t): TextureSampler() {

	Attribs.initAttribInstanceBoolArray(m_BoolProps);
	Attribs.initAttribInstanceEnumArray(m_EnumProps);
	Attribs.initAttribInstanceIntArray(m_IntProps);
	Attribs.initAttribInstanceVec4Array(m_Float4Props);

#if NAU_OPENGL_VERSION > 320
	glGenSamplers(1, &(m_UIntProps[ID]));
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
	glSamplerParameteri(m_UIntProps[ID], GL_TEXTURE_MIN_FILTER, m_EnumProps[MIN_FILTER]);
	glSamplerParameteri(m_UIntProps[ID], GL_TEXTURE_MAG_FILTER, m_EnumProps[MAG_FILTER]);
	glSamplerParameteri(m_UIntProps[ID], GL_TEXTURE_WRAP_S, 	m_EnumProps[WRAP_S]);
	glSamplerParameteri(m_UIntProps[ID], GL_TEXTURE_WRAP_T, 	m_EnumProps[WRAP_T]);
	glSamplerParameteri(m_UIntProps[ID], GL_TEXTURE_WRAP_R,		m_EnumProps[WRAP_R]);
	glSamplerParameteri(m_UIntProps[ID], GL_TEXTURE_COMPARE_FUNC, m_EnumProps[COMPARE_FUNC]);
	glSamplerParameteri(m_UIntProps[ID], GL_TEXTURE_COMPARE_MODE, m_EnumProps[COMPARE_MODE]);
#endif
}


void 
GLTextureSampler::prepare(unsigned int aUnit, int aDim) {

#if (NAU_OPENGL_VERSION > 320)
	glBindSampler(aUnit, m_UIntProps[ID]);
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
GLTextureSampler::setProp(EnumProperty prop, int value) {

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
			glSamplerParameteri(m_UIntProps[ID], GL_TEXTURE_WRAP_S, m_EnumProps[prop]);
			break;
		case WRAP_T:
			glSamplerParameteri(m_UIntProps[ID], GL_TEXTURE_WRAP_T, m_EnumProps[prop]);
			break;
		case WRAP_R:
			glSamplerParameteri(m_UIntProps[ID], GL_TEXTURE_WRAP_R, m_EnumProps[prop]);
			break;
		case MIN_FILTER:
			glSamplerParameteri(m_UIntProps[ID], GL_TEXTURE_MIN_FILTER, m_EnumProps[prop]);
			break;
		case MAG_FILTER:
			glSamplerParameteri(m_UIntProps[ID], GL_TEXTURE_MAG_FILTER, m_EnumProps[prop]);
			break;
		case COMPARE_MODE:
			glSamplerParameteri(m_UIntProps[ID], GL_TEXTURE_COMPARE_MODE, m_EnumProps[prop]);
			break;
		case COMPARE_FUNC:
			glSamplerParameteri(m_UIntProps[ID], GL_TEXTURE_COMPARE_FUNC, m_EnumProps[prop]);
			break;
	}
#endif
}


void 
GLTextureSampler::setProp(Float4Property prop, float x, float y, float z, float w) {

	m_Float4Props[prop].set(x,y,z,w);
}
			

void 
GLTextureSampler::setProp(Float4Property prop, vec4& value) {

	m_Float4Props[prop].set(value.x,value.y,value.z,value.w);
}
			



// ---------------------------------------------------------------------
//
//								TRANSLATES
//
// ---------------------------------------------------------------------


//unsigned int 
//GLTextureSampler::TranslateTexEnum(GLTextureSampler::TextureEnumProp p) {
//
//	switch(p) {
//		case TEXTURE_WRAP_S:
//			return GL_TEXTURE_WRAP_S;
//		case TEXTURE_WRAP_T:
//			return GL_TEXTURE_WRAP_T;
//		case TEXTURE_WRAP_R:
//			return GL_TEXTURE_WRAP_R;
//		case TEXTURE_MIN_FILTER:
//			return GL_TEXTURE_MIN_FILTER;
//		case TEXTURE_MAG_FILTER:
//			return GL_TEXTURE_MAG_FILTER;
//		case TEXTURE_COMPARE_FUNC:
//			return GL_TEXTURE_COMPARE_FUNC;
//		case TEXTURE_COMPARE_MODE:
//			return GL_TEXTURE_COMPARE_MODE;
//		default:
//			return GL_INVALID_ENUM;
//	}
//}
	

//unsigned int  
//GLTextureSampler::TranslateTexWrapMode (TextureSampler::TextureWrapMode aMode)
//{
//	switch (aMode) {
//		case TEXTURE_WRAP_REPEAT:
//			return GL_REPEAT;
//		case TEXTURE_WRAP_CLAMP_TO_EDGE:
//			return GL_CLAMP_TO_EDGE;
//		case TEXTURE_WRAP_CLAMP_TO_BORDER:
//			return GL_CLAMP_TO_BORDER;
//		case TEXTURE_WRAP_MIRRORED_REPEAT:
//			return GL_MIRRORED_REPEAT;
//	        default:
//		  return GL_INVALID_ENUM;
//	}
//}


//unsigned int  
//GLTextureSampler::TranslateTexCompareMode (TextureSampler::TextureCompareMode aMode)
//{
//	switch (aMode) {
//		case TEXTURE_COMPARE_NONE:
//			return GL_NONE;
//		case TEXTURE_COMPARE_REF_TO_TEXTURE:
//			return GL_COMPARE_REF_TO_TEXTURE;
//	        default:
//		  return GL_INVALID_ENUM;
//	}
//}


//unsigned int  
//GLTextureSampler::TranslateTexMagMode (TextureSampler::TextureMagMode aMode)
//{
//	switch (aMode) {
//		case TEXTURE_MAG_NEAREST:
//			return GL_NEAREST;
//		case TEXTURE_MAG_LINEAR:
//			return GL_LINEAR;
//	        default:
//		  return GL_INVALID_ENUM;
//	}
//}


//unsigned int  
//GLTextureSampler::TranslateTexMinMode (TextureSampler::TextureMinMode aMode)
//{
//	switch (aMode) {
//		case TEXTURE_MIN_NEAREST:
//			return GL_NEAREST;
//		case TEXTURE_MIN_LINEAR:
//			return GL_LINEAR;
//		case TEXTURE_MIN_NEAREST_MIPMAP_NEAREST:
//			return GL_NEAREST_MIPMAP_NEAREST;
//		case TEXTURE_MIN_NEAREST_MIPMAP_LINEAR:
//			return GL_NEAREST_MIPMAP_LINEAR;
//		case TEXTURE_MIN_LINEAR_MIPMAP_NEAREST:
//			return GL_LINEAR_MIPMAP_NEAREST;
//		case TEXTURE_MIN_LINEAR_MIPMAP_LINEAR:
//			return GL_LINEAR_MIPMAP_LINEAR;
//	    default:
//		  return GL_INVALID_ENUM;
//	}
//}
//
//
//unsigned int  
//GLTextureSampler::TranslateTexCompareFunc (TextureSampler::TextureCompareFunc aFunc)
//{
//	switch (aFunc) {
//		case COMPARE_NONE:
//			return GL_NONE;
//		case COMPARE_LEQUAL:
//			return GL_LEQUAL;
//		case COMPARE_GEQUAL:
//			return GL_GEQUAL;
//		case COMPARE_LESS:
//			return GL_LESS;
//		case COMPARE_GREATER:
//			return GL_GREATER;
//		case COMPARE_EQUAL:
//			return GL_EQUAL;
//		case COMPARE_NOTEQUAL:
//			return GL_NOTEQUAL;
//		case COMPARE_ALWAYS:
//			return GL_ALWAYS;
//		case COMPARE_NEVER:
//			return GL_NEVER;
//	    default:
//		  return GL_INVALID_ENUM;
//	}
//}


//void 
//GLTextureSampler::setMipmap(bool m) {
//
//	m_Mipmap = m;
//
//	if (m_Mipmap && (m_TexEnumProps[TEXTURE_MIN_FILTER] == TEXTURE_MIN_NEAREST  || 
//						m_TexEnumProps[TEXTURE_MIN_FILTER] == TEXTURE_MIN_LINEAR)) {
//			m_TexEnumProps[TEXTURE_MIN_FILTER] = TEXTURE_MIN_LINEAR_MIPMAP_LINEAR;
//#if NAU_OPENGL_VERSION > 320
//			glSamplerParameteri(m_SamplerID, GL_TEXTURE_MIN_FILTER, 
//					translateTexMinMode((TextureMinMode)m_TexEnumProps[TEXTURE_MIN_FILTER]));
//#endif
//	}
//
//	else if (!m_Mipmap && (m_TexEnumProps[TEXTURE_MIN_FILTER] != TEXTURE_MIN_NEAREST  && 
//						m_TexEnumProps[TEXTURE_MIN_FILTER] != TEXTURE_MIN_LINEAR)) {
//			m_TexEnumProps[TEXTURE_MIN_FILTER] = TEXTURE_MIN_LINEAR;
//#if NAU_OPENGL_VERSION > 320
//			glSamplerParameteri(m_SamplerID, GL_TEXTURE_MIN_FILTER, 
//					translateTexMinMode((TextureMinMode)m_TexEnumProps[TEXTURE_MIN_FILTER]));
//#endif
//	}
//
//
//}




//std::map<int,int> GLTextureSampler::TransTexMagMode;
//std::map<int,int> GLTextureSampler::TransTexWrapMode; 
//std::map<int,int> GLTextureSampler::TransTexEnum;
//std::map<int,int> GLTextureSampler::TransTexMinMode;
//std::map<int,int> GLTextureSampler::TransTexCompareMode;
//std::map<int,int> GLTextureSampler::TransTexCompareFunc; 
//
//bool GLTextureSampler::init = initMaps();
//
//bool GLTextureSampler::initMaps() {
//
//	TransTexEnum[TEXTURE_WRAP_S] = GL_TEXTURE_WRAP_S;
//	TransTexEnum[TEXTURE_WRAP_T] = GL_TEXTURE_WRAP_T;
//	TransTexEnum[TEXTURE_WRAP_R] = GL_TEXTURE_WRAP_R;
//	TransTexEnum[TEXTURE_MIN_FILTER] = GL_TEXTURE_MIN_FILTER;
//	TransTexEnum[TEXTURE_MAG_FILTER] = GL_TEXTURE_MAG_FILTER;
//	TransTexEnum[TEXTURE_COMPARE_FUNC] = GL_TEXTURE_COMPARE_FUNC;
//	TransTexEnum[TEXTURE_COMPARE_MODE] = GL_TEXTURE_COMPARE_MODE;
//
//	TransTexWrapMode[TEXTURE_WRAP_REPEAT] = GL_REPEAT;
//	TransTexWrapMode[TEXTURE_WRAP_CLAMP_TO_EDGE] = GL_CLAMP_TO_EDGE;
//	TransTexWrapMode[TEXTURE_WRAP_CLAMP_TO_BORDER] = GL_CLAMP_TO_BORDER;
//	TransTexWrapMode[TEXTURE_WRAP_MIRRORED_REPEAT] = GL_MIRRORED_REPEAT;
//
//	TransTexMagMode[TEXTURE_MAG_NEAREST] = GL_NEAREST;
//	TransTexMagMode[TEXTURE_MAG_LINEAR] = GL_LINEAR;
//
//	TransTexMinMode[TEXTURE_MIN_NEAREST] = GL_NEAREST;
//	TransTexMinMode[TEXTURE_MIN_LINEAR] = GL_LINEAR;
//	TransTexMinMode[TEXTURE_MIN_NEAREST_MIPMAP_NEAREST] = GL_NEAREST_MIPMAP_NEAREST;
//	TransTexMinMode[TEXTURE_MIN_NEAREST_MIPMAP_LINEAR] = GL_NEAREST_MIPMAP_LINEAR;
//	TransTexMinMode[TEXTURE_MIN_LINEAR_MIPMAP_NEAREST] = GL_LINEAR_MIPMAP_NEAREST;
//	TransTexMinMode[TEXTURE_MIN_LINEAR_MIPMAP_LINEAR] = GL_LINEAR_MIPMAP_LINEAR;
//
//	TransTexCompareMode[TEXTURE_COMPARE_NONE] = GL_NONE;
//	TransTexCompareMode[TEXTURE_COMPARE_REF_TO_TEXTURE] = GL_COMPARE_REF_TO_TEXTURE;
//
//	TransTexCompareFunc[COMPARE_NONE] = GL_NONE;
//	TransTexCompareFunc[COMPARE_LEQUAL] = GL_LEQUAL;
//	TransTexCompareFunc[COMPARE_GEQUAL] = GL_GEQUAL;
//	TransTexCompareFunc[COMPARE_LESS] = GL_LESS;
//	TransTexCompareFunc[COMPARE_GREATER] = GL_GREATER;
//	TransTexCompareFunc[COMPARE_EQUAL] = GL_EQUAL;
//	TransTexCompareFunc[COMPARE_NOTEQUAL] = GL_NOTEQUAL;
//	TransTexCompareFunc[COMPARE_ALWAYS] = GL_ALWAYS;
//	TransTexCompareFunc[COMPARE_NEVER] = GL_NEVER;
//
//	return true;
//}

#include "nau/render/passProcessTexture.h"

#include "nau.h"
#include "nau/render/iAPISupport.h"


using namespace nau::render;

bool
PassProcessTexture::Init() {

	// BOOL
	Attribs.add(Attribute(CLEAR, "CLEAR", Enums::DataType::BOOL, false, new NauInt(false), NULL,NULL, IAPISupport::APIFeatureSupport::CLEAR_TEXTURE));
	Attribs.add(Attribute(MIPMAP, "MIPMAP", Enums::DataType::BOOL, false, new NauInt(false)));

	// INT
	Attribs.add(Attribute(CLEAR_LEVEL, "CLEAR_LEVEL", Enums::DataType::INT, false, new NauInt(-1), NULL,NULL, IAPISupport::APIFeatureSupport::CLEAR_TEXTURE_LEVEL));

	//#ifndef _WINDLL
	NAU->registerAttributes("PASS_POST_PROCESS_TEXTURE", &Attribs);
	NAU->registerAttributes("PASS_PRE_PROCESS_TEXTURE", &Attribs);
	//#endif
	return true;
}


AttribSet PassProcessTexture::Attribs;
bool PassProcessTexture::Inited = Init();


PassProcessTexture::PassProcessTexture() : m_Tex(NULL), PassProcessItem() {

	registerAndInitArrays(Attribs);
}


void 
PassProcessTexture::setItem(ITexture *tex) {

	m_Tex = tex;
}


void 
PassProcessTexture::process() {

	if (!m_Tex)
		return;

	IAPISupport *sup = IAPISupport::GetInstance();

	if (sup->apiSupport(IAPISupport::APIFeatureSupport::CLEAR_TEXTURE) && m_BoolProps[CLEAR])
		m_Tex->clear();
	if (sup->apiSupport(IAPISupport::APIFeatureSupport::CLEAR_TEXTURE_LEVEL) && m_IntProps[CLEAR_LEVEL] >= 0)
		m_Tex->clearLevel(m_IntProps[CLEAR_LEVEL]);
	if (m_BoolProps[MIPMAP])
		m_Tex->generateMipmaps();
}
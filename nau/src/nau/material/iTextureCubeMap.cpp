#include "nau/material/iTextureCubeMap.h"
#include "nau/config.h"

#include "nau/loader/iTextureLoader.h"
#ifdef NAU_OPENGL
#include "nau/render/opengl/glTextureCubeMap.h"
#endif



using namespace nau::render;
using namespace nau::loader;


ITextureCubeMap*
ITextureCubeMap::Create (std::vector<std::string> files, std::string label, bool mipmap) {

	ITextureLoader *loader[6];
	unsigned char* data[6];
	
	for (int i = 0; i < 6; i++) {
			loader[i] = ITextureLoader::create(files[i]);
			loader[i]->loadImage ();
			data[i] = loader[i]->getData();
	}

	std::string aFormat = loader[0]->getFormat();
	std::string aSizedFormat = loader[0]->getSizedFormat();
	nau::material::ITextureCubeMap *t;

#ifdef NAU_OPENGL
	t = new GLTextureCubeMap (label, files, aSizedFormat, aFormat, loader[0]->getType(), 
			loader[0]->getWidth(), data, mipmap);
#elif NAU_DIRECTX
	t = new DXTexture (aDimension, aFormat, width, height);
#endif

//#ifdef __COMPOSER__
//	int aux = loader[0]->getWidth();
//	ilConvertImage(IL_RGB,IL_UNSIGNED_BYTE);
//	iluScale(96,96,1);
//	t->bitmap = new wxBitmap(wxImage(96,96,loader[0]->getData(),true).Mirror(false));
//#endif

	for (int i = 0; i < 6; i++) {
		loader[i]->freeImage();
		delete loader[i];
	}
	return t;
}


ITextureCubeMap::ITextureCubeMap(std::string label, std::vector<std::string> files, 
							   std::string internalFormat, std::string aFormat, 
							   std::string aType, int width) :
	ITexture(label), //"TEXTURE_CUBE_MAP", internalFormat, aFormat, aType,width,width),
	m_Files(6) {

	for (int i = 0; i < 6; i++) 
		m_Files[i] = files[i];
}


//ITextureCubeMap::ITextureCubeMap(std::string label, std::vector<std::string> files): 
//	ITexture(label),
//	m_Files(6)
//{
//	for (int i = 0; i < 6; i++) 
//		m_Files[i] = files[i];
//
//}


ITextureCubeMap::~ITextureCubeMap() {

}


std::string&
ITextureCubeMap::getLabel () {

	return m_Label;
}


void
ITextureCubeMap::setLabel (std::string label) {

	m_Label = label;
}


std::string&
ITextureCubeMap::getFile (TextureCubeMapFaces i) {

	return m_Files[i];
}
 

void
ITextureCubeMap::setFile (std::string file, TextureCubeMapFaces i) {

	m_Files[i] = file;
}



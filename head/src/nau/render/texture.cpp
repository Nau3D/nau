#include <nau/render/texture.h>

#ifdef NAU_OPENGL
#include <nau/render/opengl/gltexture.h>
//#include <nau/render/opengl/gltextureMS.h>
//#include <nau/render/opengl/gltexture2dArray.h>
#endif
#include <nau/loader/textureloader.h>
#include <nau.h>


using namespace nau::render;
using namespace nau::loader;
using namespace nau;


bool
Texture::Init() {

	// INT
	Attribs.add(Attribute(ID, "ID", Enums::DataType::INT, true, new int(-1)));
	Attribs.add(Attribute(WIDTH, "WIDTH", Enums::DataType::INT, false, new int (1)));
	Attribs.add(Attribute(HEIGHT, "HEIGHT", Enums::DataType::INT, false, new int(1)));
	Attribs.add(Attribute(DEPTH, "DEPTH", Enums::DataType::INT, false, new int(1)));
	Attribs.add(Attribute(SAMPLES, "SAMPLES", Enums::DataType::INT, false, new int(1)));
	Attribs.add(Attribute(LEVELS, "LEVELS", Enums::DataType::INT, true, new int(1)));
	Attribs.add(Attribute(LAYERS, "LAYERS", Enums::DataType::INT, false, new int(1)));
	Attribs.add(Attribute(COMPONENT_COUNT, "COMPONENT_COUNT", Enums::DataType::INT, true, new int(0)));
	Attribs.add(Attribute(ELEMENT_SIZE, "ELEMENT_SIZE", Enums::DataType::INT, true, new int(0)));
	// BOOL
	Attribs.add(Attribute(MIPMAP, "MIPMAP", Enums::DataType::BOOL, true, new bool(true)));
	// ENUM
	Attribs.add(Attribute(DIMENSION, "DIMENSION", Enums::DataType::ENUM, true));
	Attribs.add(Attribute(FORMAT, "FORMAT", Enums::DataType::ENUM, true));
	Attribs.add(Attribute(TYPE, "TYPE", Enums::DataType::ENUM, true));
	Attribs.add(Attribute(INTERNAL_FORMAT, "INTERNAL_FORMAT", Enums::DataType::ENUM, false));

	return true;
}


AttribSet Texture::Attribs;
bool Texture::Inited = Init();


// For loaded images
//Texture*
//Texture::Create (std::string label, std::string internalFormat,
//				std::string aFormat, std::string aType, int width, int height, 
//				unsigned char* data) {
//
//#ifdef NAU_OPENGL
//	return new GLTexture (label, internalFormat, aFormat, aType, width, height, data);
//#elif NAU_DIRECTX
//	//Put here function for DirectX
//#endif
//}


//// For empty multisample textures
//Texture*
//Texture::CreateMS (std::string label, std::string internalFormat,
//				int width, int height, 
//				int samples) {
//
//#ifdef NAU_OPENGL
//	return new GLTextureMS (label, internalFormat, width, height, samples);
//#elif NAU_DIRECTX
//	//Put here function for DirectX
//#endif
//}


Texture*
Texture::Create (std::string label, std::string internalFormat,
	int width, int height, int depth, int layers, int levels, int samples)
{
#ifdef NAU_OPENGL
//	if (layers == 0)
		return new GLTexture (label, internalFormat, width, height, depth, layers, levels, samples);
	//else
	//	return new GLTexture2DArray(label, internalFormat, width, height, layers);
#elif NAU_DIRECTX
	//Put here function for DirectX
#endif
}


Texture*
Texture::Create(std::string label) {

#ifdef NAU_OPENGL
	return new GLTexture(label);
#elif NAU_DIRECTX
	//Put here function for DirectX
#endif
}


Texture*
Texture::Create (std::string file, std::string label, bool mipmap) {

	TextureLoader *loader = TextureLoader::create();

	int success = loader->loadImage (file);
	if (success) {
		std::string aFormat = loader->getFormat();
		nau::render::Texture *t;

	#ifdef NAU_OPENGL
		t = new GLTexture (label, aFormat, aFormat, loader->getType(), 
				loader->getWidth(), loader->getHeight(), loader->getData(), mipmap);
	#elif NAU_DIRECTX
		t = new DXTexture (aDimension, aFormat, width, height);
	#endif

	#ifdef __SLANGER__
		ilConvertImage(IL_RGB,IL_UNSIGNED_BYTE);
		iluScale(96,96,1);
	
		t->bitmap = new wxBitmap(wxImage(96,96,loader->getData(),true).Mirror(false));
	#endif
		loader->freeImage();
		delete loader;
		return t;
	}
	else {
		delete loader;
		return NULL;
	}
}


//Texture::Texture(std::string label, std::string aDimension, std::string anInternalFormat, 
//				 std::string aFormat, std::string aType, int width, int height) : m_Label (label)
//#ifdef __SLANGER__
//				 , bitmap(0), m_Bitmap(0)
//#endif
//{
//	initArrays(Attribs);
//}


//Texture::Texture(std::string label, std::string aDimension, std::string anInternalFormat, 
//				 int width, int height) : m_Label (label)
//#ifdef __SLANGER__
//				 , bitmap(0), m_Bitmap(0)
//#endif
//{
//	initArrays(Attribs);
//}


Texture::Texture(std::string label) :m_Label(label), bitmap(0), m_Bitmap(0) {

	initArrays(Attribs);
}


Texture::~Texture() {

#ifdef __SLANGER__

	if (bitmap)
		delete bitmap;	
	if (m_Bitmap)
		free (m_Bitmap);
#endif
}


int 
Texture::addAtrib(std::string name, Enums::DataType dt, void *value) {

	int id= Attribs.getNextFreeID();
	switch (dt) {

		case Enums::ENUM:
			int *k = (int *)value;
			m_EnumProps[id] = *k;
			break;
	}

	return id;

}


//void 
//Texture::setProp(int prop, Enums::DataType type, void *value) {
//
//	switch (type) {
//
//		case Enums::FLOAT:
//			assert(m_FloatProps.count(prop) != 0);
//			m_FloatProps[prop] = *(float *)value;
//			break;
//		case Enums::VEC4:
//			assert(m_Float4Props.count(prop) != 0);
//			m_Float4Props[prop].set((vec4 *)value);
//			break;
//		case Enums::INT:
//			assert(m_IntProps.count(prop) != 0);
//			m_IntProps[prop] = *(int *)value;
//			break;
//	}
//}		


#ifdef __SLANGER__

wxBitmap *
Texture::getBitmap(void) {

	return bitmap;	
}
#endif


std::string&
Texture::getLabel (void) {

	return m_Label;
}


void
Texture::setLabel (std::string label) {

	m_Label = label;
}



#include "nau/resource/fontManager.h"

#include "nau/loader/fontXmlLoader.h"
#include "nau.h"

#include <assert.h>

using namespace nau::loader;
using namespace nau::resource;


std::map<std::string,Font> FontManager::mFonts;

// As is the texture is loaded as part of a project, but the font definition file is not!!!

FontManager::FontManager()
{}

FontManager::~FontManager()
{}


void
FontManager::addFont(std::string fontName, std::string fontDefFile, std::string materialName)
{

	assert(!hasFont(fontName));

	Font f;
	f.setName(fontName);
	f.setMaterialName(materialName);
	FontXMLLoader::loadFont(&f, fontDefFile);

	mFonts[fontName] = f;
}


bool 
FontManager::hasFont(std::string fontName) 
{
	return (0 != mFonts.count(fontName));
}

const Font &
FontManager::getFont(std::string fontName)
{
	assert(hasFont(fontName));

	return(mFonts[fontName]);
}
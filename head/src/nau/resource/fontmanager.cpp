#include <assert.h>

#include "nau/resource/fontmanager.h"
#include "nau/loader/fontxmlloader.h"
#include "nau.h"

using namespace nau::loader;

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
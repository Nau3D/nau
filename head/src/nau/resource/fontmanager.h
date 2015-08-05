#ifndef FONTMANAGER_H
#define FONTMANAGER_H

#include <string>
#include <vector>
#include <map>

#include "nau/resource/font.h"
#include "nau/material/materialId.h"

using namespace nau::material;
using namespace nau::resource;

namespace nau 
{
	namespace resource
	{

		class FontManager {

		public:


			static void addFont(std::string fontName, std::string fontDefFile, std::string fontImageFile);
			static bool hasFont(std::string fontName);
			static const Font &getFont(std::string fontName);

		protected:

			FontManager();
			~FontManager();
			static std::map<std::string,Font> mFonts;
		};
	};
};


#endif
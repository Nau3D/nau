#ifndef FONTXMLLOADER_H
#define FONTXMLLOADER_H

#include "nau/geometry/font.h"

using namespace nau::geometry;

namespace nau 
{

	namespace loader 
	{
		class FontXMLLoader
		{
		public:
			static void loadFont (Font *aFont, std::string &aFilename);

		private:
			FontXMLLoader(void) {};
			~FontXMLLoader(void) {};
		};
	};
};

#endif //FONTXMLLOADER_H

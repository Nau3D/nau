#include "nau/loader/passloader.h"

#include <tinyxml.h>

#include "nau.h"
#include "nau/render/pipeline.h"
#include "nau/render/pass.h"

using namespace nau::loader;
using namespace nau::render;

bool
PassLoader::load (std::string file)
{
	TiXmlDocument doc (file.c_str());
	bool loadOkay = doc.LoadFile();

	if (true == loadOkay) {
		TiXmlHandle hDoc (&doc);
		TiXmlElement *pElem;
		TiXmlHandle hRoot (0);
		TiXmlHandle handle (0);

		{ //root
			pElem = hDoc.FirstChildElement().Element();
			if (0 == pElem) {
				return false;
			}
			hRoot = TiXmlHandle (pElem);
		}
		
		{ //pipeline
		}
	}
	return loadOkay;
}

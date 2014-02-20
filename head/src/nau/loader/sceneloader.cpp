#include <nau/loader/sceneloader.h>

#include <tinyxml.h>

#include <nau.h>

#include <nau/config.h>

#ifdef NAU_PLATFORM_WIN32
#include <nau/system/dirent.h>
#else
#include <dirent.h>
#include <sys/types.h>
#endif

using namespace nau::loader;

bool
SceneLoader::load (std::string file)
{
	TiXmlDocument doc (file.c_str());
	bool loadOkay = doc.LoadFile();

	if (true == loadOkay) {
		TiXmlHandle hDoc (&doc);
		TiXmlElement* pElem;
		TiXmlHandle hRoot (0);
		TiXmlHandle handle (0);

		{ //root
			pElem = hDoc.FirstChildElement().Element();
			if (0 == pElem) {
				return false;
			}
			hRoot = TiXmlHandle (pElem);
		}
	
		{ //Scenes
		}
	
	}

	return loadOkay;
}

#include "nau/loader/fontXmlLoader.h"

#include <tinyxml.h>

#include "nau/errors.h"

using namespace nau::loader;
using namespace nau::geometry;

void
FontXMLLoader::loadFont (Font *aFont, std::string &aFilename)
{
	std::string s = aFilename;
	int numChars,height;
	TiXmlDocument doc(s.c_str());
	bool loadOK = doc.LoadFile();

	if (!loadOK) {
		NAU_THROW("Parsing Error -%s- Line(%d) Column(%d) in file: %s", doc.ErrorDesc(), doc.ErrorRow(), doc.ErrorCol(),aFilename.c_str());
	}		
	
	TiXmlHandle hDoc(&doc);
	TiXmlHandle hRoot(0);
	TiXmlElement *pElem;

	pElem = hDoc.FirstChildElement().Element();
	if (0 == pElem)
		NAU_THROW("Not a XML file: %s", doc.ErrorDesc(), doc.ErrorRow(), doc.ErrorCol(),aFilename.c_str());

	hRoot = TiXmlHandle(pElem);
	
	pElem->QueryIntAttribute("numchars",&numChars);

	if (numChars == 0)
		NAU_THROW("Zero chars in file: %s", doc.ErrorDesc(), doc.ErrorRow(), doc.ErrorCol(),aFilename.c_str());

	hRoot = hRoot.FirstChild("characters");
	pElem = hRoot.FirstChild("chardata").Element();
	if (pElem) {
		pElem->QueryIntAttribute("hgt",&height);
		if (pElem)
			aFont->setFontHeight(height);
	}
	int code,width,A,C;
	float x1,x2,y1,y2;

	for(; 0 != pElem; pElem = pElem->NextSiblingElement()) {

		pElem->QueryIntAttribute("char",&code);
		pElem->QueryIntAttribute("wid",&(width));
		pElem->QueryFloatAttribute("X1", &(x1));
		pElem->QueryFloatAttribute("X2", &(x2));
		pElem->QueryFloatAttribute("Y1", &(y1));
		pElem->QueryFloatAttribute("Y2", &(y2));
		pElem->QueryIntAttribute("A", &(A));
		pElem->QueryIntAttribute("C", &(C));
		aFont->addChar(code,width,x1,x2,y1,y2,A,C);
	}
}
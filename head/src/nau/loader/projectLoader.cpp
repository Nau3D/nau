#include "nau/loader/projectLoader.h"

#include "nau.h"

#include "nau/config.h"
#include "nau/slogger.h"

#include "nau/event/interpolatorFactory.h"
#include "nau/event/objectAnimation.h"
#include "nau/event/route.h"
#include "nau/event/sensorfactory.h"

#include "nau/geometry/primitive.h"
#include "nau/geometry/terrain.h"
#include "nau/interface/interface.h"
#include "nau/material/iBuffer.h"
#include "nau/material/programValue.h"
#include "nau/material/uniformBlockManager.h"
#include "nau/math/number.h"
#include "nau/render/iAPISupport.h"
#include "nau/render/passCompute.h"
#include "nau/render/passFactory.h"
#include "nau/render/passProcessTexture.h"
#include "nau/render/passProcessBuffer.h"
#ifdef NAU_OPTIX
#include "nau/render/passOptixPrime.h"
#include "nau/render/passOptix.h"
#endif
#include "nau/render/passQuad.h"
#include "nau/render/pipeline.h"
#include "nau/render/iRenderTarget.h"

#include "nau/scene/geometricObject.h"
#include "nau/scene/sceneObjectFactory.h"


#include "nau/system/TextUtil.h"



#ifdef NAU_PLATFORM_WIN32
#include <dirent.h>
#else
#include <dirent.h>
#include <sys/types.h>
#endif

#include <memory>

//#ifdef GLINTERCEPTDEBUG
//#include "nau/loader/projectLoaderDebugLinker.h"
//#endif


using namespace nau::loader;
using namespace nau::math;
using namespace nau::material;
using namespace nau::render;
using namespace nau::scene;
using namespace nau::geometry;
using namespace nau::system;
//
using namespace nau::event_;
//

std::string ProjectLoader::s_Path = "";
std::string ProjectLoader::s_File = "";
std::string ProjectLoader::s_Dummy;

std::map<std::string, float> ProjectLoader::s_Constants;

char ProjectLoader::s_pFullName[256] = "";

vec4 ProjectLoader::s_Dummy_vec4;
vec2 ProjectLoader::s_Dummy_vec2;
bvec4 ProjectLoader::s_Dummy_bvec4;
float ProjectLoader::s_Dummy_float;
int ProjectLoader::s_Dummy_int;
unsigned int ProjectLoader::s_Dummy_uint;
bool ProjectLoader::s_Dummy_bool;
uivec3 ProjectLoader::s_Dummy_uivec3;
uivec2 ProjectLoader::s_Dummy_uivec2;



std::string 
ProjectLoader::readFile(TiXmlElement *p, std::string tag, std::string item) {

	std::string file;
	int res = p->QueryStringAttribute(tag.c_str(), &file);
	if (TIXML_SUCCESS != res) {
		NAU_THROW("File %s\n%s\nTag %s is required", ProjectLoader::s_File.c_str(), item.c_str(), tag.c_str());
	}
	std::string aux = File::GetFullPath(File::GetPath(ProjectLoader::s_File), file);
	if (!File::Exists(aux)) {
		NAU_THROW("File %s\n%s\nFile not found: %s", ProjectLoader::s_File.c_str(), item.c_str(), aux.c_str());
	}
	return aux;
}





std::string
ProjectLoader::toLower(std::string strToConvert) {

	s_Dummy = strToConvert;
   for (std::string::iterator p = s_Dummy.begin(); s_Dummy.end() != p; ++p)
       *p = tolower(*p);

   return s_Dummy;

}


int
ProjectLoader::readItemFromLib(TiXmlElement *p, std::string tag, std::string *lib, std::string *item) {

	int res1, res2;
	res1 = p->QueryStringAttribute("fromLibrary", lib);
	res2 = p->QueryStringAttribute(tag.c_str(), item);

	if (TIXML_SUCCESS != res2)
		return ITEM_NAME_NOT_SPECIFIED;
	else return OK;
}


int
ProjectLoader::readItemFromLib(TiXmlElement *p, std::string tag, std::string *fullName) {

	std::string item, lib = "";
	int res = readItemFromLib(p, tag, &lib, &item);

	if (OK != res)
		return res;

	if (lib == "")
		*fullName = item;
	else
		*fullName = lib + "::" + item;
	return OK;
}


bool
ProjectLoader::readFloatAttribute(TiXmlElement *p, std::string label, float *value) {

	std::string s;

	if (TIXML_SUCCESS != p->QueryFloatAttribute(label.c_str(), value)) {
		if (TIXML_SUCCESS != p->QueryStringAttribute(label.c_str(), &s) || !isConstantDefined(s))
			return false;
		else {
			*value = s_Constants[s];
			return true;
		}
	}
	return true;
}


bool
ProjectLoader::readIntAttribute(TiXmlElement *p, std::string label, int *value) {

	std::string s;

	if (TIXML_SUCCESS != p->QueryIntAttribute(label.c_str(), value)) {
		if (TIXML_SUCCESS != p->QueryStringAttribute(label.c_str(), &s) || !isConstantDefined(s))
			return false;
		else {
			*value = (int)s_Constants[s];
			return true;
		}
	}
	return true;
}


bool
ProjectLoader::readUIntAttribute(TiXmlElement *p, std::string label, unsigned int *value) {

	std::string s;

	if (TIXML_SUCCESS != p->QueryUnsignedAttribute(label.c_str(), value)) {
		if (TIXML_SUCCESS != p->QueryStringAttribute(label.c_str(), &s) || !isConstantDefined(s))
			return false;
		else {
			*value = (unsigned int)s_Constants[s];
			return true;
		}
	}
	return true;
}


Data *
ProjectLoader::readChildTag(std::string pName, TiXmlElement *p, Enums::DataType type, AttribSet &attribs) {

	std::string s;

	switch (type) {
	
		case Enums::FLOAT:
			if (!readFloatAttribute(p, "value", &s_Dummy_float))
				NAU_THROW("File %s: Element %s: Float Attribute %s without a value", ProjectLoader::s_File.c_str(),pName.c_str(), p->Value()); 
			return new NauFloat(s_Dummy_float);
			break;
		case Enums::VEC4:
			s_Dummy_vec4 = vec4(0.0f);
			if ((readFloatAttribute(p, "x", &s_Dummy_vec4.x) || readFloatAttribute(p, "r", &s_Dummy_vec4.x)) 
				&& ((readFloatAttribute(p, "y", &s_Dummy_vec4.y) || readFloatAttribute(p, "g", &s_Dummy_vec4.y)) 
				&& ((readFloatAttribute(p, "z", &s_Dummy_vec4.z) || readFloatAttribute(p, "b", &s_Dummy_vec4.z))))) {
			
				if (!readFloatAttribute(p, "w", &s_Dummy_vec4.w))
					readFloatAttribute(p, "a", &s_Dummy_vec4.w);

					return new vec4(s_Dummy_vec4);

			}
			else
				NAU_THROW("File %s: Element %s: Vec4 Attribute %s has absent or incomplete value (x,y and z are required, w is optional)", ProjectLoader::s_File.c_str(),pName.c_str(),p->Value()); 
			break;
		case Enums::VEC2:
			s_Dummy_vec2 = vec2(0.0f);
			if ((readFloatAttribute(p, "x", &s_Dummy_vec2.x) || readFloatAttribute(p, "width", &s_Dummy_vec2.x)) &&
				(readFloatAttribute(p, "y", &s_Dummy_vec2.y) || readFloatAttribute(p, "height", &s_Dummy_vec2.y))) {
				return new vec2(s_Dummy_vec2);
			}
			else
				NAU_THROW("File %s: Element %s: Vec2 Attribute %s has absent or incomplete value (x,y  or width,height are required)", ProjectLoader::s_File.c_str(), pName.c_str(), p->Value());
			break;

		case Enums::BVEC4:
			if (TIXML_SUCCESS == p->QueryBoolAttribute("x", &(s_Dummy_bvec4.x)) 
				&& TIXML_SUCCESS == p->QueryBoolAttribute("y", &(s_Dummy_bvec4.y))
				&& TIXML_SUCCESS == p->QueryBoolAttribute("z", &(s_Dummy_bvec4.z))
				&& TIXML_SUCCESS == p->QueryBoolAttribute("w", &(s_Dummy_bvec4.w))) {
			
				return new bvec4(s_Dummy_bvec4);
			}
			else
				NAU_THROW("File %s: Element %s: BVec4Attribute %s has absent or incomplete value (x,y,z and w are required)", ProjectLoader::s_File.c_str(),pName.c_str(),p->Value()); 
			break;

		case Enums::INT:
			if (!readIntAttribute(p, "value", &s_Dummy_int))
				NAU_THROW("File %s: Element %s: Int Attribute %s without a value", ProjectLoader::s_File.c_str(),pName.c_str(), p->Value()); 
			return new NauInt(s_Dummy_int);
			break;
		case Enums::UINT:
			if (!readUIntAttribute(p, "value", &s_Dummy_uint))
				NAU_THROW("File %s: Element %s: UInt Attribute %s without a value", ProjectLoader::s_File.c_str(),pName.c_str(), p->Value()); 
			return new NauUInt(s_Dummy_uint);
			break;
		case Enums::UIVEC2:

			if ((readUIntAttribute(p, "x", &s_Dummy_uivec2.x) || readUIntAttribute(p, "width", &s_Dummy_uivec2.x)) &&
				(readUIntAttribute(p, "y", &s_Dummy_uivec2.y) || readUIntAttribute(p, "height", &s_Dummy_uivec2.y))) {
				return new uivec2(s_Dummy_uivec2);
			}
			else
				NAU_THROW("File %s: Element %s: UIVec2Attribute %s has absent or incomplete value (x,y   or width,height are required are required)", ProjectLoader::s_File.c_str(), pName.c_str(), p->Value());
			break;
		case Enums::UIVEC3:

			if (readUIntAttribute(p, "x", &(s_Dummy_uivec3.x)) &&
				readUIntAttribute(p, "y", &(s_Dummy_uivec3.y)) &&
				readUIntAttribute(p, "z", &(s_Dummy_uivec3.z))) {

				return new uivec3(s_Dummy_uivec3);
			}
			else
				NAU_THROW("File %s: Element %s: UIVec3Attribute %s has absent or incomplete value (x,y,z are required)", ProjectLoader::s_File.c_str(), pName.c_str(), p->Value());
			break;

		case Enums::BOOL:
			if (TIXML_SUCCESS != p->QueryBoolAttribute("value", &s_Dummy_bool))
				NAU_THROW("File %s: Element %s: Bool Attribute %s without a value", ProjectLoader::s_File.c_str(),pName.c_str(), p->Value()); 
			return new NauInt(s_Dummy_bool);
			break;
		case Enums::ENUM:
			if (TIXML_SUCCESS != p->QueryStringAttribute("value", &s))
				NAU_THROW("File %s: Element %s: Enum Attribute %s without a value", ProjectLoader::s_File.c_str(),pName.c_str(), p->Value()); 

			s_Dummy_int = attribs.getListValueOp(attribs.getID(p->Value()), s); 
			return new NauInt(s_Dummy_int);
			break;
		default:
			assert(false && "Missing attribute type in function ProjectLoader::readChildTag");
	}
	return NULL;
}


Data *
ProjectLoader::readAttribute(std::string tag, std::unique_ptr<Attribute> &attrib, TiXmlElement *p) {

	std::string s;
	const char *name = tag.c_str();
	Enums::DataType type = attrib->getType();
	switch (type) {
	
		case Enums::FLOAT:
			if (readFloatAttribute(p, tag, &s_Dummy_float))
				return new NauFloat(s_Dummy_float);
			else
				return NULL;
			break;

		case Enums::INT:
			if (readIntAttribute(p, tag, &s_Dummy_int))
				return new NauInt(s_Dummy_int);
			else
				return NULL;
			break;
		case Enums::UINT:
			if (readUIntAttribute(p, tag, &s_Dummy_uint))
				return new NauUInt(s_Dummy_uint);
			else
				return NULL;
			break;

		case Enums::BOOL:
			if (TIXML_SUCCESS != p->QueryBoolAttribute(name, &s_Dummy_bool))
				return NULL;
			else
				return new NauInt(s_Dummy_bool);
			break;
		case Enums::ENUM:
			if (TIXML_SUCCESS != p->QueryStringAttribute(name, &s))
				return NULL;
			else {
				s_Dummy_int = attrib->getOptionValue(s);
				return new NauInt(s_Dummy_int);
			}
			break;
		default:
			assert(false && "Missing attribute type in function ProjectLoader::readAttribute");
	}
	return NULL;
}


bool 
ProjectLoader::isConstantDefined(std::string s) {

	return s_Constants.count(s) != 0;
}


bool
ProjectLoader::isExcluded(std::string attr, std::vector<std::string> &excluded) {

	for (auto s : excluded) {

		if (s == attr)
			return true;
	}
	return false;
}


void 
ProjectLoader::buildExcludedVector(std::map<std::string, std::unique_ptr<Attribute> > &attrs, 
			std::set<std::string> &included, 
			std::vector<std::string> *excluded) {

	for (auto& attr : attrs) {

		if (included.count(attr.first) == 0) {

			excluded->push_back(attr.first);
		}
	}
}


std::string &
ProjectLoader::getValidValuesString(std::unique_ptr<Attribute> &a) {

	Enums::DataType type = a->getType();
	if (type != Enums::ENUM && type != Enums::BOOL) {

		std::shared_ptr<Data> min = a->getMin();
		std::shared_ptr<Data> max = a->getMax();

		if (min != NULL && max != NULL) {
			std::string smin = Enums::valueToString(type, min.get());
			std::string smax = Enums::valueToString(type, max.get());
			s_Dummy = "between" + smin + " and " + smax;
		}
		else if (min != NULL) {
			s_Dummy = "greater or equal than " + Enums::valueToString(type, min.get());
		}
		else if (max != NULL) {
			s_Dummy = "less or equal than " + Enums::valueToString(type, max.get());

		}
	}
	else if (type == Enums::BOOL) {
	
		s_Dummy = "true, false";
	}
	else {
		std::vector<std::string> validValues;
		a->getOptionStringListSupported(&validValues);
		TextUtil::Join(validValues, ", ", &s_Dummy);
	}
	return s_Dummy;

}


void 
ProjectLoader::readAttributes(std::string parent, AttributeValues *anObj, nau::AttribSet &attribs, std::vector<std::string> &excluded, TiXmlElement *pElem) {

	TiXmlElement *p = pElem->FirstChildElement();
	std::map<std::string, std::unique_ptr<nau::Attribute>>& attributes = attribs.getAttributes();
	Data *value;

	TiXmlAttribute* attrib = pElem->FirstAttribute();

	while (attrib) {
		 // skip previously excluded attributes
		if (!isExcluded(attrib->Name(), excluded)) {
			 // trying to define an attribute that does not exist?		
			if (attributes.count(attrib->Name()) == 0) {
				std::vector<std::string> attribVec;
				getKeystoVector(attributes, &attribVec);
				attribVec.insert(attribVec.end(), excluded.begin(), excluded.end());
				std::string result;
				TextUtil::Join(attribVec, ", ", &result);
				NAU_THROW("File %s: Element %s: \"%s\" is not a valid attribute\nValid tags are: %s", 
					ProjectLoader::s_File.c_str(), parent.c_str(), attrib->Name(), result.c_str());
			}
			std::unique_ptr<Attribute> &a = attributes[attrib->Name()];
			// trying to set the value of a read only attribute?
			if (a->getReadOnlyFlag())
				NAU_THROW("File %s\nElement %s: \"%s\" is a read-only attribute", ProjectLoader::s_File.c_str(), parent.c_str(), attrib->Name());

			int id = a->getId();
			value = readAttribute(a->getName(), a, pElem);
			if (!anObj->isValid(id, a->getType(), value)) {
				std::string s = getValidValuesString(a);
				if (s != "") {
					NAU_THROW("File %s\nElement %s: \"%s\" has an invalid value\nValid values are\n%s", ProjectLoader::s_File.c_str(), parent.c_str(), attrib->Name(), s.c_str());
				}
				else {
					NAU_THROW("File %s\nElement %s: \"%s\" is not supported", ProjectLoader::s_File.c_str(), parent.c_str(), attrib->Name());
				}
			}

			anObj->setProp(id, a->getType(), value);
			delete value;
		}
		attrib = attrib->Next();
	}
}


void
ProjectLoader::validateObjectAttribute(std::string type, std::string context, std::string component, std::string *message) {

	if (!NAU->validateObjectType(type)) {
		std::vector<std::string> validTypes;
		NAU->getValidObjectTypes(&validTypes);
		std::string messageAux;
		TextUtil::Join(validTypes, ", ", &messageAux);
		*message = "Invalid type value\nValid values are: " + messageAux;
		return;
	}
	if (context != "CURRENT" && !NAU->validateObjectContext(type, context)) {
		*message = "Invalid context value\nValid values are name of objects (or CURRENT if aplicable)";
		return;
	}
	if (!NAU->validateObjectComponent(type, component)) {
		std::vector<std::string> validComponents;
		NAU->getValidObjectComponents(type, &validComponents);
		std::string messageAux;
		TextUtil::Join(validComponents, ", ", &messageAux);
		*message = "Invalid component value\nValid values are: " + messageAux;
		return;
	}
	*message = "";
}


void
ProjectLoader::validateObjectTypeAndComponent(std::string type, std::string component, std::string *message) {

	if (!NAU->validateObjectType(type)) {
		std::vector<std::string> validTypes;
		NAU->getValidObjectTypes(&validTypes);
		std::string messageAux;
		TextUtil::Join(validTypes, ", ", &messageAux);
		*message = "Invalid type value\nValid values are: " + messageAux;
		return;
	}
	if (!NAU->validateObjectComponent(type, component)) {
		std::vector<std::string> validComponents;
		NAU->getValidObjectComponents(type, &validComponents);
		std::string messageAux;
		TextUtil::Join(validComponents, ", ", &messageAux);
		*message = "Invalid component value\nValid values are: " + messageAux;
		return;
	}
	*message = "";
}


void
ProjectLoader::readChildTags(std::string parent, AttributeValues *anObj, nau::AttribSet &attribs, std::vector<std::string> &excluded, TiXmlElement *pElem, bool showOnlyExcluded) {

	TiXmlElement *p = pElem->FirstChildElement();
	std::map<std::string, std::unique_ptr<nau::Attribute>> &attributes = attribs.getAttributes();
	Data *value;

	while (p) {
		// skip previously excluded attributes
		if (!isExcluded(p->Value(), excluded)) {
			// trying to define an attribute that does not exist?		
			if (attributes.count(p->Value()) == 0) {
				std::vector<std::string> attribVec;
				if (showOnlyExcluded) {
					attribVec = excluded;
				}
				else {
					getKeystoVector(attributes, &attribVec);
					attribVec.insert(attribVec.end(), excluded.begin(), excluded.end());
				}
				std::string result;
				TextUtil::Join(attribVec, ", ", &result);
				NAU_THROW("File %s\nElement %s\nInvalid child tag \"%s\"\nValid tags are: %s", 
					ProjectLoader::s_File.c_str(), parent.c_str(), p->Value(), result.c_str());
			}
			std::unique_ptr<Attribute> &a = attributes[p->Value()];
			// trying to set the value of a read only attribute?
			if (a->getReadOnlyFlag())
				NAU_THROW("File %s\nElement %s\nRead-only tag\"%s\"", ProjectLoader::s_File.c_str(), parent.c_str(), p->Value());

			int id = a->getId();
			Enums::DataType type = a->getType();
			value = readChildTag(parent, p, type, attribs);
			if (!anObj->isValid(id, type, value)) {
				std::string s = getValidValuesString(a);
				if (s != "") {
					NAU_THROW("File %s\nElement %s: \"%s\" has an invalid value\nValid values are\n%s", ProjectLoader::s_File.c_str(), parent.c_str(), a->getName().c_str(), s.c_str());
				}
				else {
					NAU_THROW("File %s\nElement %s: \"%s\" is not supported", ProjectLoader::s_File.c_str(), parent.c_str(), a->getName().c_str());
				}
			}
			anObj->setProp(id, a->getType(), value);
			delete value;
		}
		p = p->NextSiblingElement();
	}
}


void 
ProjectLoader::checkForNonValidChildTags(std::string parent, std::vector<std::string> &ok, TiXmlElement *pElem) {

	if (pElem == NULL)
		return;

	TiXmlElement *p = pElem->FirstChildElement();

	while (p) {
		// skip previously excluded attributes
		if (!isExcluded(p->Value(), ok)) {
			std::string result;
			TextUtil::Join(ok, ", ", &result);
			// trying to define an attribute that does not exist?		
			NAU_THROW("File %s\nElement %s\n\"%s\" is not a valid child tag\nValid tags are: %s", 
				ProjectLoader::s_File.c_str(), parent.c_str(), p->Value(), result.c_str());
		}
		p = p->NextSiblingElement();
	}
}


void 
ProjectLoader::checkForNonValidAttributes(std::string parent, std::vector<std::string> &ok, TiXmlElement *pElem) {

	TiXmlAttribute* attrib = pElem->FirstAttribute();

	while (attrib) {
		// skip previously excluded attributes
		if (!isExcluded(attrib->Name(), ok)) {
			std::string result;
			TextUtil::Join(ok, ", ", &result);
			// trying to define an attribute that does not exist?		
			NAU_THROW("File %s\nElement %s\n\"%s\" is not an attribute\nValid attributes are: %s", 
				ProjectLoader::s_File.c_str(), parent.c_str(), attrib->Name(), result.c_str());
		}
		attrib = attrib->Next();
	}
}


void
ProjectLoader::getKeystoVector(std::map<std::string, std::unique_ptr<Attribute> > &theMap, std::vector<std::string> *result) {

	for (auto& s : theMap) {
		if (!s.second->getReadOnlyFlag())
		result->push_back(s.first);
	}
}


/*--------------------------------------------------------------------
Project Specification

<?xml version="1.0" ?>
<project name="teste1-shadows" width=512 height=512>

	<assets>
		...
	</assets>

	<pipelines>
		...
	</pipelines>

	<interface>
		...
	</interface>

</project>

-------------------------------------------------------------------*/

void
ProjectLoader::load (std::string file, int *width, int *height)
{
#if NAU_DEBUG == 1
	LOG_INFO ("Loading project: %s", file.c_str()); 
#endif

	ProjectLoader::s_Path = File::GetPath(file);
	ProjectLoader::s_File = file;
	s_Constants.clear();

	TiXmlDocument doc (file.c_str());
	bool loadOkay = doc.LoadFile();
	std::vector<std::string> matLibs;

	if (!loadOkay) {
		NAU_THROW("Parsing Error -%s- Line(%d) Column(%d) in file: %s", doc.ErrorDesc(), doc.ErrorRow(), doc.ErrorCol(),file.c_str());
	}

	TiXmlHandle hDoc (&doc);
	TiXmlHandle hRoot (0);
	TiXmlElement *pElem;

	pElem = hDoc.FirstChildElement().Element();
	if (0 == pElem) {
		NAU_THROW("Parsing Error in file: %s", file.c_str());
	}
	hRoot = TiXmlHandle (pElem);

	std::string name;

	try {
		*width = 0;
		*height = 0;

		if (TIXML_SUCCESS != pElem->QueryStringAttribute("name", &name))
			NAU_THROW("Project without a name!");

		if (TIXML_SUCCESS == pElem->QueryIntAttribute("width",width) &&
			TIXML_SUCCESS == pElem->QueryIntAttribute("height",height)) {
				if (*width <= 0 || *height <= 0) {
					*width = 512;
					*height = 512;
				}
				NAU->setWindowSize(*width, *height);
		}
		std::vector<std::string> ok = {"name", "width", "height"};
		checkForNonValidAttributes("project", ok, pElem);
		
//#ifdef GLINTERCEPTDEBUG
//		loadDebug(hRoot);
//#endif
		loadAssets (hRoot, matLibs);
		loadPipelines (hRoot);
		loadInterface(hRoot);

	}
	catch(std::string &s) {
		throw(s);
	}
	std::vector<std::string> v = { "assets" , "pipelines" , "interface"};
	checkForNonValidChildTags("project", v, pElem);

#if NAU_DEBUG == 1
	LOG_INFO ("Loading done"); 
#endif
	NAU->setProjectName(name);
}


/* ----------------------------------------------------------------
Specification of User Attributes:

<attributes>
	<attribute context="LIGHT" name="DIR" type="VEC4" x="-1.0" y="-1.0" z="-1.0" w = "0" />
	<attribute context="CAMERA" name="DIST" type="FLOAT" value="10" />
	<attribute context="STATE" name="FOG_MIN_DIST" type="FLOAT" value = 0.0 />
	<attribute context="STATE" name="FOG_MAX_DIST" type="FLOAT" value = 100.0 />
</attributes>

Notes:
Context see nau.cpp (getAttribs)
name is the name of the attribute
type see readChildTag()

----------------------------------------------------------------- */


void 
ProjectLoader::loadUserAttrs(TiXmlHandle handle) 
{
	TiXmlElement *pElem, *pElem2;
	std::string delim="\n", s;

	pElem2 = handle.FirstChild("attributes").Element();
	std::vector<std::string> v = {"attribute"};
	checkForNonValidChildTags("attributes", v, pElem2);

	pElem = handle.FirstChild ("attributes").FirstChild ("attribute").Element();
	for (; 0 != pElem; pElem = pElem->NextSiblingElement("attribute")) {


		//std::vector<std::string> ok;
		//ok.push_back("context"); ok.push_back("name"); ok.push_back("type");
		//checkForNonValidAttributes("attribute", ok, pElem);

		const char *pContext = pElem->Attribute("context");
		const char *pName = pElem->Attribute("name");
		const char *pType = pElem->Attribute("type");

		if (0 == pContext) {
			NAU_THROW("File %s\nAttribute without a context", ProjectLoader::s_File.c_str());
		}
		if (!NAU->validateUserAttribContext(pContext)) {
			std::vector<std::string> objTypes;
			NAU->getObjTypeList(&objTypes);
			nau::system::TextUtil::Join(objTypes, delim.c_str(), &s);
			NAU_THROW("File %s\nAttribute with an invalid context %s\nValid Values are: \n%s", ProjectLoader::s_File.c_str(), pContext, s.c_str());
		}
		if (0 == pName) {
			NAU_THROW("File %s\nAttribute without a name", ProjectLoader::s_File.c_str());
		}
		if (!NAU->validateUserAttribName(pContext, pName)) {
			NAU_THROW("File %s\nAttribute name %s is already in use in context %s", ProjectLoader::s_File.c_str(), pName, pContext);
		}
		if (0 == pType) {
			NAU_THROW("File %s\nAttribute %s without a type", ProjectLoader::s_File.c_str(), pName);
		}
		if (!Attribute::isValidUserAttrType(pType)) {
			nau::system::TextUtil::Join(Attribute::getValidUserAttrTypes(), delim.c_str(), &s);
			NAU_THROW("File %s\nAttribute %s with an invalid type: %s\nValid types are: \n%s", ProjectLoader::s_File.c_str(), pName, pType, s.c_str());
		}

		AttribSet *attribs = NAU->getAttribs(pContext);
		Enums::DataType dt = Enums::getType(pType);
		Data *v = readChildTag(pName, pElem, dt, *attribs);
		//Attribute *a = new Attribute(attribs->getNextFreeID(), pName, dt, false, v);
		attribs->add(Attribute(attribs->getNextFreeID(), pName, dt, false, v));
		std::string s;
		SLOG("User Attribute : %s::%s", pContext, pName);
				
	}
}


/* ----------------------------------------------------------------
Specification of Constants:

<constants>
	<constant name="BufferDim" value=128 />
	<constant name="PI" value=3.14 />
</constants>

Constants can be used everywhere an attribute requires a number. Constants are read as floats 
and can be used to replace any numeric type. 

----------------------------------------------------------------- */

void 
ProjectLoader::loadConstants(TiXmlHandle &handle) 
{
	TiXmlElement *pElem, *pElem2;
	std::string delim="\n", s;

	pElem2 = handle.FirstChild("constants").Element();
	std::vector<std::string> v = {"constant"};
	checkForNonValidChildTags("constants", v, pElem2);

	pElem = handle.FirstChild ("constants").FirstChild ("constant").Element();
	for (; 0 != pElem; pElem = pElem->NextSiblingElement("constant")) {

		const char *pName = pElem->Attribute("name");

		if (0 == pName) {
			NAU_THROW("File %s\nConstant without a name", ProjectLoader::s_File.c_str());
		}

		float value;

		if (TIXML_SUCCESS != pElem->QueryFloatAttribute("value", &value)) {
			NAU_THROW("File %s\nConstant %s value is absent or is not a number", ProjectLoader::s_File.c_str(), pName);
		}

		s_Constants[pName] = value;

		SLOG("Constant %s: %f", pName, value);
				
	}
}

/* ----------------------------------------------------------------
Specification of the scenes:

		<scenes>
			<scene name="MainScene" param="SWAP_YZ">
				<file name="..\ntg-bin-3\fonte-finallambert.dae"/>
				<folder name="..\ntg-bin-pl3dxiv"/>
			</scene>
			...
		</scenes>

scenes can have multiple "scene" defined
each scene can have files and folders .
the path may be relative to the project file or absolute
type see sceneFactory

param is passed to the loader
	SWAP_YZ to indicate that ZY axis should be swaped 
	USE_ADJACENCY to create an index list with adjacency information
	UNITIZE causes the scene vertices to be scaled to fit in a box from -1 to 1


----------------------------------------------------------------- */



void 
ProjectLoader::loadScenes(TiXmlHandle handle) 
{
	TiXmlElement *pElem;

	TiXmlElement *pElem2 = handle.FirstChild("scenes").Element();
	std::vector<std::string> v;
	v.push_back("scene"); 
	checkForNonValidChildTags("scenes", v, pElem2);

	pElem = handle.FirstChild ("scenes").FirstChild ("scene").Element();
	for (; 0 != pElem; pElem = pElem->NextSiblingElement("scene")) {
		const char *pName = pElem->Attribute("name");
		const char *pType = pElem->Attribute("type");
		const char *pFilename = pElem->Attribute("filename");
		const char *pParam = pElem->Attribute("param");

		std::string s;

		if (0 == pName) {
			NAU_THROW("File %s\nscene has no name", ProjectLoader::s_File.c_str());
		}

		SLOG("Scene : %s", pName);

		if (pParam == NULL)
			s = "";
		else
			s = pParam;

		std::shared_ptr<IScene> is;
		if (0 == pType)
			is = RENDERMANAGER->createScene(pName);
		else {
			is = RENDERMANAGER->createScene(pName, pType);
			if (!is)
				NAU_THROW("File %s\nScene %s\nInvalid type for scene", ProjectLoader::s_File.c_str(), pName);
		}

		// the filename should point to a scene
		if (0 != pFilename) {

			if (!File::Exists(File::GetFullPath(ProjectLoader::s_Path, pFilename)))
				NAU_THROW("File %s\nScene %s\nFile %s does not exist", ProjectLoader::s_File.c_str(), pName, pFilename);

			try {
				NAU->loadAsset(File::GetFullPath(ProjectLoader::s_Path, pFilename), pName, s);
			}
			catch (std::string &s) {
				throw(s);
			}
		}
		else {
			handle = TiXmlHandle(pElem);
			TiXmlElement* pElementAux;
			pElementAux = handle.FirstChild("geometry").Element();
			for (; 0 != pElementAux; pElementAux = pElementAux->NextSiblingElement("geometry")) {
				const char *pNameSO = pElementAux->Attribute("name");
				const char *pPrimType = pElementAux->Attribute("type");
				const char *pMaterial = pElementAux->Attribute("material");

				if (pNameSO == NULL)
					NAU_THROW("File %s\nScene %s\nGeometry object without a name", ProjectLoader::s_File.c_str(), pName);

				if (pPrimType == NULL)
					NAU_THROW("File %s\nScene %s, Object:%s\ntype is not defined", ProjectLoader::s_File.c_str(), pName, pNameSO);

				std::shared_ptr<GeometricObject> go = 
					dynamic_pointer_cast<GeometricObject>(nau::scene::SceneObjectFactory::Create("Geometry"));
				if (go == NULL)
					NAU_THROW("File %s\nScene %s\nInvalid scene type", ProjectLoader::s_File.c_str(), pName);
				
				go->setName(pNameSO);

				bool alreadyThere = false;
				std::shared_ptr<Primitive> p;
				if (RESOURCEMANAGER->hasRenderable(pNameSO, "")) {
					alreadyThere = true;
					p = dynamic_pointer_cast<Primitive>(RESOURCEMANAGER->getRenderable(pNameSO, ""));
				}
				else {
					p = dynamic_pointer_cast<Primitive>(RESOURCEMANAGER->createRenderable(pPrimType, pNameSO));
				}
				if (!p)
					NAU_THROW("File %s\nScene %s\nPrimitive %s has an invalid primitive type - %s", ProjectLoader::s_File.c_str(), pName, pNameSO, pPrimType);

				if (!alreadyThere) {
					AttribSet *as = NAU->getAttribs(pPrimType);
					if (as != NULL) {
						std::vector <std::string> excluded;
						excluded.push_back("name"); excluded.push_back("type"); excluded.push_back("material");
						readAttributes(pName, (AttributeValues *)p.get(), *as, excluded, pElementAux);
					}
					p->build();
				}
				go->setRenderable(dynamic_pointer_cast<IRenderable>(p));

				if (!alreadyThere) {
					if (pMaterial) {
						if (!MATERIALLIBMANAGER->hasMaterial(DEFAULTMATERIALLIBNAME, pMaterial)) {
							MATERIALLIBMANAGER->createMaterial(pMaterial);
						}
						go->setMaterial(pMaterial);
					}
				}

				std::vector<std::string> excluded;
				readChildTags(pNameSO, (AttributeValues *)go.get(), SceneObject::Attribs, excluded, pElementAux);

				is->add(dynamic_pointer_cast<SceneObject>(go));
			}
			pElementAux = handle.FirstChild("terrain").Element();
			for (; 0 != pElementAux; pElementAux = pElementAux->NextSiblingElement("terrain")) {
				const char *pNameSO = pElementAux->Attribute("name");
				const char *pMaterial = pElementAux->Attribute("material");
				const char *pHeightMap = pElementAux->Attribute("heightMap");

				if (pNameSO == NULL)
					NAU_THROW("File %s\nScene %s\nTerrain object without a name", ProjectLoader::s_File.c_str(), pName);

				std::shared_ptr<GeometricObject> go =
					dynamic_pointer_cast<GeometricObject>(nau::scene::SceneObjectFactory::Create("Geometry"));

				go->setName(pNameSO);

				bool alreadyThere = false;
				std::shared_ptr<nau::geometry::Terrain> p;
				if (RESOURCEMANAGER->hasRenderable(pNameSO, "")) {
					alreadyThere = true;
					p = dynamic_pointer_cast<Terrain>(RESOURCEMANAGER->getRenderable(pNameSO, ""));
				}
				else {
					p = dynamic_pointer_cast<Terrain>(RESOURCEMANAGER->createRenderable("Terrain", pNameSO));
				}

				if (!alreadyThere) {
					AttribSet *as = NAU->getAttribs("TERRAIN");
					if (as != NULL) {
						std::vector <std::string> excluded;
						excluded.push_back("name"); excluded.push_back("heightMap"); excluded.push_back("material");
						readAttributes(pName, (AttributeValues *)p.get(), *as, excluded, pElementAux);
					}
					if (!File::Exists(File::GetFullPath(ProjectLoader::s_Path, pHeightMap)))
						NAU_THROW("File %s\nScene %s\nTerrain heightmap does not exist", ProjectLoader::s_File.c_str(), pName);
					p->setHeightMap(File::GetFullPath(ProjectLoader::s_Path, pHeightMap));
					p->build();
				}
				go->setRenderable(dynamic_pointer_cast<IRenderable>(p));

				if (!alreadyThere) {
					if (pMaterial) {
						if (!MATERIALLIBMANAGER->hasMaterial(DEFAULTMATERIALLIBNAME, pMaterial)) {
							MATERIALLIBMANAGER->createMaterial(pMaterial);
						}
						go->setMaterial(pMaterial);
					}
				}

				std::vector<std::string> excluded;
				readChildTags(pNameSO, (AttributeValues *)go.get(), SceneObject::Attribs, excluded, pElementAux);

				is->add(dynamic_pointer_cast<SceneObject>(go));
			}

			pElementAux = handle.FirstChild("buffers").Element();
			std::string primString;
			for (; 0 != pElementAux; pElementAux = pElementAux->NextSiblingElement("buffers")) {

				pElementAux->QueryStringAttribute("primitive", &primString);
				const char *pMaterial = pElementAux->Attribute("material");
				const char *pNameSO = pElementAux->Attribute("name");

				if (!pNameSO) {
					NAU_THROW("File %s\nScene: %s\nBuffer scene object without a name", ProjectLoader::s_File.c_str(), pName);
				}

				if (!RENDERER->primitiveTypeSupport(primString)) {
					NAU_THROW("File %s\nScene: %s\nInvalid primitive type %s in buffers definition", ProjectLoader::s_File.c_str(), pName, primString.c_str());
				}
				IRenderable::DrawPrimitive dp = IRenderer::PrimitiveTypes[primString];
				std::shared_ptr<SceneObject> &so = SceneObjectFactory::Create("SimpleObject");
				so->setName(pNameSO);
				std::shared_ptr<IRenderable> &i = RESOURCEMANAGER->createRenderable("Mesh", pNameSO);
				i->setDrawingPrimitive(dp);
				std::shared_ptr<MaterialGroup> mg;
				if (pMaterial)
					mg = MaterialGroup::Create(i.get(), pMaterial);
				else
					mg = MaterialGroup::Create(i.get(), "dirLightDifAmbPix");

				std::shared_ptr<VertexData> &v = i->getVertexData();
				std::string bufferName;
				TiXmlElement *p = pElementAux->FirstChildElement();
				while (p) {

					int res = readItemFromLib(p, "name", &bufferName);
					if (res != OK)
						NAU_THROW("File %s\nScene: %s\nBuffer has no name", ProjectLoader::s_File.c_str(), pName);

					IBuffer * b;
					b = RESOURCEMANAGER->createBuffer(bufferName);
					int attribIndex = VertexData::GetAttribIndex(std::string(p->Value()));

					if (attribIndex != VertexData::MaxAttribs) {
						v->setBuffer(attribIndex, b->getPropi(IBuffer::ID));
					}
					else if (p->Value() == "index"){

						mg->getIndexData()->setBuffer(b->getPropi(IBuffer::ID));
					}
					else {
						NAU_THROW("File %s\nScene: %s\nVertex Attribute %s is not valid", ProjectLoader::s_File.c_str(), pName, p->Value());
					}
					p = p->NextSiblingElement();
				}

				i->addMaterialGroup(mg);
				so->setRenderable(i);
				is->add(so);
			}

			pElementAux = handle.FirstChild("file").Element();
			for (; 0 != pElementAux; pElementAux = pElementAux->NextSiblingElement("file")) {

				const char *pFileName = pElementAux->Attribute("name");
				if (!pFileName)
					NAU_THROW("File %s\nScene: %s\nFile is not specified", ProjectLoader::s_File.c_str(), pName);

				std::string fullName = File::GetFullPath(ProjectLoader::s_Path, pFileName);

				if (!File::Exists(fullName)) {
					NAU_THROW("File %s\nScene: %s\nFile %s does not exist", ProjectLoader::s_File.c_str(), pName, pFileName);
				}
				NAU->loadAsset(fullName, pName, s);
			}

			pElementAux = handle.FirstChild("folder").Element();
			for (; 0 != pElementAux; pElementAux = pElementAux->NextSiblingElement("folder")) {

				DIR *dir;
				struct dirent *ent;

				const char * pDirName = pElementAux->Attribute("name");
				dir = opendir(File::GetFullPath(ProjectLoader::s_Path, pDirName).c_str());

				if (!dir)
					NAU_THROW("File %s\nScene: %s\nFolder %s does not exist", ProjectLoader::s_File.c_str(), pName, pDirName);

				if (0 != dir) {

					int count = 0;
					while ((ent = readdir(dir)) != 0) {
						char file[1024];

#ifdef NAU_PLATFORM_WIN32
						sprintf(file, "%s\\%s", (char *)File::GetFullPath(ProjectLoader::s_Path, pDirName).c_str(), ent->d_name);
#else
						sprintf (file, "%s/%s", pDirName, ent->d_name);						
#endif
						try {
							NAU->loadAsset(file, pName, s);
						}
						catch (std::string &s) {
							closedir(dir);
							throw(s);
						}
						++count;
					}
					if (count < 3)
						NAU_THROW("File %s\nScene: %s\nFolder %s is empty", ProjectLoader::s_File.c_str(), pName, pDirName);

					closedir(dir);
				}
			}
			std::vector<std::string> excluded = {"file", "folder", "geometry", "terrain", "buffers"};
//			excluded.push_back("file"); excluded.push_back("folder");
//			excluded.push_back("geometry"); excluded.push_back("buffers");
			readChildTags(pName, (AttributeValues *)is.get(), IScene::Attribs, excluded, pElem);
		}
		if (pParam) {
			std::string params = std::string(pParam);
			if (params.find("UNITIZE") != std::string::npos)
				is->unitize();
		}
	}
}

/* ----------------------------------------------------------------
Specification of the atomic semantics:

		<atomics>
			<atomic buffer = "name" offset=0 semantics="Red Pixels"/>
			...
		</atomics>

Each atomic must have an id and a name.
----------------------------------------------------------------- */

void
ProjectLoader::loadAtomicSemantics(TiXmlHandle handle) 
{
	TiXmlElement *pElem;

	pElem = handle.FirstChild ("atomics").FirstChild ("atomic").Element();
	for (; 0 != pElem; pElem = pElem->NextSiblingElement("atomic")) {
		std::vector<std::string> ok = { "semantics", "offset", "buffer", "fromLibrary" };
		checkForNonValidAttributes("Atomics", ok, pElem);
		const char *pName = pElem->Attribute ("semantics");
		int offset;
		std::string buffer;
		int res = readItemFromLib(pElem, "buffer", &buffer);
		//const char *pBuffer = pElem->Attribute("buffer");

		if (res != OK) {
			NAU_THROW("File %s\nAtomic has no buffer name", ProjectLoader::s_File.c_str());
		}

		int failure = pElem->QueryIntAttribute("offset", &offset);
		if (failure) {
			NAU_THROW("File %s\nAtomic has no offset", ProjectLoader::s_File.c_str());
		}

		if (0 == pName) {
			NAU_THROW("File %s\nAtomic from buffer %s with offset %d has no semantics", ProjectLoader::s_File.c_str(), buffer.c_str(), offset);
		}

		SLOG("Atomic : %s %d %s", buffer.c_str(), offset, pName);
		RENDERER->addAtomic(buffer, offset, pName);
	} 
}


/* ----------------------------------------------------------------
Specification of the viewport:

		<viewports>
			<viewport name="MainViewport">
				<CLEAR_COLOR r = 0.0 g = 0.0 b = 0.3 />
				<ORIGIN x = 0.66 y = 0 />
				<SIZE width= 0.33 height = 1.0 />
			</viewport>
			...
		</viewports>

		or

		<viewports>
			<viewport name="MainViewport">
				<CLEAR_COLOR r = 0.0 g = 0.0 b = 0.3 />
				<ORIGIN x = 0.66 y = 0 />
				<SIZE width=0.33 height = 1.0 />
				<RATIO value=0.5 />
			</viewport>
			...
		</viewports>

CLEAR_COLOR is optional, if not specified it will be black

geometry can be relative or absolute, if values are smaller than, or equal to 1
it is assumed to be relative.

geometry is specified with ORIGIN and SIZE

ratio can be used instead of height, in which case height = absolute_width*ratio
----------------------------------------------------------------- */

void
ProjectLoader::loadViewports(TiXmlHandle handle) 
{
	TiXmlElement *pElem;
	std::shared_ptr<Viewport> v;

	pElem = handle.FirstChild ("viewports").FirstChild ("viewport").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
		const char *pName = pElem->Attribute ("name");

		if (0 == pName) {
			NAU_THROW("File %s\nViewport has no name", ProjectLoader::s_File.c_str());
		}
		if (RENDERMANAGER->hasViewport(pName)) {
			NAU_THROW("File %s\nViewport %s is already defined", ProjectLoader::s_File.c_str(), pName);
		}

		SLOG("Viewport : %s", pName);
		v = RENDERMANAGER->createViewport(pName, vec4(0.0f, 0.0f, 0.0f, 1.0f));

		// Reading remaining viewport attributes
		std::vector<std::string> excluded;
		readChildTags(pName, (AttributeValues *)v.get(), Viewport::Attribs, excluded, pElem);

	} //End of Viewports
}


/* ----------------------------------------------------------------
Specification of the cameras:

		<cameras>
			<camera name="MainCamera">
				<viewport name="MainViewport" />
				<projection TYPE="PERSPECTIVE" FOV="60.0" NEAR="1.0" FAR="1000.0">
				<projection TYPE="ORTHO" LEFT="-1.0" RIGHT="1.0" BOTTOM="-1.0" TOP="1.0" NEAR="-30.0" FAR="10.0" />
				<POSITION x="-240.0" y="180.0" z="-330" />
				<VIEW x="0.54" y="-0.37" z="0.75" />
				<UP x="0.0" y="1.0" z="0.0" />
			</camera>
			...
		</cameras>

type is optional and can be either "perspective" or "ortho", 
	if not specified it will be "perspective"

viewport is optional, if specified indicats the name of a previously defined viewport
	otherwise the default viewport ( a full screen viewport) will be used.

ortho and perspective, depending on the type of the camera, one of them must be defined.
position is optional, if not specified it will be (0.0 0.0, 0.0)
view is optional, if not specified it will be (0.0, 0.0, -1.0)
up is optional, if not defined it will be (0.0, 1.0, 0.0)
----------------------------------------------------------------- */

void
ProjectLoader::loadCameras(TiXmlHandle handle) 
{
	TiXmlElement *pElem;

	std::vector<std::string> ok = {"camera"};
	checkForNonValidChildTags("cameras", ok, handle.FirstChild("cameras").Element());
	pElem = handle.FirstChild ("cameras").FirstChild ("camera").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement("camera")) {
		const char *pName = pElem->Attribute ("name");

		if (0 == pName) {
			NAU_THROW("File %s\nCamera has no name", ProjectLoader::s_File.c_str());
		}

		SLOG("Camera: %s", pName);

		if (RENDERMANAGER->hasCamera(pName))
			NAU_THROW("File %s\nCamera %s is already defined", ProjectLoader::s_File.c_str(), pName);

		std::shared_ptr<Camera> &aNewCam = RENDERMANAGER->getCamera (pName);

		TiXmlElement *pElemAux = 0;
		std::string s;

		// Read Viewport
		pElemAux = pElem->FirstChildElement ("viewport");
		std::shared_ptr<Viewport> v;
		if (0 == pElemAux) {
			v = NAU->getDefaultViewport ();
		} else {
			if (TIXML_SUCCESS != pElemAux->QueryStringAttribute("name", &s))
				NAU_THROW("File %s\nElement %s\nviewport name is required", ProjectLoader::s_File.c_str(), pName);

			// Check if previously defined
			v = RENDERMANAGER->getViewport(s);
			if (!v)
				NAU_THROW("File %s\nElement %s\nviewport %s is not previously defined", ProjectLoader::s_File.c_str(), pName, s.c_str());

			aNewCam->setViewport (v);
		}
		// read projection values
		pElemAux = pElem->FirstChildElement("projection");

		if (pElemAux != NULL)
		{
			std::vector<std::string> excluded;
			readAttributes(std::string(pName), (AttributeValues *)aNewCam.get(), Camera::Attribs, excluded, pElemAux);
		}
		// Reading remaining camera attributes

		std::vector<std::string> excluded;
		excluded.push_back("projection"); excluded.push_back("viewport");
		readChildTags(pName, (AttributeValues *)aNewCam.get(), Camera::Attribs, excluded, pElem);
	} //End of Cameras
}


/* ----------------------------------------------------------------
Specification of the lights:

		<lights>
			<light name="Sun">
				<POSITION x="0.0" y="0.0" z="0.0" />
				<DIRECTION x="0.0" y="0.0" z="-1.0" />
				<COLOR r="1.0" g="1.0" b="1.0" />
				<AMBIENT r="0.2", g="0.2", b="0.2" />
			</light>
			...
		</lights>

position is optional, if not specified it will be (0.0 0.0, .0.)
direction is optional, if not specified it will be (0.0, 0.0, -1.0)
color is optional, if not defined it will be (1.0, 1.0, 1.0)
	TODO: must add ambient color
----------------------------------------------------------------- */
void 
ProjectLoader::loadLights(TiXmlHandle handle) 
{
	TiXmlElement *pElem;

	std::vector<std::string> ok = {"light"};
	checkForNonValidChildTags("lights", ok, handle.FirstChild("lights").Element());

	pElem = handle.FirstChild ("lights").FirstChild ("light").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement("light")) {
		const char *pName = pElem->Attribute ("name");
		const char *pClass = pElem->Attribute("class"); 
		
		if (0 == pName) 
			NAU_THROW("File %s\nLight has no name", ProjectLoader::s_File.c_str());

		SLOG("Light: %s", pName);


		if (RENDERMANAGER->hasLight(pName))
			NAU_THROW("File %s\nLight %s is already defined", ProjectLoader::s_File.c_str(), pName);

		std::vector<std::string> excluded;
		std::shared_ptr<Light> l;
		if (0 == pClass) {
			l = RENDERMANAGER->getLight(pName);
		}
		else {
			l = RENDERMANAGER->createLight(pName, pClass);
		}
			readChildTags(pName, (AttributeValues *)l.get(), Light::Attribs, excluded, pElem);
		
		// Reading Light Attributes

	}//End of lights
}


/* ----------------------------------------------------------------
Specification of the assets:

	<assets>
		<constants>
			...
		<constants>
		<attributes>
			...
		</attributes>
		<scenes>
			...
		</scenes>
		<viewports>
			...
		</viewports>
		<cameras>
			...
		</cameras>
		<lights>
			...
		</lights>
		<atomics>
			...
		</atomics>
		<materiallibs>
			<mlib filename="..\mlibs\vision.mlib"/>
			<mlib filename="..\mlibs\quadMaterials.mlib"/>
		</materiallibs>
	</assets>


----------------------------------------------------------------- */


void
ProjectLoader::loadAssets (TiXmlHandle &hRoot, std::vector<std::string>  &matLibs)
{
	TiXmlElement *pElem;
	TiXmlHandle handle (hRoot.FirstChild ("assets").Element());


	std::vector<std::string> ok = {"constants", "attributes", "scenes", "viewports", "cameras",
		"lights", "events", "atomics", "materialLibs", "sensors", "routes", "interpolators"};
	checkForNonValidChildTags("Assets", ok, hRoot.FirstChild ("assets").Element());
	loadConstants(handle);
	loadUserAttrs(handle);
	loadScenes(handle);
	loadViewports(handle);
	loadCameras(handle);
	loadLights(handle);	
	loadEvents(handle);
	if (APISupport->apiSupport(IAPISupport::BUFFER_ATOMICS))
		loadAtomicSemantics(handle);

	pElem = handle.FirstChild ("materialLibs").FirstChild ("materialLib").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
		const char *pFilename = pElem->Attribute ("filename");

		if (0 == pFilename) {
			NAU_THROW("File %s\nNo file specified for material lib", ProjectLoader::s_File.c_str());
		}

		try {
			SLOG("Loading Material Lib from file : %s", File::GetFullPath(ProjectLoader::s_Path,pFilename).c_str());
			loadMatLib(File::GetFullPath(ProjectLoader::s_Path,pFilename));
		}
		catch(std::string &s) {
			throw(s);
		}
	}
}


/*-------------------------------------------------------------*/
/*                   PASS        ELEMENTS                      */
/*-------------------------------------------------------------*/

/* -----------------------------------------------------------------------------
CAMERAS

	<camera name="MainCamera">

Specifies a previously defined camera (in the assets part of the file)
-----------------------------------------------------------------------------*/

void 
ProjectLoader::loadPassCamera(TiXmlHandle hPass, Pass *aPass) 
{
	TiXmlElement *pElem;

	pElem = hPass.FirstChild ("camera").Element();
	if (0 == pElem && aPass->getClassName() != "compute" && aPass->getClassName() != "quad") {
		NAU_THROW("File %s\nPass %s\nNo camera element found", ProjectLoader::s_File.c_str(), aPass->getName().c_str());
	}
	else if (pElem != 0) {
		const char *pCam = pElem->Attribute("name");
		if (pCam) {
			if (!RENDERMANAGER->hasCamera(pCam))
				NAU_THROW("File %s\nPass %s\nCamera %s is not defined", ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pCam);

			aPass->setCamera(pCam);
		}
	}
}


/* -----------------------------------------------------------------------------
MODE

<mode value="RUN_ALWAYS" />

Specifies the run mode. Valid values are defined in class Pass (file pass.h)
-----------------------------------------------------------------------------*/

void
ProjectLoader::loadPassMode(TiXmlHandle hPass, Pass *aPass)
{
	TiXmlElement *pElem;

	pElem = hPass.FirstChild("mode").Element();
	if (pElem != 0) {
		const char *pMode = pElem->Attribute("value");
		bool valid = Pass::Attribs.isValid("RUN_MODE", pMode);
		if (!valid) {
			std::unique_ptr<Attribute> &a = aPass->getAttribSet()->get("RUN_MODE");
			std::string s = getValidValuesString(a);
			if (s != "") {
				NAU_THROW("File %s\nElement %s: \"%s\" has an invalid value\nValid values are\n%s", ProjectLoader::s_File.c_str(), aPass->getName().c_str(), a->getName().c_str(), s.c_str());
			}
			else {
				NAU_THROW("File %s\nElement %s: \"%s\" is not supported", ProjectLoader::s_File.c_str(), aPass->getName().c_str(), a->getName().c_str());
			}
		}

		aPass->setMode((Pass::RunMode)Pass::Attribs.getListValueOp(Pass::RUN_MODE, pMode));
	}
}


/* -----------------------------------------------------------------------------
testScript

<testScript file="test.lua" script="testFunction" />


Specifies a test script for the pass. The pass will only execute if the test returns true

<preScript file="test.lua" script="function1" />
<preScript file="test.lua" script="function2" />

Specifies scripts to be executed before and after the pass
-----------------------------------------------------------------------------*/

void
ProjectLoader::loadPassScripts(TiXmlHandle hPass, Pass *aPass)
{
	TiXmlElement *pElem;

	pElem = hPass.FirstChild("testScript").Element();
	if (pElem != 0) {
		const char *pFile = pElem->Attribute("file");
		const char *pFunction = pElem->Attribute("script");
		if (!pFile || !pFunction) {
			NAU_THROW("File %s\nPass %s\nElement: testScript\nBoth file and script fields are required", ProjectLoader::s_File.c_str(), aPass->getName().c_str());
		}
		std::unique_ptr<Attribute> &a = aPass->getAttribSet()->get("TEST_MODE");

		Data *val = readAttribute("TEST_MODE", a, pElem);
		if (val != NULL) {
			int value = dynamic_cast<NauInt *>(val)->getNumber();
			if (a->isValid(value))
				aPass->setProp(Pass::TEST_MODE, Enums::ENUM, val);
			else {
				std::string s = getValidValuesString(a);
				NAU_THROW("File %s\nElement testScript: \"%s\" has an invalid value\nValid values are\n%s", ProjectLoader::s_File.c_str(), a->getName().c_str(), s.c_str());

			}
			delete val;
		}

		aPass->setTestScript(File::GetFullPath(ProjectLoader::s_Path, pFile), pFunction);
	}

	pElem = hPass.FirstChild("preScript").Element();
	if (pElem) {
		const char *pFile = pElem->Attribute("file");
		const char *pFunction = pElem->Attribute("script");
		if (!pFile || !pFunction) {
			NAU_THROW("File %s\nPass %s\nElement: preScript\nBoth file and script fields are required", ProjectLoader::s_File.c_str(), aPass->getName().c_str());
		}
		aPass->setPreScript(File::GetFullPath(ProjectLoader::s_Path, pFile), pFunction);
	}

	pElem = hPass.FirstChild("postScript").Element();
	if (pElem) {
		const char *pFile = pElem->Attribute("file");
		const char *pFunction = pElem->Attribute("script");
		if (!pFile || !pFunction) {
			NAU_THROW("File %s\nPass %s\nElement: postScript\nBoth file and script fields are required", ProjectLoader::s_File.c_str(), aPass->getName().c_str());
		}
		aPass->setPostScript(File::GetFullPath(ProjectLoader::s_Path, pFile), pFunction);
	}
}


/* -----------------------------------------------------------------------------
LIGHTS

	<lights>
		<light name="Sun" />
		...
	</lights>

Specifies a previously defined light (in the assets part of the file)
-----------------------------------------------------------------------------*/

void 
ProjectLoader::loadPassLights(TiXmlHandle hPass, Pass *aPass) 
{
	TiXmlElement *pElem;

	pElem = hPass.FirstChild ("lights").FirstChild ("light").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
		const char *pName = pElem->Attribute("name");

		if (0 == pName) {
			NAU_THROW("File %s\nPass %s\nLight has no name in pass", ProjectLoader::s_File.c_str(), aPass->getName().c_str());
		}
		if (!RENDERMANAGER->hasLight(pName))
			NAU_THROW("File %s\nPass %s\nLight %s is not defined", ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pName);

		aPass->addLight (pName);
	}//End of lights
}



/* -----------------------------------------------------------------------------
SCENES

	<scenes>
		<scene name="MainScene" />
		...
	</scenes>

Specifies a previously defined scene (in the assets part of the file)
-----------------------------------------------------------------------------*/



void 
ProjectLoader::loadPassScenes(TiXmlHandle hPass, Pass *aPass) 
{
	TiXmlElement *pElem;

	pElem = hPass.FirstChild("scenes").Element();
	if (pElem == NULL)
		return;

	AttribSet *attrs = aPass->getAttribSet();
	std::unique_ptr<Attribute> &attr = attrs->get(Pass::INSTANCE_COUNT, Enums::UINT);
	Data *d = readAttribute("instances", attr, pElem);
	if (d != NULL) {
		unsigned int ui = dynamic_cast<NauUInt *>(d)->getNumber();
		aPass->setPropui(Pass::INSTANCE_COUNT, ui);
		delete d;
	}
	const char* pDrawIndirect = pElem->Attribute("drawIndirectBuffer");
	if (pDrawIndirect != NULL)
		aPass->setBufferDrawIndirect(pDrawIndirect);

	std::vector<std::string> ok = { "instances", "drawIndirectBuffer" };
	checkForNonValidAttributes("Pass:Scenes", ok, pElem);

	pElem = hPass.FirstChild ("scenes").FirstChild ("scene").Element();
	if (0 == pElem && aPass->getClassName() != "compute" && aPass->getClassName() != "quad") {
		NAU_THROW("File %s\nPass %s\nNo Scene element found", ProjectLoader::s_File.c_str(), aPass->getName().c_str());
	}
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {

		const char *pName = pElem->Attribute("name");

		if (pName && !RENDERMANAGER->hasScene(pName)) {
			NAU_THROW("File %s\nPass %s\nScene %s is not defined", ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pName);
		}
		else if (!pName) {
			NAU_THROW("File %s\nPass %s\nScene has no name attribute", ProjectLoader::s_File.c_str(), aPass->getName().c_str());
		}

		aPass->addScene (pName);
	} //End of scenes
}


/* -----------------------------------------------------------------------------
CLEAR DEPTH AND COLOR

	<color clear=true enable=true/>
	<depth clear=true clearValue=1.0 test=true write=true/>

	<stencil clear=true clearValue=0 test=true>
		<stencilFunc func=ALWAYS ref=1 mask=255/>
		<stencilOp sfail=KEEP dfail=KEEP dpass=KEEP />
	</stencil>

By default these fields will be true and can be omitted.
Clear color is the one from the viewport
-----------------------------------------------------------------------------*/

//void 
//ProjectLoader::loadPassClearDepthAndColor(TiXmlHandle hPass, Pass *aPass)
//{
//	TiXmlElement *pElem;
//	bool *b = new bool();
//	std::vector<std::string> included = { "DEPTH_ENABLE" , "DEPTH_CLEAR", "DEPTH_MASK", "DEPTH_CLEAR_VALUE",
//		"STENCIL_ENABLE", "STENCIL_CLEAR", "STENCIL_CLEAR_VALUE", 
//		"STENCIL_FUNC", "STENCIL_OP_REF", "STENCIL_OP_MASK", 
//		"STENCIL_FAIL", "STENCIL_DEPTH_FAIL", "STENCIL_DEPTH_PASS",
//		"COLOR_CLEAR", "COLOR_ENABLE"
//	};
//	std::map<std::string, std::unique_ptr<Attribute>> fullList = Pass::Attribs.getAttributes();
//
//	
//	// Clear Color and Depth
//	pElem = hPass.FirstChild ("depth").Element();
//	if (0 != pElem) {
//
//		std::string s = "Pass " + aPass->getName() + ": Element depth";
//		readAttributeList(s, (AttributeValues *)aPass, fullList, Pass::Attribs, excluded, pElem);
//	}
//
//	pElem = hPass.FirstChild ("stencil").Element();
//	if (0 != pElem) {
//		std::map<std::string, Attribute> attrList;
//		std::map<std::string, std::unique_ptr<Attribute>> fullList = Pass::Attribs.getAttributes();
//		attrList["test"] = fullList["STENCIL_ENABLE"];
//		attrList["clear"] = fullList["STENCIL_CLEAR"];
//		attrList["clearValue"] = fullList["STENCIL_CLEAR_VALUE"];
//		std::string s = "Pass " + aPass->getName() + ": Element stencil";
//		readAttributeList(s, (AttributeValues *)aPass, attrList, Pass::Attribs, excluded, pElem);
//
//		TiXmlElement *pElemAux = pElem->FirstChildElement("stencilFunc");
//		if (pElemAux != NULL) {
//			std::map<std::string, Attribute> attrList;
//			std::map<std::string, Attribute> fullList = Pass::Attribs.getAttributes();
//			attrList["func"] = fullList["STENCIL_FUNC"];
//			attrList["ref"] = fullList["STENCIL_OP_REF"];
//			attrList["mask"] = fullList["STENCIL_OP_MASK"];
//			std::string s = "Pass " + aPass->getName() + ": Element stencilFunc";
//			readAttributeList(s, (AttributeValues *)aPass, attrList, Pass::Attribs, excluded, pElemAux);
//		}
//		pElemAux = pElem->FirstChildElement("stencilOp");
//		if (pElemAux != NULL) {
//			std::map<std::string, Attribute> attrList;
//			std::map<std::string, Attribute> fullList = Pass::Attribs.getAttributes();
//			attrList["sfail"] = fullList["STENCIL_FAIL"];
//			attrList["dfail"] = fullList["STENCIL_DEPTH_FAIL"];
//			attrList["dpass"] = fullList["STENCIL_DEPTH_PASS"];
//			std::string s = "Pass " + aPass->getName() + ": Element stencilOp";
//			readAttributeList(s, (AttributeValues *)aPass, attrList, Pass::Attribs, excluded, pElemAux);
//
//		}
//	}
//	pElem = hPass.FirstChild ("color").Element();
//	if (0 != pElem) {
//		std::map<std::string, Attribute> attrList;
//		std::map<std::string, Attribute> fullList = Pass::Attribs.getAttributes();
//		attrList["clear"] = fullList["COLOR_CLEAR"];
//		attrList["enable"] = fullList["COLOR_ENABLE"];
//		std::string s = "Pass " + aPass->getName() + ": Element color";
//		readAttributeList("color", (AttributeValues *)aPass, attrList, Pass::Attribs, excluded, pElem);
//	}
//}

/* -----------------------------------------------------------------------------
VIEWPORTS

	<viewport>SmallViewport2</viewport>				

If a viewport is defined it will replace the viewport of the passe's camera
-----------------------------------------------------------------------------*/

void 
ProjectLoader::loadPassViewport(TiXmlHandle hPass, Pass *aPass)
{
	TiXmlElement *pElem;


	pElem = hPass.FirstChild ("viewport").Element();
	if (0 != pElem) {
		const char *pViewport = pElem->Attribute("name");
		if (pViewport) {
			std::shared_ptr<Viewport> vp = RENDERMANAGER->getViewport(pViewport);
			if (!vp) {
				NAU_THROW("File %s\nPass %s\nViewport %s is not defined", ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pViewport);
			}
			else
				aPass->setViewport(vp);
		}
	}
}
		
	
/* -----------------------------------------------------------------------------
TEXTURE - Used in quad pass

	<texture name="bla" fromLibrary="bli" />				

Should be an existing texture which will be displayed in the quad, 
usually it is a render target
-----------------------------------------------------------------------------*/

void 
ProjectLoader::loadPassTexture(TiXmlHandle hPass, Pass *aPass)
{
	TiXmlElement *pElem;


	pElem = hPass.FirstChild ("texture").Element();
	if (0 != pElem) {
		//const char *pName = pElem->Attribute ("name");
		//const char *pLib = pElem->Attribute("fromLibrary");
		//
		//if (!pName )
		//	NAU_THROW("File %s\nPass %s\nTexture without name", ProjectLoader::s_File.c_str(), aPass->getName().c_str());
		//if (!pLib) 
		//	sprintf(s_pFullName, "%s", pName);
		//else
		//	sprintf(s_pFullName, "%s::%s", pLib, pName);

		std::string fullName;
		if (ITEM_NAME_NOT_SPECIFIED == readItemFromLib(pElem, "name", &fullName))
			NAU_THROW("File %s\nPass %s\nTexture without name", ProjectLoader::s_File.c_str(), aPass->getName().c_str());

		if (!RESOURCEMANAGER->hasTexture(fullName))
			NAU_THROW("File %s\nPass %s\nTexture %s is not defined", ProjectLoader::s_File.c_str(), aPass->getName().c_str(), fullName.c_str());

		std::shared_ptr<Material> &srcMat = MATERIALLIBMANAGER->getMaterialFromDefaultLib("__Quad");
		std::shared_ptr<Material> &dstMat = MATERIALLIBMANAGER->cloneMaterial(srcMat);
		dstMat->attachTexture(0,fullName);
		MATERIALLIBMANAGER->addMaterial(aPass->getName(),dstMat);
		aPass->remapMaterial ("__Quad", aPass->getName(), "__Quad");
	}
}


/* -----------------------------------------------------------------------------
MATERIAL - Used in quad pass

	<material name="bla" fromLibrary="bli" />				

Should be an existing material which will be used to render the quad
-----------------------------------------------------------------------------*/

void 
ProjectLoader::loadPassMaterial(TiXmlHandle hPass, Pass *aPass)
{
	TiXmlElement *pElem;


	pElem = hPass.FirstChild ("material").Element();
	if (0 != pElem) {
		const char *pName = pElem->Attribute ("name");
		const char *pLib = pElem->Attribute("fromLibrary");
		
		if (!pName )
			NAU_THROW("File %s\nPass %s\nMaterial without name", ProjectLoader::s_File.c_str(), aPass->getName().c_str());
		if (!pLib) 
			NAU_THROW("File %s\nPass %s\nMissing \'fromLibrary' for material %s", ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pName);

		sprintf(s_pFullName, "%s::%s", pLib, pName);

		if (!MATERIALLIBMANAGER->hasMaterial(pLib, pName))
				NAU_THROW("File %s\nPass %s\nMaterial %s is not defined", ProjectLoader::s_File.c_str(), aPass->getName().c_str(), s_pFullName);

		nau::render::PassQuad *p = (PassQuad *)aPass;
		std::string lib = std::string(pLib);
		std::string name = std::string(pName);
		p->setMaterialName(lib, name);
	}
}
/* -----------------------------------------------------------------------------
PARAMS

	<userAttribute  value = 2 />

Passes can take user attributes
-----------------------------------------------------------------------------*/

void 
ProjectLoader::loadPassParams(TiXmlHandle hPass, Pass *aPass)
{
	std::vector<std::string> excluded = {"testScript", "preProcess", "postProcess", "mode", "scene", "scenes", 
		"camera", "lights", "viewport", "renderTarget",
		"materialMaps", "injectionMaps", "texture", "material", "rays", "hits", "rayCount",
		"optixEntryPoint", "optixDefaultMaterial", "optixMaterialMap", "optixInput", "optixVertexAttributes",
		"optixGeometryProgram", "optixOutput", "optixMaterialAttributes", "optixGlobalAttributes", "preScript", "postScript"};
	readChildTags(aPass->getName(), (AttributeValues *)aPass, Pass::Attribs, excluded, hPass.Element(),false);
}


/* -----------------------------------------------------------------------------
RENDERTARGET

	<rendertarget name = "deferredOuput" fromLibrary="testMaterials" />

	or

	<rendertarget sameas="pass2"/>
-----------------------------------------------------------------------------*/

void 
ProjectLoader::loadPassRenderTargets(TiXmlHandle hPass, Pass *aPass,std::map<std::string, Pass*> passMapper)
{
	TiXmlElement *pElem;


	pElem = hPass.FirstChild ("renderTarget").Element();

	if (0 != pElem) {
		
		const char* pSameAs = pElem->Attribute ("sameas");
		const char *pName = pElem->Attribute("name");
		const char *pLib = pElem->Attribute("fromLibrary");

		if (0 != pSameAs) {
			if (passMapper.count (pSameAs) > 0) {
				aPass->setRenderTarget (passMapper[pSameAs]->getRenderTarget());
			} else {
				NAU_THROW("File %s\nPass %s\nRender Target in pass %s is not defined", ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pSameAs);
			}
		} 
		else if (0 != pName && 0 != pLib) {
	
			sprintf(s_pFullName, "%s::%s", pLib, pName);
			IRenderTarget *rt = RESOURCEMANAGER->getRenderTarget(s_pFullName);
			if (rt != NULL)
				aPass->setRenderTarget(rt);
			else
				NAU_THROW("File %s\nPass %s\nRender Target %s is not defined in material lib %s", ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pName, pLib);
			
		}
		else {
			NAU_THROW("File %s\nPass %s\nRender Target Definition error", ProjectLoader::s_File.c_str(), aPass->getName().c_str());
		}
	}
}


/* ----------------------------------------------------------------------------

	OPTIX SETTINGS

	<optixEntryPoint>
		<optixProgram type="RayGen" file="optix/common.ptx" proc="pinhole_camera"/> 
		<optixProgram type="Exception" file="optix/common.ptx" proc="exception"/> 
	</optixEntryPoint>

	<optixDefaultMaterial>
		<optixProgram type="Closest_Hit" ray="Phong" file="optix/common.ptx" proc="shade"/> 
		<optixProgram type="Miss" ray="Phong" file="optix/common.ptx" proc="background"/> 

		<optixProgram type="Any_Hit" ray="Shadow" file="optix/common.ptx" proc="shadows"/> 
	</optixDefaultMaterial>

	// for closest and any hit rays
	<optixMaterialMap>
		<optixMap to="Vidro">
			<optixProgram type="Any_Hit" ray="Phong" file="optix/common.ptx" proc="keepGoing"/> 
			<optixProgram type="Any_Hit" ray="Shadow" file="optix/common.ptx" proc="keepGoingShadow"/> 
		</optixMap>	
	</optixMaterialMap>

	// for input buffers, as in texture buffers
	<optixInput>
		<buffer var="bla" texture="lib::texname" />
	</optixInput>

	// selects which vertex attributes are fed to optix
	<optixVertexAttributes>
		<attribute name="position"/>
	</optixVertexAttributes>

	// Geometry and bonuding box programs
	<optixGeometryProgram> 
			<optixProgram type="Geometry_Intersect" file="optix/common.ptx" proc="bla"/> 
			<optixProgram type="Bounding_Box" file="optix/common.ptx" proc="bla2"/> 
	</optixGeometryProgram>

	// optix output buffers
	// note: not required for render targets
	<optixOutput>
		<buffer var="dataBuffer" texture="Optix Ray Tracer Render Target::dataBuffer" />
	</optixOutput>

	// For material attributes. Tells optix which attributes to use from the materials
	<optixMaterialAttributes>
		<valueof optixVar="diffuse" type="CURRENT" context="COLOR" component="DIFFUSE" />
		<valueof optixVar="ambient" type="CURRENT" context="COLOR" component="AMBIENT" />
		<valueof uniform="texCount"	type="CURRENT" context="TEXTURE" component="COUNT" />
	</optixMaterialAttributes>

	// For globl attributes, i.e. attributes that remain constant per frame
	<optixGlobalAttributes>
		<valueof optixVar="lightDir" type="LIGHT" context="Sun" component="DIRECTION" />
	</optixGlobalAttributes>


-------------------------------------------------------------------------------*/
#ifdef NAU_OPTIX

void
ProjectLoader::loadPassOptixSettings(TiXmlHandle hPass, Pass *aPass) {

	TiXmlElement *pElem, *pElemAux, *pElemAux2;
	PassOptix *p = (PassOptix *)aPass;

	pElem = hPass.FirstChild("optixEntryPoint").FirstChildElement("optixProgram").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {

		const char *pType = pElem->Attribute ("type");
		const char *pProc = pElem->Attribute ("proc");
		std::string message = "Pass " + aPass->getName() + "\nElement optixEntryPoint";
		std::string file = readFile(pElem, "file", message);
		
		if (!pType || (0 != strcmp(pType, "RayGen") && 0 != strcmp(pType, "Exception")))
			NAU_THROW("File: %s\nPass: %s\nInvalid Optix entry point type\nValid Values are: RayGen and Exception", ProjectLoader::s_File.c_str(), aPass->getName().c_str());

		if (!pProc)
			NAU_THROW("File: %s\nPass: %s\nMissing Optix entry point procedure. Tag	\'proc\' is required", ProjectLoader::s_File.c_str(), aPass->getName().c_str());

		if (!strcmp(pType, "RayGen"))
			p->setOptixEntryPointProcedure(nau::render::optixRender::OptixRenderer::RAY_GEN, file, pProc);
		else 
			p->setOptixEntryPointProcedure(nau::render::optixRender::OptixRenderer::EXCEPTION, file, pProc);
	}
	pElem = hPass.FirstChild("optixDefaultMaterial").FirstChildElement("optixProgram").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {

		const char *pType = pElem->Attribute ("type");
		const char *pProc = pElem->Attribute ("proc");
		const char *pRay  = pElem->Attribute ("ray");
		
		if (!pType || (0 != strcmp(pType, "Closest_Hit") && 0 != strcmp(pType, "Any_Hit")  && 0 != strcmp(pType, "Miss")))
			NAU_THROW("File: %s\nPass: %s\nInvalid Optix Default Material Proc Type", ProjectLoader::s_File.c_str(), aPass->getName().c_str());

		if (!pProc)
			NAU_THROW("File: %s\nPass: %s\nMissing Optix Default Material Proc", ProjectLoader::s_File.c_str(), aPass->getName().c_str());

		if (!pRay)
			NAU_THROW("File: %s\nPass: %s\nMissing Optix Default Material Ray", ProjectLoader::s_File.c_str(), aPass->getName().c_str());
		
		std::string message = "Pass " + aPass->getName() + "\nOptix default material - Procedure " + pProc;
		std::string file = readFile(pElem, "file", message);

		if (!strcmp("Closest_Hit", pType)) 
			p->setDefaultMaterialProc(nau::render::optixRender::OptixMaterialLib::CLOSEST_HIT, pRay, file, pProc);
		else if (!strcmp("Any_Hit", pType)) 
			p->setDefaultMaterialProc(nau::render::optixRender::OptixMaterialLib::ANY_HIT, pRay, file, pProc);
		else if (!strcmp("Miss", pType)) 
			p->setDefaultMaterialProc(nau::render::optixRender::OptixMaterialLib::MISS, pRay, file, pProc);
	}

	pElem = hPass.FirstChild("optixMaterialMap").FirstChildElement("optixMap").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {

		const char *pName = pElem->Attribute("to");

		pElemAux = pElem->FirstChildElement("optixProgram");
		for ( ; 0 != pElemAux; pElemAux = pElemAux->NextSiblingElement()) {

		const char *pType = pElemAux->Attribute ("type");
		const char *pProc = pElemAux->Attribute ("proc");
		const char *pRay  = pElemAux->Attribute ("ray");

		if (!pType || (0 != strcmp(pType, "Closest_Hit") && 0 != strcmp(pType, "Any_Hit")))
			NAU_THROW("File: %s\nPass: %s\nInvalid Optix Material Proc Type", ProjectLoader::s_File.c_str(), aPass->getName().c_str());

		if (!pProc)
			NAU_THROW("File: %s\nPass: %s\nMissing Optix Material Proc", ProjectLoader::s_File.c_str(), aPass->getName().c_str());

		if (!pRay)
			NAU_THROW("File: %s\nPass: %s\nMissing Optix Material Ray", ProjectLoader::s_File.c_str(), aPass->getName().c_str());
		
		std::string message = "Pass " + aPass->getName() + "\nOptix material map - Procedure " + pProc;
		std::string file = readFile(pElemAux, "file", message);

		if (!strcmp("Closest_Hit", pType)) 
			p->setMaterialProc(pName, nau::render::optixRender::OptixMaterialLib::CLOSEST_HIT, pRay, file, pProc);
		else if (!strcmp("Any_Hit", pType)) 
			p->setMaterialProc(pName, nau::render::optixRender::OptixMaterialLib::ANY_HIT, pRay, file, pProc);
		}
	}

	pElem = hPass.FirstChild("optixInput").FirstChildElement("buffer").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {

		const char *pVar = pElem->Attribute ("var");
		const char *pTexture = pElem->Attribute ("texture");
		
		if (!pVar)
			NAU_THROW("File: %s\nPass: %s\nOptix Variable required in Input Definition", ProjectLoader::s_File.c_str(), aPass->getName().c_str());

		if (!pTexture)
			NAU_THROW("File: %s\nPass: %s\nMissing texture in Optix Input Definitiont", ProjectLoader::s_File.c_str(), aPass->getName().c_str());

		if (!RESOURCEMANAGER->hasTexture(pTexture))
				NAU_THROW("File: %s\nPass: %s\nTexture %s is not defined", ProjectLoader::s_File.c_str(), aPass->getName().c_str(),pTexture);

		p->setInputBuffer(pVar, pTexture);
	}

	pElem = hPass.FirstChild("optixOutput").FirstChildElement("buffer").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {

		const char *pVar = pElem->Attribute ("var");
		const char *pTexture = pElem->Attribute ("texture");
		
		if (!pVar)
			NAU_THROW("File: %s\nPass: %s\nOptix Variable required in Input Definition", ProjectLoader::s_File.c_str(), aPass->getName().c_str());

		if (!pTexture)
			NAU_THROW("File: %s\nPass: %s\nMissing texture in Optix Input Definitiont", ProjectLoader::s_File.c_str(), aPass->getName().c_str());

		if (!RESOURCEMANAGER->hasTexture(pTexture))
				NAU_THROW("File: %s\nPass: %s\nTexture %s is not defined", ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pTexture);

		p->setOutputBuffer(pVar, pTexture);
	}

	pElem = hPass.FirstChild("optixGeometryProgram").FirstChildElement("optixProgram").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {

		const char *pType = pElem->Attribute ("type");
		const char *pProc = pElem->Attribute ("proc");
		
		if (!pType || (0 != strcmp(pType, "Geometry_Intersection") && 0 != strcmp(pType, "Bounding_Box")))
			NAU_THROW("File: %s\nPass: %s\nInvalid Optix Geometry Program", ProjectLoader::s_File.c_str(), aPass->getName().c_str());

		if (!pProc)
			NAU_THROW("File: %s\nPass: %s\nMissing Optix Geometry Program Proc", ProjectLoader::s_File.c_str(), aPass->getName().c_str());

		std::string message = "Pass " + aPass->getName() + "\nOptix geometry program";
		std::string file = readFile(pElem, "file", message);

		if (!strcmp(pType, "Geometry_Intersection"))
			p->setGeometryIntersectProc(file, pProc);
		else
			p->setBoundingBoxProc(file, pProc);
	}

	pElem = hPass.FirstChild("optixVertexAttributes").FirstChildElement("attribute").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {

		const char *pType = pElem->Attribute ("name");
		unsigned int vi = VertexData::GetAttribIndex(std::string(pType));
		if (!pType || (VertexData::MaxAttribs == vi ))
			NAU_THROW("File: %s\nPass: %s\nInvalid Optix Vertex Attribute", 
				ProjectLoader::s_File.c_str(), aPass->getName().c_str());
	
		p->addVertexAttribute(vi);
	}

	PassOptix *po = (PassOptix *)aPass;
	pElemAux2 = hPass.FirstChild("optixMaterialAttributes").FirstChildElement("valueof").Element();
	for ( ; 0 != pElemAux2; pElemAux2 = pElemAux2->NextSiblingElement()) {
	

		const char *pUniformName = pElemAux2->Attribute ("optixVar");
		const char *pComponent = pElemAux2->Attribute ("component");
		const char *pContext = pElemAux2->Attribute("context");
		const char *pType = pElemAux2->Attribute("type");
		//const char *pId = pElemAux2->Attribute("id");

		if (0 == pUniformName) {
			NAU_THROW("File: %s\nPass: %s\nNo optix variable name", 
				ProjectLoader::s_File.c_str(), aPass->getName().c_str());
		}
		if (0 == pType) {
			NAU_THROW("File: %s\nPass: %s\nNo type found for optix variable %s", 
				ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pUniformName);
		}
		if (0 == pContext) {
			NAU_THROW("File: %s\nPass: %s\nNo context found for optix variable %s", 
				ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pUniformName);
		}
		if (0 == pComponent) {
			NAU_THROW("File: %s\nPass: %s\nNo component found for optix variable %s", 
				ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pUniformName);
		}
		if (!NAU->validateShaderAttribute(pType, pContext, pComponent))
			NAU_THROW("File: %s\nPass: %s\nOptix variable %s is not valid", 
				ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pUniformName);

		int id = 0;
		if (((strcmp(pContext,"LIGHT") == 0) || (0 == strcmp(pContext,"TEXTURE"))) &&  (0 != strcmp(pComponent,"COUNT"))) {
			if (TIXML_SUCCESS != pElemAux2->QueryIntAttribute ("id", &id))
				NAU_THROW("File: %s\nPass: %s\nNo id found for optix variable %s", 
					ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pUniformName);
			if (id < 0)
				NAU_THROW("File: %s\nPass: %s\nid must be non negative, in optix variable %s", 
					ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pUniformName);
		}
		std::string s(pType);
		if (s == "TEXTURE") {
			if (!RESOURCEMANAGER->hasTexture(pContext)) {
				NAU_THROW("File: %s\nPass: %s\nTexture %s is not defined", 
					ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pContext);
			}
			else
				po->addMaterialAttribute (pUniformName, ProgramValue (pUniformName,pType, pContext, pComponent, id));
		}

		else if (s == "CAMERA") {
			// Must consider that a camera can be defined internally in a pass, example:lightcams
			/*if (!RENDERMANAGER->hasCamera(pContext))
				NAU_THROW("Camera %s is not defined in the project file", pContext);*/
			po->addMaterialAttribute (pUniformName, ProgramValue (pUniformName,pType, pContext, pComponent, id));
		}
		else if (s == "LIGHT") {
			if (!RENDERMANAGER->hasLight(pContext))
				NAU_THROW("File: %s\nPass: %s\nLight %s is not defined in the project file", 
					ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pContext);
			po->addMaterialAttribute (pUniformName, ProgramValue (pUniformName,pType, pContext, pComponent, id));

		}
		else
			po->addMaterialAttribute (pUniformName, ProgramValue (pUniformName,pType, pContext, pComponent, id));

			//sprintf(s_pFullName, "%s(%s,%s)",pType,pContext,pComponent );

			
	}
		pElemAux2 = hPass.FirstChild("optixMaterialAttributes").FirstChildElement("valueof").Element();
	for ( ; 0 != pElemAux2; pElemAux2 = pElemAux2->NextSiblingElement()) {
	

		const char *pUniformName = pElemAux2->Attribute ("optixVar");
		const char *pComponent = pElemAux2->Attribute ("component");
		const char *pContext = pElemAux2->Attribute("context");
		const char *pType = pElemAux2->Attribute("type");
		//const char *pId = pElemAux2->Attribute("id");

		if (0 == pUniformName) {
			NAU_THROW("File: %s\nPass: %s\nNo optix variable name", 
				ProjectLoader::s_File.c_str(), aPass->getName().c_str());
		}
		if (0 == pType) {
			NAU_THROW("File: %s\nPass: %s\nNo type found for optix variable %s", 
				ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pUniformName);
		}
		if (0 == pContext) {
			NAU_THROW("File: %s\nPass: %s\nNo context found for optix variable %s", 
				ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pUniformName);
		}
		if (0 == pComponent) {
			NAU_THROW("File: %s\nPass: %s\nNo component found for optix variable %s", 
				ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pUniformName);
		}
		if (!NAU->validateShaderAttribute(pType, pContext, pComponent))
			NAU_THROW("File: %s\nPass: %s\nOptix variable %s is not valid", 
				ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pUniformName);

		int id = 0;
		if (((strcmp(pContext,"LIGHT") == 0) || (0 == strcmp(pContext,"TEXTURE"))) &&  (0 != strcmp(pComponent,"COUNT"))) {
			if (TIXML_SUCCESS != pElemAux2->QueryIntAttribute ("id", &id))
				NAU_THROW("File: %s\nPass: %s\nNo id found for optix variable %s", 
					ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pUniformName);
			if (id < 0)
				NAU_THROW("File: %s\nPass: %s\nId must be non negative, in optix variable %s", 
					ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pUniformName);
		}
		std::string s(pType);
		if (s == "TEXTURE") {
			if (!RESOURCEMANAGER->hasTexture(pContext)) {
				NAU_THROW("File: %s\nPass: %s\nTexture %s is not defined", 
					ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pContext);
			}
			else
				po->addMaterialAttribute (pUniformName, ProgramValue (pUniformName,pType, pContext, pComponent, id));
		}

		else if (s == "CAMERA") {
			// Must consider that a camera can be defined internally in a pass, example:lightcams
			/*if (!RENDERMANAGER->hasCamera(pContext))
				NAU_THROW("Camera %s is not defined in the project file", pContext);*/
			po->addMaterialAttribute (pUniformName, ProgramValue (pUniformName,pType, pContext, pComponent, id));
		}
		else if (s == "LIGHT") {
			if (!RENDERMANAGER->hasLight(pContext))
				NAU_THROW("File: %s\nPass: %s\nLight %s is not defined in the project file", 
					ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pContext);
			po->addMaterialAttribute (pUniformName, ProgramValue (pUniformName,pType, pContext, pComponent, id));

		}
		else
			po->addMaterialAttribute (pUniformName, ProgramValue (pUniformName,pType, pContext, pComponent, id));

			//sprintf(s_pFullName, "%s(%s,%s)",pType,pContext,pComponent );

			
	}
	pElemAux2 = hPass.FirstChild("optixGlobalAttributes").FirstChildElement("valueof").Element();
	for ( ; 0 != pElemAux2; pElemAux2 = pElemAux2->NextSiblingElement()) {
	

		const char *pUniformName = pElemAux2->Attribute ("optixVar");
		const char *pComponent = pElemAux2->Attribute ("component");
		const char *pContext = pElemAux2->Attribute("context");
		const char *pType = pElemAux2->Attribute("type");
		//const char *pId = pElemAux2->Attribute("id");

		if (0 == pUniformName) {
			NAU_THROW("File: %s\nPass: %s\nNo optix variable name", 
				ProjectLoader::s_File.c_str(), aPass->getName().c_str());
		}
		if (0 == pType) {
			NAU_THROW("File: %s\nPass: %s\nNo type found for optix variable %s", 
				ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pUniformName);
		}
		if (0 == pContext) {
			NAU_THROW("File: %s\nPass: %s\nNo context found for optix variable %s", 
				ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pUniformName);
		}
		if (0 == pComponent) {
			NAU_THROW("File: %s\nPass: %s\nNo component found for optix variable %s", 
				ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pUniformName);
		}
		if (!NAU->validateShaderAttribute(pType, pContext, pComponent))
			NAU_THROW("File: %s\nPass: %s\nOptix variable %s is not valid", 
				ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pUniformName);

		int id = 0;
		if (((strcmp(pContext,"LIGHT") == 0) || (0 == strcmp(pContext,"TEXTURE"))) &&  (0 != strcmp(pComponent,"COUNT"))) {
			if (TIXML_SUCCESS != pElemAux2->QueryIntAttribute ("id", &id))
				NAU_THROW("File: %s\nPass: %s\nNo id found for optix variable %s", 
					ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pUniformName);
			if (id < 0)
				NAU_THROW("File: %s\nPass: %s\nId must be non negative, in optix variable %s", 
					ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pUniformName);
		}
		std::string s(pType);
		if (s == "TEXTURE" && strcmp(pContext, "CURRENT")) {
			if (!RESOURCEMANAGER->hasTexture(pContext)) {
				NAU_THROW("File: %s\nPass: %s\nTexture %s is not defined", 
					ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pContext);
			}
			else
				po->addGlobalAttribute (pUniformName, ProgramValue (pUniformName,pType, pContext, pComponent, id));
		}

		else if (s == "CAMERA" && strcmp(pContext, "CURRENT")) {
			// Must consider that a camera can be defined internally in a pass, example:lightcams
			/*if (!RENDERMANAGER->hasCamera(pContext))
				NAU_THROW("Camera %s is not defined in the project file", pContext);*/
			po->addGlobalAttribute (pUniformName, ProgramValue (pUniformName,pType, pContext, pComponent, id));
		}
		else if (s == "LIGHT" && strcmp(pContext, "CURRENT")) {
			if (!RENDERMANAGER->hasLight(pContext))
				NAU_THROW("File: %s\nPass: %s\nLight %s is not defined in the project file", 
					ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pContext);
			po->addGlobalAttribute (pUniformName, ProgramValue (pUniformName,pType, pContext, pComponent, id));

		}
		else
			po->addGlobalAttribute (pUniformName, ProgramValue (pUniformName,pType, pContext, pComponent, id));

			//sprintf(s_pFullName, "%s(%s,%s)",pType,pContext,pComponent );

			
	}


}

#endif

/* ----------------------------------------------------------------------------

OPTIX PRIME SETTINGS


<pass name="Optix Prime" class="optix prime">
	<scene name="Main">
	<rays buffer="primeShadows::rays" queryType = "CLOSEST"/>
	<hits buffer ="primeShadows::hits" />
</pass>
-------------------------------------------------------------------------------*/
#ifdef NAU_OPTIX 

void
ProjectLoader::loadPassOptixPrimeSettings(TiXmlHandle hPass, Pass *aPass) {


	TiXmlElement *pElem;
	PassOptixPrime *p = (PassOptixPrime *)aPass;

	pElem = hPass.FirstChildElement("scene").Element();
	if (pElem != NULL) {
		const char *pSceneName = pElem->Attribute("name");

		if (pSceneName != NULL) {

			if (!RENDERMANAGER->hasScene(pSceneName)) {
				NAU_THROW("File %s\nPass %s\nScene %s has not been defined", ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pSceneName);
			}
			else
				p->addScene(pSceneName);
		}
	}
	else {
		NAU_THROW("File %s\nPass %s\nScene is not defined", ProjectLoader::s_File.c_str(), aPass->getName().c_str());
	}

	pElem = hPass.FirstChildElement("rays").Element();
	if (pElem != NULL) {
		const char *pQueryType = pElem->Attribute("queryType");

		std::string bufferName;
		int res = readItemFromLib(pElem, "buffer", &bufferName);
		if (res == OK) {

			if (!RESOURCEMANAGER->hasBuffer(bufferName)) {
				NAU_THROW("File %s\nPass %s\nRay buffer %s is not defined", ProjectLoader::s_File.c_str(), aPass->getName().c_str(), bufferName.c_str());
			}
			else
				p->addRayBuffer(RESOURCEMANAGER->getBuffer(bufferName));
		}
		else {
			NAU_THROW("File %s\nPass %s\nRay buffer has no name", ProjectLoader::s_File.c_str(), aPass->getName().c_str());
		}
		if (pQueryType != NULL) {
			if (!p->setQueryType(pQueryType))
				NAU_THROW("File %s\nPass %s\nRay buffer %s: Query Type is invalid\nValidValues are: ANY or CLOSEST", ProjectLoader::s_File.c_str(), aPass->getName().c_str(), bufferName.c_str());

		}
		else {
			NAU_THROW("File %s\nPass %s\nRay buffer %s: Query Type not defined", ProjectLoader::s_File.c_str(), aPass->getName().c_str(), bufferName.c_str());
		}
	}
	else {
		NAU_THROW("File %s\nPass %s\nRay buffer is not defined", 
			ProjectLoader::s_File.c_str(), aPass->getName().c_str());
	}

	pElem = hPass.FirstChildElement("hits").Element();
	if (pElem != NULL) {
		std::string bufferName;
		int res = readItemFromLib(pElem, "buffer", &bufferName);
		if (res == OK) {

			if (!RESOURCEMANAGER->hasBuffer(bufferName)) {
				NAU_THROW("File %s\nPass %s\nHit Buffer %s has not been defined", 
					ProjectLoader::s_File.c_str(), aPass->getName().c_str(), bufferName.c_str());
			}
			else
				p->addHitBuffer(RESOURCEMANAGER->getBuffer(bufferName));
		}
		else {
			NAU_THROW("File %s\nPass %s\nHit buffer has no name", 
				ProjectLoader::s_File.c_str(), aPass->getName().c_str());
		}
	}
	else {
		NAU_THROW("File %s\nPass %s\nHit buffer is not defined", 
			ProjectLoader::s_File.c_str(), aPass->getName().c_str());
	}

	pElem = hPass.FirstChildElement("rayCount").Element();
	if (pElem != NULL) {

		if (readIntAttribute(pElem, "value", &s_Dummy_int))
			p->setPropi(PassOptixPrime::RAY_COUNT, s_Dummy_int);
		else { // read value from buffer
			std::string buff;
			int offset = 0;
			if (TIXML_SUCCESS != pElem->QueryStringAttribute("buffer", &buff)) {
				NAU_THROW("File %s\nPass %s\nrayCount attribute requires 'value' or 'buffer' tags",
					ProjectLoader::s_File.c_str(), aPass->getName().c_str());
			}
			if (!RESOURCEMANAGER->hasBuffer(buff)) {
				NAU_THROW("File %s\nPass %s\nrayCount buffer %s has not been defined",
					ProjectLoader::s_File.c_str(), aPass->getName().c_str(), buff.c_str());

			}
			pElem->QueryIntAttribute("offset", &offset);
			IBuffer *b = RESOURCEMANAGER->getBuffer(buff);
			if ((unsigned int)offset > b->getPropui(IBuffer::SIZE)- 4) {
				NAU_THROW("File %s\nPass %s\noffset %d is greater than buffer size - sizeof(float)",
					ProjectLoader::s_File.c_str(), aPass->getName().c_str(), offset);
			}
			p->setBufferForRayCount(b, offset);

		}
	}

}


#endif
/* ----------------------------------------------------------------------------

	COMPUTE SETTINGS

			<pass class="compute" name="test">
				<material name="computeShader" fromLibrary="myLib" dimX=256, dimY=256, dimZ=0/>
				<!-- or -->
				<material name="computeShader" fromLibrary="myLib" bufferX="tt::aa" offsetX=4 bufferY="tt::aa"* offsetY=0/>
			</pass>

	The strings in the buffer(X,Y,Z) are buffer labels that must be previously defined
	Dims and buffers can be mixed, but for each dimension there must be only one
-------------------------------------------------------------------------------*/



void
ProjectLoader::loadPassComputeSettings(TiXmlHandle hPass, Pass *aPass) {

	TiXmlElement *pElem;
	PassCompute *p = (PassCompute *)aPass;

	pElem = hPass.FirstChildElement("material").Element();
	if (pElem != NULL) {
		const char *pMatName = pElem->Attribute ("name");
		const char *pLibName = pElem->Attribute ("fromLibrary");
		const char *pAtX = pElem->Attribute("bufferX");
		const char *pAtY = pElem->Attribute("bufferY");
		const char *pAtZ = pElem->Attribute("bufferZ");


		if (pMatName != NULL && pLibName != NULL) {
			if (!MATERIALLIBMANAGER->hasMaterial(pLibName,pMatName))
				NAU_THROW("File %s\nPass %s\nMaterial %s::%s is not defined", ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pLibName, pMatName);
		}
		else 
			NAU_THROW("File %s\nPass %s\nMaterial not defined", ProjectLoader::s_File.c_str(), aPass->getName().c_str());

		IBuffer * bX = NULL, *bY = NULL, *bZ = NULL;
		unsigned int offX = 0, offY = 0, offZ = 0;
		unsigned int r1=1, r2=1, r3=1;
		// Read value or buffer id for dimX
		AttribSet *attrs = aPass->getAttribSet();
		std::unique_ptr<Attribute> &attr = attrs->get(PassCompute::DIM_X, Enums::UINT);
		Data *res = readAttribute("dimX", attr, pElem);
		//unsigned int *res = (unsigned int *)readAttribute("dimX", attr, pElem);
		//int res = pElem->QueryIntAttribute("dimX", &dimX);
		if (!res && pAtX == NULL) {
			NAU_THROW("File %s\nPass %s\ndimX or bufferX are not defined", ProjectLoader::s_File.c_str(), aPass->getName().c_str());
		}
		else if (res && pAtX != NULL) {
			NAU_THROW("File %s\nPass %s\ndimX and bufferX are both defined", ProjectLoader::s_File.c_str(), aPass->getName().c_str());
		}
		// 
		if (pAtX != NULL) {
			bX = RESOURCEMANAGER->getBuffer(pAtX);
			if (!bX) {
				NAU_THROW("File %s\nPass %s\nbuffer %s is not defined", ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pAtX);
			}
			else {
				if (TIXML_SUCCESS != pElem->QueryUnsignedAttribute("offsetX", &offX))
					NAU_THROW("File %s\nPass %s\nNo offset defined for buffer %s", ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pAtX);
			}
		}
		if (res) {
			r1 = dynamic_cast<NauUInt *>(res)->getNumber();
			delete res;
		}
		// Read value or buffer id for dimY
		std::unique_ptr<Attribute> &attr2 = attrs->get(PassCompute::DIM_Y, Enums::UINT);
		Data *res2 = readAttribute("dimY", attr, pElem);
		//unsigned int *res2 = (unsigned int *)readAttribute("dimY", attr2, pElem);
		if (res2 && pAtY != NULL) {
			NAU_THROW("File %s\nPass %s\ndimY and bufferY are both defined", ProjectLoader::s_File.c_str(), aPass->getName().c_str());
		}

		//if (!res2)
		//	*res2 = 1;
		if (pAtY != NULL) {
			bY = RESOURCEMANAGER->getBuffer(pAtY);
			if (!bY) {
				NAU_THROW("File %s\nPass %s\nbuffer %s is not defined", ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pAtY);
			}
			else {
				if (TIXML_SUCCESS != pElem->QueryUnsignedAttribute("offsetY", &offY))
					NAU_THROW("File %s\nPass %s\nNo offset defined for buffer %s", ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pAtY);
			}
		}
		if (res2) {
			r2 = dynamic_cast<NauUInt *>(res2)->getNumber();
			delete res2; 
		}
		// Read value or buffer id for dimZ
		std::unique_ptr<Attribute> &attr3 = attrs->get(PassCompute::DIM_Z, Enums::UINT);

		Data *res3 = readAttribute("dimZ", attr, pElem);
		//unsigned int *res3 = (unsigned int *)readAttribute("dimZ", attr3, pElem);
		if (res3 && pAtZ != NULL) {
			NAU_THROW("File %s\nPass %s\ndimZ and bufferZ are both defined", ProjectLoader::s_File.c_str(), aPass->getName().c_str());
		}

		//if (!res3)
		//	*res3 = 1;
		if (pAtZ != NULL) {
			bZ = RESOURCEMANAGER->getBuffer(pAtZ);
			if (!bZ) {
				NAU_THROW("File %s\nPass %s\nbuffer %s is not defined", ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pAtZ);
			}
			else {
				if (TIXML_SUCCESS != pElem->QueryUnsignedAttribute("offsetZ", &offZ))
					NAU_THROW("File %s\nPass %s\nNo offset defined for buffer %s", ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pAtZ);
			}
		}
		if (res3) {
			r3 = dynamic_cast<NauUInt *>(res3)->getNumber();
			delete res3;
		}

		p->setMaterialName (pLibName, pMatName);
		p->setDimension( r1, r2, r3);	
		p->setDimFromBuffer(bX, offX, bY, offY, bZ, offZ);
	}
	else
		NAU_THROW("File %s\nPass %s\nMissing material", ProjectLoader::s_File.c_str(), aPass->getName().c_str());
	
}

/* -----------------------------------------------------------------------------
MATERIAL LIBRAY RENDERTARGET DEFINITION

	<renderTargets>
		<renderTarget name = "test" />
			<SIZE width=2048 height=2048 />
			<SAMPLES value = 16 />
			<LAYERS value = 2/>
			<CLEAR_VALUES r=0.0 g=0.0 b=0.0 />
			<colors>
				<color name="offscreenrender" internalFormat="RGBA32F" />
			</colors>
			<depth name="shadowMap1"  internalFormat="DEPTH_COMPONENT24"  />
				or
			<depthStencil name="bli" />
		</renderTarget>
	</renderTargets>

The names of both color and depth RTs will be available for other passes to use
	as textures

layers is optional, by default regular (non-array) textures are built.
samples is optional. Setting samples to 0 or 1 implies no multisampling

There can be multiple color, but only one depth.
depth and depthStencil can't be both defined in the same render target 
Depth and Color can be omitted, but at least one of them must be present.

-----------------------------------------------------------------------------*/

void 
ProjectLoader::loadMatLibRenderTargets(TiXmlHandle hRoot, MaterialLib *aLib, std::string path)
{
	TiXmlElement *pElem;
	IRenderTarget *rt;
	//int rtWidth, rtHeight, rtSamples = 0, rtLayers = 0; 

	pElem = hRoot.FirstChild ("renderTargets").FirstChild ("renderTarget").Element();
	for ( ; 0 != pElem; pElem=pElem->NextSiblingElement()) {
		const char *pRTName = pElem->Attribute ("name");

		if (!pRTName)
			NAU_THROW("File %s\nLibrary %s\nRender Target has no name", ProjectLoader::s_File.c_str(), aLib->getName().c_str());

		sprintf(s_pFullName, "%s::%s", aLib->getName().c_str(), pRTName);
		rt = RESOURCEMANAGER->createRenderTarget (s_pFullName);
		std::vector<std::string> excluded;
		excluded.push_back("depth"); excluded.push_back("colors");excluded.push_back("depthStencil");
		readChildTags(pRTName, (AttributeValues *)rt, IRenderTarget::Attribs, excluded, pElem);
		//// Render Target size
		//TiXmlElement *pElemSize;
		//pElemSize = pElem->FirstChildElement ("size");
		//	
		//if (0 != pElemSize) {
		//	rtSamples = 0;
		//	pElemSize->QueryIntAttribute("samples", &rtSamples);
		//	
		//	rtLayers = 0;
		//	pElemSize->QueryIntAttribute("layers", &rtLayers);

		//	if (TIXML_SUCCESS != pElemSize->QueryIntAttribute ("width", &rtWidth)) {
		//		NAU_THROW("Library %s: Render Target %s: WIDTH is required", aLib->getName().c_str(), pRTName);
		//	}

		//	if (TIXML_SUCCESS != pElemSize->QueryIntAttribute ("height",  &rtHeight)) {
		//		NAU_THROW("Library %s: Render Target %s: HEIGHT is required", aLib->getName().c_str(), pRTName);
		//	}
		//	sprintf(s_pFullName, "%s::%s", aLib->getName().c_str(), pRTName);
		//	m_RT = RESOURCEMANAGER->createRenderTarget (s_pFullName);	
		//	m_RT->setPropui(IRenderTarget::SAMPLES, rtSamples);
		//	m_RT->setPropui(IRenderTarget::LAYERS, rtLayers);
		//	m_RT->setPropui2(IRenderTarget::SIZE, uivec2(rtWidth, rtHeight));
		//} 
		//else {
		//	NAU_THROW("Library %s: Render Target %s: No size element found", aLib->getName().c_str(), pRTName);
		//} //End of  rendertargets size


		//// Render Target clear values
		//TiXmlElement *pElemClear;
		//pElemClear = pElem->FirstChildElement ("clear");
		//	
		//if (0 != pElemClear) {

		//	float r = 0.0f, g = 0.0f, b = 0.0f, a = 0.0f; 
		//	pElemClear->QueryFloatAttribute ("r", &r);
		//	pElemClear->QueryFloatAttribute ("g", &g);
		//	pElemClear->QueryFloatAttribute ("b", &b);
		//	pElemClear->QueryFloatAttribute ("a", &a);

		//	m_RT->setPropf4(IRenderTarget::CLEAR_VALUES, vec4(r,g,b,a));
		//} //End of  rendertargets clear

		TiXmlElement *pElemColor;	
		TiXmlNode *pElemColors;
		pElemColors = pElem->FirstChild("colors");
		if (pElemColors != NULL) {
			pElemColor = pElemColors->FirstChildElement("color");

			for ( ; 0 != pElemColor; pElemColor = pElemColor->NextSiblingElement()) {

				const char *pNameColor = pElemColor->Attribute ("name");
				const char *internalFormat = pElemColor->Attribute("internalFormat");
				int layers = 0;

				if (0 == pNameColor) {
					NAU_THROW("File %s\nLibrary %s\nColor rendertarget has no name, in render target %s", ProjectLoader::s_File.c_str(), aLib->getName().c_str(), pRTName);							
				}

				if (internalFormat == 0) {
					NAU_THROW("File %s\nLibrary %s\nColor rendertarget %s has no internal format, in render target %s", ProjectLoader::s_File.c_str(), aLib->getName().c_str(), pNameColor, pRTName);
				}
				if (!ITexture::Attribs.isValid("INTERNAL_FORMAT", internalFormat)) //isValidInternalFormat(internalFormat))
					NAU_THROW("File %s\nLibrary %s\nRender Target %s: Color rendertarget %s internal format %s is invalid", ProjectLoader::s_File.c_str(), aLib->getName().c_str(), pRTName, pNameColor,internalFormat);

				sprintf(s_pFullName, "%s::%s", aLib->getName().c_str(), pNameColor);
					
				rt->addColorTarget (s_pFullName, internalFormat);
			}//End of rendertargets color
		}

		TiXmlElement *pElemDepth;
		pElemDepth = pElem->FirstChildElement ("depth");

		if (0 != pElemDepth) {
			const char *pNameDepth = pElemDepth->Attribute ("name");
			const char *internalFormat = pElemDepth->Attribute("internalFormat");

			if (0 == pNameDepth) {
				NAU_THROW("File %s\nLibrary %s\nRender Target %s: Depth rendertarget has no name", ProjectLoader::s_File.c_str(), aLib->getName().c_str(), pRTName);							
			}

			if (internalFormat == 0) {
				NAU_THROW("File %s\nLibrary %s\nRender Target %s: Depth rendertarget %s has no internal format", ProjectLoader::s_File.c_str(), aLib->getName().c_str(), pRTName, pNameDepth);
			}

			if (!ITexture::Attribs.isValid("INTERNAL_FORMAT", internalFormat)) //isValidInternalFormat(internalFormat))
					NAU_THROW("File %s\nLibrary %s\nRender Target %s: Depth rendertarget's internal format %s is invalid", ProjectLoader::s_File.c_str(), aLib->getName().c_str(), pRTName, internalFormat);

			sprintf(s_pFullName, "%s::%s", aLib->getName().c_str(), pNameDepth);
			rt->addDepthTarget (s_pFullName, internalFormat);
		}

		pElemDepth = pElem->FirstChildElement ("depthStencil");

		if (0 != pElemDepth) {
			const char *pNameDepth = pElemDepth->Attribute ("name");

			if (0 == pNameDepth) {
				NAU_THROW("File %s\nLibrary %s\nDepth/Stencil rendertarget has no name, in render target %s", ProjectLoader::s_File.c_str(), aLib->getName().c_str(), pRTName);							
			}
						
			sprintf(s_pFullName, "%s::%s", aLib->getName().c_str(), pNameDepth);
			rt->addDepthStencilTarget (s_pFullName);
		}

		SLOG("Render Target : %s width:%d height:%d samples:%d layers:%d", pRTName, 
			rt->getPropui2(IRenderTarget::SIZE).x,
			rt->getPropui2(IRenderTarget::SIZE).y, 
			rt->getPropui(IRenderTarget::SAMPLES),
			rt->getPropui(IRenderTarget::LAYERS));
		if (!rt->checkStatus()) {
			NAU_THROW("File %s\nLibrary %s\nRender target %s is not OK", ProjectLoader::s_File.c_str(), aLib->getName().c_str(), pRTName);							

		}
	}//End of rendertargets
}


/* -----------------------------------------------------------------------------
EVENTS

	NEED TO CHECK IF EVERY ASSET REQUIRED IS DEFINED	

-----------------------------------------------------------------------------*/

void
ProjectLoader::loadEvents(TiXmlHandle handle)
{
	TiXmlElement *pElem;


	pElem = handle.FirstChild ("sensors").FirstChild ("sensor").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
		const char *pName = pElem->Attribute ("name");
		const char *pClass = pElem->Attribute("class"); 
		const char *pObject = pElem->Attribute("object"); 

		if (0 == pName) {
			NAU_THROW("Sensor has no name in file %s", ProjectLoader::s_File.c_str());
		}

		if (0 == pClass) {
			NAU_THROW("Sensor %s has no class in file %s", pName, ProjectLoader::s_File.c_str());
		}


		if (!SensorFactory::validate(pClass)) {
			NAU_THROW("Invalid Class for Sensor %s, in file %s", pName, ProjectLoader::s_File.c_str());
		}

		Sensor *s;
		std::string propName;
		int iVal;
		float fVal;
		vec3 v3Val;
		float xVal, yVal, zVal;
		s = EVENTMANAGER->getSensor(pName,pClass);
		TiXmlElement *pElemAux;

		for (unsigned int i = 0; i < Sensor::COUNT_PROPTYPE; ++i) {

			switch (i) {
				case Sensor::BOOL: 
					for (unsigned int prop = 0; prop < s->getBoolPropCount(); ++prop) {
	
						propName = s->getBoolPropNames(prop);
						pElemAux = pElem->FirstChildElement(propName.c_str());
						if (pElemAux) {
							if (TIXML_SUCCESS != pElemAux->QueryIntAttribute ("value", &iVal)) {
								NAU_THROW("Sensor %s def error, field %s, in file %s", pName, propName.c_str(), ProjectLoader::s_File.c_str());
							}
							else
								s->setBool(prop, (iVal != 0));
						}
					}
					break;
				case Sensor::FLOAT:
					for (unsigned int prop = 0; prop < s->getFloatPropCount(); ++prop) {
	
						propName = s->getFloatPropNames(prop);
						pElemAux = pElem->FirstChildElement(propName.c_str());
						if (pElemAux) {
							if (TIXML_SUCCESS != pElemAux->QueryFloatAttribute ("value", &fVal)) {
								NAU_THROW("Sensor %s def error, field %s, in file %s", pName, propName.c_str(), ProjectLoader::s_File.c_str());
							}
							else
								s->setFloat(prop, fVal);
						}
					}
					break;
				case Sensor::VEC3:
					for (unsigned int prop = 0; prop < s->getVec3PropCount(); ++prop) {
	
						propName = s->getVec3PropNames(prop);
						pElemAux = pElem->FirstChildElement(propName.c_str());
						if (pElemAux) {
							if (TIXML_SUCCESS != pElemAux->QueryFloatAttribute ("x", &xVal) ||
								TIXML_SUCCESS != pElemAux->QueryFloatAttribute ("y", &yVal) ||
								TIXML_SUCCESS != pElemAux->QueryFloatAttribute ("z", &zVal)) {
								NAU_THROW("Sensor %s def error, field %s, in file %s", pName, propName.c_str(), ProjectLoader::s_File.c_str());
							}
							else {
								v3Val.set(xVal, yVal, zVal);
								s->setVec3(prop, v3Val);
							}
						}
					}
					break;
			}
		}
		s->init();
	}	
	// End of Sensors

	//Begin of Interpolators //Marta
	pElem = handle.FirstChild ("interpolators").FirstChild ("interpolator").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
		const char *pName = pElem->Attribute ("name");
		const char *pClass = pElem->Attribute("class"); 

		if (0 == pName) {
			NAU_THROW("Interpolator has no name in file %s", ProjectLoader::s_File.c_str());
		}
		if (0 == pClass) {
			NAU_THROW("Interpolator %s has no class in file %s", pName, ProjectLoader::s_File.c_str());
		}

		Interpolator *in= EVENTMANAGER->getInterpolator(pName, pClass);
		if (in==0) {
			NAU_THROW("Class definition error for interpolator %s in file %s ",pName, ProjectLoader::s_File.c_str());
		}

		TiXmlHandle hKeyFrames(pElem);
		TiXmlElement *pElemAux;
		pElemAux = hKeyFrames.FirstChild("keyFrames").FirstChild("keyFrame").Element();
		for (; 0 != pElemAux; pElemAux = pElemAux->NextSiblingElement()) {
		
				float key = 0.0, x=0.0f, y=0.0f, z=0.0f, w=0.0f;
				pElemAux->QueryFloatAttribute ("key", &key);
				pElemAux->QueryFloatAttribute ("x", &x);
				pElemAux->QueryFloatAttribute ("y", &y);
				pElemAux->QueryFloatAttribute ("z", &z);
				pElemAux->QueryFloatAttribute ("w", &w);

				in->addKeyFrame(key,vec4(x,y,z,w));
		}
	}

	// End of Interpolators


	//pElem = handle.FirstChild ("moveableobjects").FirstChild ("moveableobject").Element();
	//for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
	//	const char *pName = pElem->Attribute ("name");
	//	const char *pObject = pElem->Attribute("object");

	//	if (0 == pName) 
	//		NAU_THROW("MoveableObject has no name in file %s ", ProjectLoader::s_File.c_str());

	//	if (0 == pObject) {
	//		o=0;
	//	}

	//	std::shared_ptr<SceneObject> &o = RENDERMANAGER->getScene("MainScene")->getSceneObject(pObject); // substituir o MainScene, pode ter mais do q uma Cena?

	//	nau::event_::ObjectAnimation *oa= new nau::event_::ObjectAnimation(pName, o);

		//in->init((char *) pName, o, (char *)pKey, (char *)pKeyValue); 
		
	//}
	// End of MoveableObjects

	////Begin of routes //Marta
	pElem = handle.FirstChild ("routes").FirstChild ("route").Element();

	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {

		const char *pName = pElem->Attribute ("name");
		const char *pSender = pElem->Attribute ("sender");
		const char *pReceiver = pElem->Attribute ("receiver");
		const char *pIn = pElem->Attribute ("eventIn");
		const char *pOut = pElem->Attribute ("eventOut");

		if (0 == pName) 
			NAU_THROW("Route has no name in file %s", ProjectLoader::s_File.c_str());

		if (0 == pSender) 
			NAU_THROW("Route %s has no sender in file %s", pName, ProjectLoader::s_File.c_str());

		if (0 == pReceiver) 
			NAU_THROW("Route %s has no receiver in file %s", pName, ProjectLoader::s_File.c_str());

		if (0 == pIn) 
			NAU_THROW("Route %s has no eventIn in file %s", pName, ProjectLoader::s_File.c_str());

		if (0 == pOut) 
			NAU_THROW("Route %s has no eventOut in file %s", pName, ProjectLoader::s_File.c_str());		

		Route *r= EVENTMANAGER->getRoute(pName);
		r->init((char *) pName, (char *)pSender,(char *)pReceiver,(char *)pOut,(char *)pIn);
	}
	// End of routes
}

/* -----------------------------------------------------------------------------
MATERIAL MAPS

	MAP ALL TO ONE
	<materialMaps>
		<map fromMaterial="*" toLibrary="quadMaterials" toMaterial="flat-with-shadow" />
	</materialMaps>

	OR 

	MAP INDIVIDUAL
	<materialMaps>
		<map fromMaterial="quad" toLibrary="quadMaterials" toMaterial="quadpass2" />
	</materialMaps>

The field toLibrary indicates a previously defined material library in the assets part.

-----------------------------------------------------------------------------*/

void 
ProjectLoader::loadPassMaterialMaps(TiXmlHandle hPass, Pass *aPass)
{
	TiXmlElement *pElem;
	std::string library;

	pElem = hPass.FirstChild ("materialMaps").FirstChild ("map").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
		const char *pFromMaterial = pElem->Attribute ("fromMaterial");
		
		const char *pToLibrary = pElem->Attribute ("toLibrary");

		if (pToLibrary == 0)
			library = DEFAULTMATERIALLIBNAME;
		else
			library = pToLibrary;

		const char *pToMaterial = pElem->Attribute ("toMaterial");

		if ((pToMaterial == 0)) {
		    NAU_THROW("File %s\nPass %s\nMaterial map error: Missing destination material", 
				ProjectLoader::s_File.c_str(), aPass->getName().c_str());
		}

		if (0 == pFromMaterial && 0 != pToMaterial) {
		  
		    NAU_THROW("File %s\nPass %s\nMaterial map error: Missing origin material", 
				ProjectLoader::s_File.c_str(), aPass->getName().c_str());
		}
		else if (0 == pFromMaterial) {
			if (MATERIALLIBMANAGER->hasLibrary(library))
				aPass->remapAll (library);
			else
				NAU_THROW("File %s\nPass %s\nMaterial map error: Destination library %s is not defined", 
					ProjectLoader::s_File.c_str(), aPass->getName().c_str(), library.c_str());
		}
		else if (0 == strcmp (pFromMaterial, "*")) {
			if (MATERIALLIBMANAGER->hasMaterial(library, pToMaterial))
				aPass->remapAll (library, pToMaterial);
			else
				NAU_THROW("File %s\nPass %s\nMaterial map error: Destination material (%s,%s) is not defined", 
					ProjectLoader::s_File.c_str(), aPass->getName().c_str(), library.c_str(), pToMaterial);
		}
		else {
			if (MATERIALLIBMANAGER->hasMaterial(library, pToMaterial))
				aPass->remapMaterial (pFromMaterial, library, pToMaterial);
			else
				NAU_THROW("File %s\nPass %s\nMaterial map error: Destination material (%s,%s) is not defined", 
					ProjectLoader::s_File.c_str(), aPass->getName().c_str(), library.c_str(), pToMaterial);
		}
	} //End of map

}




/* -----------------------------------------------------------------------------
MAPS - Allow the setting of individual settings of loaded materials

	<injectionMaps>
		<map toMaterial="Grade*">
			<state name="Grades"  fromLibrary="bli"/>
			<shader fromMaterial="bla" fromLibrary="bli" />
			<color fromMaterial="bla" fromLibrary="bli" ambient=true diffuse=false emission=false specular=false shininess=false />
			<textures>
				<texture name="tex" fromLibrary="bli" toUnit="0" >
					<depthCompare mode="COMPARE_REF_TO_TEXTURE" func="LEQUAL" />
					<filtering min="LINEAR" mag="LINEAR" />
				</texture>
			</textures>
			<buffers>
				<buffer name="rays" fromLibrary="PrimeShadows">
					<TYPE value="SHADER_STORAGE" />
					<BINDING_POINT value="1" />
				</buffer>
			</buffers>

		</map>
	</injectionMaps>

Creates a new material in the passe's library with the same name, and the same 
values of the original material, replaces the defined properties.

An * can be used at the end of the material, and it works as a wildcard.

-----------------------------------------------------------------------------*/

void 
ProjectLoader::loadPassInjectionMaps(TiXmlHandle hPass, Pass *aPass)
{
	TiXmlElement *pElem, *pElemAux;

	std::vector<std::string> ok = {"map"};
	std::string s = aPass->getName() + " injectionMaps";
	checkForNonValidChildTags(s, ok, hPass.FirstChild("injectionMaps").Element());


	pElem = hPass.FirstChild ("injectionMaps").FirstChild ("map").Element();
	for ( ; pElem != NULL; pElem = pElem->NextSiblingElement()) {
	
		std::vector<std::string> ok = {"state", "shader", "color", "textures", "imageTextures", "buffers"};
		std::string s = aPass->getName() + " injectionMaps";
		checkForNonValidChildTags(s, ok, pElem);

		const char* pMat = pElem->Attribute("toMaterial");

		if (0 == pMat)
			NAU_THROW("File %s\nPass %s\nInjection map error: A name is required for the material", ProjectLoader::s_File.c_str(), aPass->getName().c_str());
	
		MaterialLib *defLib = MATERIALLIBMANAGER->getLib(DEFAULTMATERIALLIBNAME);
		std::vector<std::string> names;
		std::string sss = std::string(pMat);
		defLib->getMaterialNames(sss, &names);

		if (names.size() == 0)
			NAU_THROW("File %s\nPass %s\nInjection map error: No materials match %s", ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pMat);

		for(auto& name:names) {

			std::shared_ptr<Material> &srcMat = defLib->getMaterial(name);
			std::shared_ptr<Material> &dstMat = MATERIALLIBMANAGER->cloneMaterial(srcMat);
			MATERIALLIBMANAGER->addMaterial(aPass->getName(), dstMat);

			aPass->remapMaterial (name, aPass->getName(), name);
		}


		pElemAux = pElem->FirstChildElement("state");
		if (pElemAux) {
	
			const char *pState = pElemAux->Attribute ("name");
			const char *pLib = pElemAux->Attribute ("fromLibrary");

			if (0 == pState || 0 == pLib) {
			  NAU_THROW("File %s\nPass%s\nInjection map error: State map error", ProjectLoader::s_File.c_str(), aPass->getName().c_str());
			}

			
			std::string fullName = std::string(pLib) + "::" + pState;
			if (!RESOURCEMANAGER->hasState(fullName))
				NAU_THROW("File %s\nPass%s\nInjection map error: State %s is not defined in lib %s", ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pState, pLib);
		
			for(auto& name: names) {

				std::shared_ptr<Material> &dstMat = MATERIALLIBMANAGER->getMaterial(aPass->getName(), name);
				dstMat->setState(RESOURCEMANAGER->getState(fullName));
			}
		}
	
		pElemAux = pElem->FirstChildElement("shader");
		if (pElemAux) {

			const char *pMat = pElemAux->Attribute ("fromMaterial");
			const char *pLib = pElemAux->Attribute ("fromLibrary");

			if (0 == pMat || 0 == pLib) {
			  NAU_THROW("File %s\nPass%s\nInjection map error: Shader name and library are required", 
				  ProjectLoader::s_File.c_str(), aPass->getName().c_str());
			}

			if (!MATERIALLIBMANAGER->hasMaterial(pLib,pMat))
				NAU_THROW("File %s\nPass%s\nInjection map error: Shader material %s is not defined in lib %s", 
					ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pMat, pLib);

			for(auto& name:names) {

				std::shared_ptr<Material> &dstMat = MATERIALLIBMANAGER->getMaterial(aPass->getName(), name);
				dstMat->cloneProgramFromMaterial(MATERIALLIBMANAGER->getMaterial(pLib,pMat));
			}
		}

		TiXmlNode *pElemNode = pElem->FirstChild("textures");
		if (pElemNode) {
			pElemAux = pElemNode->FirstChildElement ("texture");
			for ( ; pElemAux != NULL; pElemAux = pElemAux->NextSiblingElement()) {
	
				const char *pName = pElemAux->Attribute ("name");
				const char *pLib = pElemAux->Attribute ("fromLibrary");
		

				if (0 == pName || 0 == pLib) {
				  NAU_THROW("File %s\nPass%s\nInjection map error: Texture name and library are required", ProjectLoader::s_File.c_str(), aPass->getName().c_str());
				}

				int unit;
				if (TIXML_SUCCESS != pElemAux->QueryIntAttribute("toUnit", &unit))
				  NAU_THROW("File %s\nPass%s\nInjection map error: Texture unit not specified", ProjectLoader::s_File.c_str(), aPass->getName().c_str());

				sprintf(s_pFullName, "%s::%s", pLib, pName);
				if (!RESOURCEMANAGER->hasTexture(s_pFullName))
					NAU_THROW("File %s\nPass%s\nInjection map error: Texture %s is not defined in lib %s", ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pName, pLib);

				for(auto &name:names) {

					std::shared_ptr<Material> &dstMat = MATERIALLIBMANAGER->getMaterial(aPass->getName(), name);
					dstMat->attachTexture(unit, RESOURCEMANAGER->getTexture(s_pFullName));
					std::map<std::string, std::unique_ptr<Attribute>> &attribs = ITextureSampler::Attribs.getAttributes();

					std::string s = aPass->getName() + " injectionMaps: texture " + s_pFullName;
					std::vector <std::string> excluded;
					readChildTags(s, (AttributeValues *)dstMat->getTextureSampler(unit), ITextureSampler::Attribs,
						excluded, pElemAux);
				}	
			}
		}

		pElemNode = pElem->FirstChild("imageTextures");
		if (pElemNode) {

			if (!APISupport->apiSupport(IAPISupport::IMAGE_TEXTURE))
				NAU_THROW("Image textures require OpenGL 4.2 or greater. Current version is %d", APISupport->getVersion());

			pElemAux = pElemNode->FirstChildElement("imageTexture");
			for (; pElemAux != NULL; pElemAux = pElemAux->NextSiblingElement()) {

				const char *pName = pElemAux->Attribute("texture");
				const char *pLib = pElemAux->Attribute("fromLibrary");


				if (0 == pName || 0 == pLib) {
					NAU_THROW("File %s\nPass%s\nInjection map error: Image Texture name and library are required", ProjectLoader::s_File.c_str(), aPass->getName().c_str());
				}

				int unit;
				if (TIXML_SUCCESS != pElemAux->QueryIntAttribute("toUnit", &unit))
					NAU_THROW("File %s\nPass%s\nInjection map error: Image Texture unit not specified in material maps", ProjectLoader::s_File.c_str(), aPass->getName().c_str());

				sprintf(s_pFullName, "%s::%s", pLib, pName);
				if (!RESOURCEMANAGER->hasTexture(s_pFullName))
					NAU_THROW("File %s\nPass%s\nInjection map error: Texture %s is not defined in lib %s", ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pName, pLib);

				for (auto& name:names) {

					std::shared_ptr<Material> &dstMat = MATERIALLIBMANAGER->getMaterial(aPass->getName(), name);
					ITexture *t = RESOURCEMANAGER->getTexture(s_pFullName);
					dstMat->attachImageTexture(t->getLabel(), unit, t->getPropi(ITexture::ID));

					std::string s = "File " + ProjectLoader::s_File + "\nPass " + aPass->getName() + "\nInjection map error: Image Texture unit " + std::to_string(unit);
					std::vector <std::string> excluded;
					readChildTags(s, (AttributeValues *)dstMat->getImageTexture(unit), IImageTexture::Attribs,
						excluded, pElemAux);
				}
			}
		}

		pElemNode = pElem->FirstChild("buffers");
		if (pElemNode) {
			if (!APISupport->apiSupport(IAPISupport::BUFFER_ATOMICS))
				NAU_THROW("Buffers require OpenGL 4.2 or greater. Current version is %d", APISupport->getVersion());

			pElemAux = pElemNode->FirstChildElement("buffer");
			for (; pElemAux != NULL; pElemAux = pElemAux->NextSiblingElement()) {

				const char *pName = pElemAux->Attribute("name");
				const char *pLib = pElemAux->Attribute("fromLibrary");

				if (0 == pName || 0 == pLib) {
					NAU_THROW("File %s\nPass%s\nInjection map error: Buffer name and library are required", ProjectLoader::s_File.c_str(), aPass->getName().c_str());
				}

				sprintf(s_pFullName, "%s::%s", pLib, pName);
				if (!RESOURCEMANAGER->hasBuffer(s_pFullName))
					NAU_THROW("File %s\nPass%s\nInjection map error: Buffer %s is not defined in lib %s", ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pName, pLib);

				for (auto &name:names) {

					std::shared_ptr<Material> &dstMat = MATERIALLIBMANAGER->getMaterial(aPass->getName(), name);
					IBuffer *buffer = RESOURCEMANAGER->getBuffer(s_pFullName);
					IMaterialBuffer *imb = IMaterialBuffer::Create(buffer);
					//IBuffer *b = buffer->clone();

					std::string s = aPass->getName() + " injectionMaps: buffer " + s_pFullName;
					std::vector <std::string> excluded;
					readChildTags(s, (AttributeValues *)imb, IMaterialBuffer::Attribs,
						excluded, pElemAux);

					dstMat->attachBuffer(imb);
				}
			}
		}

		pElemAux = pElem->FirstChildElement("color");
		if (pElemAux) {

			const char *pMat = pElemAux->Attribute ("fromMaterial");
			const char *pLib = pElemAux->Attribute ("fromLibrary");
			const char *pDiffuse = pElemAux->Attribute("diffuse");
			const char *pAmbient = pElemAux->Attribute("ambient");
			const char *pSpecular = pElemAux->Attribute("specular");
			const char *pEmission = pElemAux->Attribute("emission");
			const char *pShininess = pElemAux->Attribute("shininess");

			if (0 == pMat || 0 == pLib) {
			  NAU_THROW("File %s\nPass%s\nInjection map error: Color material name and library are required", 
				  ProjectLoader::s_File.c_str(), aPass->getName().c_str());
			}

			if (!MATERIALLIBMANAGER->hasMaterial(pLib,pMat))
				NAU_THROW("File %s\nPass%s\nInjection map error: Material %s is not defined in lib %s", 
					ProjectLoader::s_File.c_str(), aPass->getName().c_str(), pMat, pLib);

			std::shared_ptr<Material> &srcMat = MATERIALLIBMANAGER->getMaterial(pLib,pMat);
			for (auto &name : names) {

				std::shared_ptr<Material> &dstMat = MATERIALLIBMANAGER->getMaterial(aPass->getName(), name);
				if (!pAmbient && !pDiffuse && !pSpecular && !pEmission && !pShininess)
					dstMat->getColor().clone(srcMat->getColor());
				if (pAmbient && !strcmp("true",pAmbient))
					dstMat->getColor().setPropf4(ColorMaterial::AMBIENT, srcMat->getColor().getPropf4(ColorMaterial::AMBIENT));
				if (pDiffuse && !strcmp("true",pDiffuse))
					dstMat->getColor().setPropf4(ColorMaterial::DIFFUSE, srcMat->getColor().getPropf4(ColorMaterial::DIFFUSE));
				if (pSpecular && !strcmp("true",pSpecular))
					dstMat->getColor().setPropf4(ColorMaterial::SPECULAR, srcMat->getColor().getPropf4(ColorMaterial::SPECULAR));
				if (pEmission && !strcmp("true",pEmission))
					dstMat->getColor().setPropf4(ColorMaterial::EMISSION, srcMat->getColor().getPropf4(ColorMaterial::EMISSION));
				if (pShininess && !strcmp("true",pShininess))
					dstMat->getColor().setPropf(ColorMaterial::SHININESS, srcMat->getColor().getPropf(ColorMaterial::SHININESS));
			}

		}
	}
}

/* -----------------------------------------------------------------------------
PIPELINES   AND    PASSES     

	<pipelines>
		<pipeline name="shadow" default="true" defaultCamera="aName">
			<pass class="default" name="pass1">
				...
			</pass>
		</pipeline>
	</pipelines>

in the pipeline definition, if default is not present then the first pipeline
will be the default

in the pass definition if class is not present it will be "default"
-----------------------------------------------------------------------------*/
void
ProjectLoader::loadPipelines (TiXmlHandle &hRoot) {

	TiXmlElement *pElem;
	TiXmlHandle handle (0);
	std::map<std::string, Pass*> passMapper;


	char activePipeline[256];

	memset (activePipeline, 0, 256);


	pElem = hRoot.FirstChild ("pipelines").FirstChild ("pipeline").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement("pipeline")) {
		const char *pNamePip = pElem->Attribute ("name");
		const char *pDefaultCamera = pElem->Attribute("defaultCamera");

		if (0 == pNamePip) 
			NAU_THROW("File %s\nPipeline has no name", ProjectLoader::s_File.c_str());

		if (pDefaultCamera && !(RENDERMANAGER->hasCamera(pDefaultCamera)))
			NAU_THROW("File %s\nPipeline %s\nCamera %s, is not defined", ProjectLoader::s_File.c_str(), pNamePip, pDefaultCamera);

		if (RENDERMANAGER->hasPipeline(pNamePip))
			NAU_THROW("File %s\nPipeline %s redefinition", ProjectLoader::s_File.c_str(), pNamePip);

		std::shared_ptr<Pipeline> &aPipeline = RENDERMANAGER->createPipeline (pNamePip);
		
		unsigned int k = 0;
		pElem->QueryUnsignedAttribute("frameCount", &k);

		aPipeline->setFrameCount(k);

		handle = TiXmlHandle (pElem);
		TiXmlElement *pElemPass;

#ifdef NAU_LUA
		pElemPass = handle.FirstChild("preScript").Element();
		if (pElemPass != NULL) {
			

			const char *pPreScriptFile = pElemPass->Attribute("file");
			const char *pPreScriptName = pElemPass->Attribute("script");
			if (pPreScriptFile && pPreScriptName)
				aPipeline->setPreScript(File::GetFullPath(ProjectLoader::s_Path, pPreScriptFile), pPreScriptName);
			else {
				NAU_THROW("File %s\nPipeline %s\nPre script definition must have both file and script attributes", ProjectLoader::s_File.c_str(), pNamePip);
			}
		}

		pElemPass = handle.FirstChild("postScript").Element();
		if (pElemPass != NULL) {

			const char *pPostScriptFile = pElemPass->Attribute("file");
			const char *pPostScriptName = pElemPass->Attribute("script");
			if (pPostScriptFile && pPostScriptName)
				aPipeline->setPostScript(File::GetFullPath(ProjectLoader::s_Path, pPostScriptFile), pPostScriptName);
			else {
				NAU_THROW("File %s\nPipeline %s\nPost script definition must have both file and name attributes", ProjectLoader::s_File.c_str(), pNamePip);
			}
		}
#endif
		pElemPass = handle.FirstChild ("pass").Element();
		for ( ; 0 != pElemPass; pElemPass = pElemPass->NextSiblingElement("pass")) {
			
			
			TiXmlHandle hPass (pElemPass);
			const char *pName = pElemPass->Attribute ("name");
			const char *pClass = pElemPass->Attribute ("class");

			if (0 == pName) 
				NAU_THROW("File %s\nPipeline %s\nPass has no name", ProjectLoader::s_File.c_str(), pNamePip);

			if (RENDERMANAGER->hasPass(pNamePip, pName))
				NAU_THROW("File %s\nPipeline %s\nPass %s redefinition", ProjectLoader::s_File.c_str(), pNamePip, pName);

			Pass *aPass = 0;
			std::string passClass;
			if (0 == pClass) {
				aPass = aPipeline->createPass(pName);
				passClass = "default";
			} else {
				if (PASSFACTORY->isClass(pClass))
					aPass = aPipeline->createPass (pName, pClass);
				else {
					NAU_THROW("File %s\nPipeline %s\nClass %s is not available, in pass %s", ProjectLoader::s_File.c_str(), pNamePip, pClass, pName);
				}
				passClass = pClass;
			}
			passMapper[pName] = aPass;

				
			loadPassMode(hPass, aPass);	

			loadPassScripts(hPass, aPass);

			loadPassPreProcess(hPass, aPass);
			loadPassPostProcess(hPass, aPass);

			if (passClass != "optixPrime" && passClass != "quad" && passClass != "profiler") {

				loadPassScenes(hPass,aPass);
				loadPassCamera(hPass,aPass);	
			}
			else if (passClass == "quad") {
				loadPassTexture(hPass,aPass);
				loadPassMaterial(hPass, aPass);
			}
#ifdef NAU_OPTIX
			if (passClass == "optix")
				loadPassOptixSettings(hPass, aPass);
#endif
			if (passClass == "compute") {
				if (APISupport->apiSupport(IAPISupport::COMPUTE_SHADER))
					loadPassComputeSettings(hPass, aPass);
				else {
					NAU_THROW("Compute Shader is not supported");
				}
			}
#ifdef NAU_OPTIX
			if (passClass == "optixPrime")
				loadPassOptixPrimeSettings(hPass, aPass);
#endif

			loadPassParams(hPass, aPass);
			//loadPassClearDepthAndColor(hPass, aPass);
			loadPassRenderTargets(hPass, aPass, passMapper);
			loadPassViewport(hPass, aPass);
			loadPassLights(hPass, aPass);

			loadPassInjectionMaps(hPass, aPass);
			loadPassMaterialMaps(hPass, aPass);
		} //End of pass
		if (pDefaultCamera)
			aPipeline->setDefaultCamera(pDefaultCamera);

	} //End of pipeline
	
	pElem = hRoot.FirstChild("pipelines").Element();
	const char *pDefault = pElem->Attribute("default");
	const char *pMode = pElem->Attribute("mode");

	if (pMode && strcmp(pMode, "RUN_DEFAULT"))
		RENDERMANAGER->setRunMode(pMode);
	if (pDefault) {
		if (RENDERMANAGER->hasPipeline(pDefault)) {
			RENDERMANAGER->setActivePipeline(pDefault);
		}
		else {
			NAU_THROW("File %s\nDefault pipeline %s is not defined", s_File.c_str(), pDefault);
		}
	}
	else {
		RENDERMANAGER->setActivePipeline(0);
	}

}


/* -----------------------------------------------------------------------------
INTERFACE

<interface>
	<window name="bla"  label="My Bar">
		<var label="direction" type="LIGHT" context="Sun" component="DIRECTION" option="DIRECTION"/>
		<var label="darkColor" type="RENDERER" context="CURRENT" component="dark" option="COLOR" />
		<pipelines />
	<window>

	<window ... >
	</window>
</interface>

 ----------------------------------------------------------------------------- */


void 
ProjectLoader::loadInterface(TiXmlHandle & hRoot) {

	TiXmlElement *pElem;
	TiXmlHandle handle(0);

	char activePipeline[256];

	memset(activePipeline, 0, 256);


	handle = hRoot.FirstChild("interface");
	loadAtomicSemantics(handle);

	pElem = hRoot.FirstChild("interface").FirstChild("window").Element();
	for (; 0 != pElem; pElem = pElem->NextSiblingElement("window")) {
		const char *pWindowName = pElem->Attribute("label");
		//const char *pWindowLabel = pElem->Attribute("label");

		if (0 == pWindowName /*|| 0 == pWindowLabel*/) {
			NAU_THROW("File %s\nInterface window needs a name", s_File.c_str());
		}

		INTERFACE->createWindow(pWindowName);// , pWindowLabel);

		TiXmlElement *pElemAux = pElem->FirstChildElement();
		for (; 0 != pElemAux; pElemAux = pElemAux->NextSiblingElement()) {

			const char *tag = pElemAux->Value();
			if (strcmp(tag, "pipelineList") == 0) {
				const char *pLabel = pElemAux->Attribute("label");
				if (0 == pLabel) {
					NAU_THROW("File %s\nWindow %s, Pipeline list needs a label", s_File.c_str(), pWindowName);
				}
				INTERFACE->addPipelineList(pWindowName, pLabel);
			}
			if (strcmp(tag, "var") == 0) {
				const char *pLabel = pElemAux->Attribute("label");
				const char *pType = pElemAux->Attribute("type");
				const char *pContext = pElemAux->Attribute("context");
				const char *pComponent = pElemAux->Attribute("component");
				const char *pControl = pElemAux->Attribute("option");
				int id = 0;
				pElemAux->QueryIntAttribute("id", &id);

				// check if fields are filled)
				if (0 == pLabel) {
					NAU_THROW("File %s\nWindow %s, Variable needs a label", s_File.c_str(), pWindowName);
				}

				if (0 == pType || 0 == pContext || 0 == pComponent) {
					NAU_THROW("File %s\nWindow %s, Variable %s\nVariable needs a type, a context and a component",
						s_File.c_str(), pWindowName, pLabel);
				}

				// check if var is well defined
				std::string message;
				validateObjectAttribute(pType, pContext, pComponent, &message);
				if (message != "") {
					NAU_THROW("File %s\nWindow %s, Variable %s\n%s", s_File.c_str(), pWindowName, pLabel, message.c_str());
				}

				//if (!NAU->validateShaderAttribute(pType, pContext, pComponent))
				//	NAU_THROW("File %s\nWindow %s, Variable %s\nVariable not valid", 
				//		s_File.c_str(), pWindowName, pLabel);
				if (pControl) {
					if (strcmp(pControl, "DIRECTION") == 0)
						INTERFACE->addDir(pWindowName, pLabel, pType, pContext, pComponent, id);
					else if (strcmp(pControl, "COLOR") == 0)
						INTERFACE->addColor(pWindowName, pLabel, pType, pContext, pComponent, id);
				}
				else
				INTERFACE->addVar(pWindowName, pLabel, pType, pContext, pComponent, id);
			}

		}

	}

}


/* -----------------------------------------------------------------------------
PRE POST PROCESS

<preProcess>
	<texture name="bla" fromLibrary="blu" CLEAR_LEVEL=0 />
	<texture name="ble" fromLibrary="blu" CLEAR=true />
</preProcess>

<preProcess>
	<texture name="bla" fromLibrary="blu" MIPMAP=true />
</preProcess>

 ----------------------------------------------------------------------------- */

void 
ProjectLoader::loadPassPreProcess(TiXmlHandle hPass, Pass *aPass) 
{
	TiXmlElement *pElem;
	std::vector <std::string> excluded = {"name", "fromLibrary"};

	pElem = hPass.FirstChild("preProcess").FirstChild().Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
		if (!strcmp(pElem->Value(), "texture")) {

			PassProcessTexture *pp  = new PassProcessTexture();

			std::string s;
			readItemFromLib(pElem, "name", &s);
			ITexture *t = RESOURCEMANAGER->getTexture(s);
			if (!t) {
				NAU_THROW("File %s\nPass %s\nPre process texture %s does not exist", s_File.c_str(), aPass->getName().c_str(), s.c_str());
			}

			readAttributes(aPass->getName(), (AttributeValues *)pp, PassProcessTexture::Attribs, excluded, pElem);
			pp->setItem(t);
			aPass->addPreProcessItem(pp);
		}

		else if (!strcmp(pElem->Value(), "buffer")) {
				
			PassProcessBuffer *pp  = new PassProcessBuffer();

			std::string s;
			readItemFromLib(pElem, "name", &s);
			IBuffer *b = RESOURCEMANAGER->getBuffer(s);
			if (!b) {
				NAU_THROW("File %s\nPass %s\nPost process buffer %s does not exist", s_File.c_str(), aPass->getName().c_str(), s.c_str());
			}

			readAttributes(aPass->getName(), (AttributeValues *)pp, PassProcessBuffer::Attribs, excluded, pElem);
			pp->setItem(b);
			aPass->addPreProcessItem(pp);
		}
		else {
			NAU_THROW("File %s\nPass %s\nError in pre process tag\nValid child tags are: texture or buffer", s_File.c_str(), aPass->getName().c_str());
		}


	}
}


void 
ProjectLoader::loadPassPostProcess(TiXmlHandle hPass, Pass *aPass) 
{
	TiXmlElement *pElem;
	std::vector <std::string> excluded = {"name", "fromLibrary"};

	pElem = hPass.FirstChild("postProcess").FirstChild().Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
		if (!strcmp(pElem->Value(), "texture")) {

			PassProcessTexture *pp  = new PassProcessTexture();

			std::string s;
			readItemFromLib(pElem, "name", &s);
			ITexture *t = RESOURCEMANAGER->getTexture(s);
			if (!t) {
				NAU_THROW("File %s\nPass %s\nPost process texture %s does not exist", s_File.c_str(), aPass->getName().c_str(), s.c_str());
			}

			readAttributes(aPass->getName(), (AttributeValues *)pp, PassProcessTexture::Attribs, excluded, pElem);
			pp->setItem(t);
			aPass->addPostProcessItem(pp);
		}

		else if (!strcmp(pElem->Value(), "buffer")) {
				
			PassProcessBuffer *pp  = new PassProcessBuffer();

			std::string s;
			readItemFromLib(pElem, "name", &s);
			IBuffer *b = RESOURCEMANAGER->getBuffer(s);
			if (!b) {
				NAU_THROW("File %s\nPass %s\nPost process buffer %s does not exist", s_File.c_str(), aPass->getName().c_str(), s.c_str());
			}

			readAttributes(aPass->getName(), (AttributeValues *)pp, PassProcessBuffer::Attribs, excluded, pElem);
			pp->setItem(b);
			aPass->addPostProcessItem(pp);
		}
		else {
			NAU_THROW("File %s\nPass %s\nError in pre process tag\nValid child tags are: texture or buffer", s_File.c_str(), aPass->getName().c_str());
		}
	}
}


/* -----------------------------------------------------------------------------
BUFFERS

<buffer name="atBuffer">
	<CLEAR value="BY_FRAME" />

	<SIZE value=16 />
	or
	<DIM x=256 y=256 z=1 />
	<structure>
		<item value="FLOAT" />
		<item value="FLOAT" />
	</structure>
</buffer>

All fields are required. SIZE is in bytes. DIM requires the definition of structure.
-----------------------------------------------------------------------------*/


void
ProjectLoader::loadMatLibBuffers(TiXmlHandle hRoot, MaterialLib *aLib, std::string path)
{

	TiXmlElement *pElem;
	int layers = 0;
	pElem = hRoot.FirstChild("buffers").FirstChild("buffer").Element();
	for (; 0 != pElem; pElem = pElem->NextSiblingElement("buffer")) {
		const char* pName = pElem->Attribute("name");

		if (0 == pName) {
			NAU_THROW("Mat Lib %s\nBuffer has no name", aLib->getName().c_str());
		}
		sprintf(s_pFullName, "%s::%s", aLib->getName().c_str(), pName);

		IBuffer *b = RESOURCEMANAGER->createBuffer(s_pFullName);

		SLOG("Buffer : %s", s_pFullName);
		TiXmlHandle handle(pElem);
		TiXmlElement *pElemAux = handle.FirstChild("structure").FirstChild("field").Element();

		for (; pElemAux != NULL; pElemAux = pElemAux->NextSiblingElement("field")) {
			const char *pType = pElemAux->Attribute("value");
			if (!pType) {
				NAU_THROW("Mat Lib %s\nBuffer %s - field has no value", aLib->getName().c_str(), pName);
			}
			if (!Enums::isValidType(pType)) {
				NAU_THROW("Mat Lib %s\nBuffer %s - field has an invalid type. Valid types are: INT, IVEC2, IVEC3, IVEC4, UINT, UIVEC2, UIVEC3, UIVEC4, BOOL, BVEC2, BVEC3, BVEC4, FLOAT, VEC2, VEC3, VEC4, DOUBLE, DVEC2, DVEC3, DVEC4, MAT2, MAT3, MAT4, MAT2x3, MAT2x4, MAT3x2, MAT3x4, MAT4x2, MAT4x3, DMAT2, DMAT3, DMAT4, DMAT2x3, DMAT2x4, DMAT3x2, DMAT3x4, DMAT4x2, DMAT4x3, SAMPLER, ENUM, BYTE, UBYTE, SHORT, USHORT", aLib->getName().c_str(), pName);
			}
			b->appendItemToStruct(Enums::getType(pType));
		}

		// Reading buffer attributes
		std::vector<std::string> excluded;
		excluded.push_back("structure");
		readChildTags(pName, (AttributeValues *)b, IBuffer::Attribs, excluded, pElem);
	}
}



/* -----------------------------------------------------------------------------
TEXTURES

	<textures>
		<texture name="Grade_01_02_03" filename="../Texturas/AL01_Grade1.tif" mipmap="1" />
		<cubeMap name="CMHouse" 
			filePosX="../TextureCubeMaps/cubemaphouse/cm_right.jpg"
			fileNegX="../TextureCubeMaps/cubemaphouse/cm_left.jpg"
			filePosY="../TextureCubeMaps/cubemaphouse/cm_top.jpg"
			fileNegY="../TextureCubeMaps/cubemaphouse/cm_bottom.jpg"
			filePosZ="../TextureCubeMaps/cubemaphouse/cm_front.jpg"
			fileNegZ="../TextureCubeMaps/cubemaphouse/cm_back.jpg"	
		/>
		<texture name="tex">
			<WIDTH value=512 />
			<HEIGHT value=512 />
			<INTERNAL_FORMAT value="RGBA" />
			<MIPMAP value="true" />
		</texture>	
		<texture name="source" filename="../../models/Textures/bla.tif" mipmap="0" />
	</textures>

.
The paths may be relative to the material lib file, or absolute.
-----------------------------------------------------------------------------*/
void 
ProjectLoader::loadMatLibTextures(TiXmlHandle hRoot, MaterialLib *aLib, std::string path)
{
	TiXmlElement *pElem;
	int layers = 0;
	pElem = hRoot.FirstChild ("textures").FirstChild ("texture").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement("texture")) {
		const char* pTextureName = pElem->Attribute ("name");
		const char* pFilename = pElem->Attribute ("filename");

		if (0 == pTextureName) {
			NAU_THROW("Mat Lib %s\nTexture has no name", aLib->getName().c_str());
		} 

		sprintf(s_pFullName,"%s::%s", aLib->getName().c_str(), pTextureName);

		SLOG("Texture : %s", s_pFullName);

		if (RESOURCEMANAGER->hasTexture(s_pFullName))
			NAU_THROW("Mat Lib %s\nTexture %s is already defined", aLib->getName().c_str(), s_pFullName);

		// empty texture
		if (0 == pFilename) {
			ITexture *tex = RESOURCEMANAGER->createTexture(s_pFullName);

			std::vector<std::string> excluded;
			readChildTags(s_pFullName, (AttributeValues *)tex, ITexture::Attribs, excluded, pElem);

			tex->build();
		}
		// texture from a file
		else {
			bool mipmap = false;
			pElem->QueryBoolAttribute("mipmap", &mipmap);

			RESOURCEMANAGER->addTexture (File::GetFullPath(path,pFilename), s_pFullName, mipmap);
		}
	}

	pElem = hRoot.FirstChild ("textures").FirstChild ("cubeMap").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement("cubeMap")) {
		const char* pTextureName = pElem->Attribute ("name");
		const char* pFilePosX = pElem->Attribute ("filePosX");
		const char* pFileNegX = pElem->Attribute ("fileNegX");
		const char* pFilePosY = pElem->Attribute ("filePosY");
		const char* pFileNegY = pElem->Attribute ("fileNegY");
		const char* pFilePosZ = pElem->Attribute ("filePosZ");
		const char* pFileNegZ = pElem->Attribute ("fileNegZ");
		const char *pMipMap = pElem->Attribute("mipmap");

		if (0 == pTextureName) {
			NAU_THROW("Mat Lib %s\nCube Map texture has no name", aLib->getName().c_str());
		} 

		if (!(pFilePosX && pFileNegX && pFilePosY && pFileNegY && pFilePosZ && pFileNegZ)) {
			NAU_THROW("Mat Lib %s\nCube Map is not complete", aLib->getName().c_str());
		}

		sprintf(s_pFullName,"%s::%s", aLib->getName().c_str(), pTextureName);
		if (RESOURCEMANAGER->hasTexture(s_pFullName))
			NAU_THROW("Mat Lib %s\nCube Map %s is already defined", aLib->getName().c_str(), s_pFullName);

		bool mipmap = false;
		if (pMipMap != 0 && strcmp(pMipMap,"1") == 0)
			mipmap = true;


		std::vector<std::string> files(6);
		files[0] = File::GetFullPath(path,pFilePosX);
		files[1] = File::GetFullPath(path,pFileNegX);
		files[2] = File::GetFullPath(path,pFilePosY);
		files[3] = File::GetFullPath(path,pFileNegY);
		files[4] = File::GetFullPath(path,pFilePosZ);
		files[5] = File::GetFullPath(path,pFileNegZ);

		RESOURCEMANAGER->addTexture (files, s_pFullName, mipmap);		
	}
}




/* -----------------------------------------------------------------------------
STATES

	<states>
		<state name="Grades">
			<ORDER value="2" />
			<BLEND value=true />
			<BLEND_SRC value="SRC_ALPHA" />
			<BLEND_DST value="ONE_MINUS_SRC_ALPHA" />
			<CULL_FACE value="0" />
		</state>
	</states>

func: NEVER, ALWAYS, LESS, LEQUAL, EQUAL, GEQUAL, GREATER, NOT_EQUAL

order - a number which indicates the order for material drawing: higher values are drawn later,
	negative values are not drawn

	
-----------------------------------------------------------------------------*/
void
ProjectLoader::loadState(TiXmlElement *pElemAux, MaterialLib *aLib, Material *aMat, IState *s) 
{

	//std::map<std::string, Attribute> attribs = IState::Attribs.getAttributes();
	//TiXmlElement *p = pElemAux->FirstChildElement();
		
	std::vector<std::string> excluded;
	readChildTags(s->getName(), (AttributeValues *)s, IState::Attribs, excluded, pElemAux);
	//Attribute a;
	//void *value;
	//while (p) {
	//	// trying to define an attribute that does not exist		
	//	if (attribs.count(p->Value()) == 0)
	//		NAU_THROW("Library %s: State %s: %s is not an attribute", aLib->getName().c_str(), s->getName().c_str(), p->Value());
	//	// trying to set the value of a read only attribute
	//	a = attribs[p->Value()];
	//	if (a.m_ReadOnlyFlag)
	//		NAU_THROW("Library %s: State %s: %s is a read-only attribute, in file %s", aLib->getName().c_str(),  s->getName().c_str(), p->Value());

	//	value = readChildTag(s->getName(), p, a.m_Type, IState::Attribs);
	//	s->setProp(a.m_Id, a.m_Type, value);
	//	//aMat->getTextureSampler(unit)->setProp(a.mId, a.mType, value);
	//	p = p->NextSiblingElement();
	//}
}


void 
ProjectLoader::loadMatLibStates(TiXmlHandle hRoot, MaterialLib *aLib)
{
	TiXmlElement *pElem;
	pElem = hRoot.FirstChild("states").FirstChild("state").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
		const char* pStateName = pElem->Attribute ("name");
		
		if (0 == pStateName) {
			NAU_THROW("Mat Lib %s\nState has no name", aLib->getName().c_str());
		}

		SLOG("State: %s", pStateName);

		//sprintf(s_pFullName, "%s::%s", .c_str(), pStateName);
		string fullName = aLib->getName() + "::" + pStateName;
		if (RESOURCEMANAGER->hasState(fullName))
			NAU_THROW("Mat Lib %s\nState %s is already defined", aLib->getName().c_str(), pStateName);


		IState *s = IState::create();
		s->setName(fullName);

		loadState(pElem,aLib,NULL,s);
		RESOURCEMANAGER->addState(s);
	}
}

/* -----------------------------------------------------------------------------
SHADERS

	<shaders>
		<shader name="perpixel-color" ps="../shaders/perpixel-color.frag" vs="../shaders/perpixel-color.vert" />
		...
	</shaders>


	
-----------------------------------------------------------------------------*/
void 
ProjectLoader::loadMatLibShaders(TiXmlHandle hRoot, MaterialLib *aLib, std::string path)
{
	TiXmlElement *pElem;
	pElem = hRoot.FirstChild ("shaders").FirstChild ("shader").Element();
	for ( ; 0 != pElem; pElem=pElem->NextSiblingElement()) {
		const char *pProgramName = pElem->Attribute ("name");

		if (0 == pProgramName) {
			NAU_THROW("Mat Lib %s\nShader has no name", aLib->getName().c_str());
		}

		SLOG("Shader : %s", pProgramName);

		sprintf(s_pFullName, "%s::%s", aLib->getName().c_str(), pProgramName); 
		if (RESOURCEMANAGER->hasProgram(s_pFullName))
			NAU_THROW("Mat Lib %s\nShader %s is already defined", aLib->getName().c_str(), pProgramName);


		const char *pVSFile = pElem->Attribute ("vs");
		const char *pPSFile = pElem->Attribute ("ps");
		const char *pGSFile = pElem->Attribute("gs");
		const char *pTCFile = pElem->Attribute("tc");
		const char *pTEFile = pElem->Attribute("te");
		const char *pCSFile = pElem->Attribute("cs");

		//if ((0 == pCSFile) && (0 == pVSFile || (0 == pPSFile && 0 == pGSFile))) {
		if (0 == pCSFile && 0 == pVSFile) {
				NAU_THROW("Mat Lib %s\nShader %s missing files", aLib->getName().c_str(), pProgramName);
		}

		if (0 != pCSFile && (0 != pVSFile || 0 != pPSFile || 0 != pGSFile || 0 != pTEFile || 0 != pTCFile)) 
			NAU_THROW("Mat Lib %s\nShader %s: Mixing Compute Shader with other shader stages",aLib->getName().c_str(), pProgramName);

			
		if (pCSFile) {
			if (APISupport->apiSupport(IAPISupport::COMPUTE_SHADER)) {
				IProgram *aShader = RESOURCEMANAGER->getProgram(s_pFullName);
				std::string CSFileName(File::GetFullPath(path, pCSFile));
				if (!File::Exists(CSFileName))
					NAU_THROW("Mat Lib %s\nShader file %s does not exist", aLib->getName().c_str(), pCSFile);
				SLOG("Program %s", pProgramName);

				aShader->loadShader(IProgram::COMPUTE_SHADER, File::GetFullPath(path, pCSFile));
				aShader->linkProgram();
				SLOG("Linker: %s", aShader->getProgramInfoLog().c_str());
			}
			else {
				NAU_THROW("Mat Lib %s\nShader %s: Compute shader is not allowed with OpenGL < 4.3", aLib->getName().c_str(), pProgramName);
			}
		}
		else {
			std::string GSFileName;
			std::string FSFileName;
			std::string TEFileName, TCFileName;

			std::string VSFileName(File::GetFullPath(path,pVSFile));
			if (!File::Exists(VSFileName))
				NAU_THROW("Mat Lib %s\nShader file %s does not exist", aLib->getName().c_str(), pVSFile);

			if (pGSFile) {
				if (APISupport->apiSupport(IAPISupport::GEOMETRY_SHADER)) {
					GSFileName = File::GetFullPath(path, pGSFile);
					if (!File::Exists(GSFileName))
						NAU_THROW("Mat Lib %s\nShader file %s does not exist", aLib->getName().c_str(), pGSFile);
				}
				else {
					NAU_THROW("Mat Lib %s\nShader %s: Geometry Shader shader is not allowed with OpenGL < 3.2", aLib->getName().c_str(), pProgramName);
				}
			}
			if (pPSFile) {
				FSFileName = File::GetFullPath(path,pPSFile);
				if (!File::Exists(FSFileName))
					NAU_THROW("Mat Lib %s\nShader file %s does not exist", aLib->getName().c_str(), FSFileName.c_str());
			}
			if (pTEFile) {
				if (APISupport->apiSupport(IAPISupport::TESSELATION_SHADERS)) {
					TEFileName = File::GetFullPath(path, pTEFile);
					if (!File::Exists(TEFileName))
						NAU_THROW("Mat Lib %s\nShader file %s does not exist", aLib->getName().c_str(), TEFileName.c_str());
				}
				else {
					NAU_THROW("Mat Lib %s\nShader %s: Tesselation shaders are not allowed with OpenGL < 4.0", aLib->getName().c_str(), pProgramName);
				}
			}
			if (pTCFile) {
				if (APISupport->apiSupport(IAPISupport::TESSELATION_SHADERS)) {

					TCFileName = File::GetFullPath(path, pTCFile);
					if (!File::Exists(TCFileName))
						NAU_THROW("Mat Lib %s\nShader file %s does not exist", aLib->getName().c_str(), TCFileName.c_str());
				}
				else {
					NAU_THROW("Mat Lib %s\nShader %s: Tesselation shaders are not allowed with OpenGL < 4.0", aLib->getName().c_str(), pProgramName);
				}
			}

	
			SLOG("Program %s", pProgramName);
			IProgram *aShader = RESOURCEMANAGER->getProgram (s_pFullName);
			aShader->loadShader(IProgram::VERTEX_SHADER, File::GetFullPath(path,pVSFile));
			SLOG("Shader file %s - %s",pVSFile, aShader->getShaderInfoLog(IProgram::VERTEX_SHADER).c_str());
			if (pPSFile) {
				aShader->loadShader(IProgram::FRAGMENT_SHADER, File::GetFullPath(path,pPSFile));
				SLOG("Shader file %s - %s",pPSFile, aShader->getShaderInfoLog(IProgram::FRAGMENT_SHADER).c_str());
			}
			if (pTCFile) {
				aShader->loadShader(IProgram::TESS_CONTROL_SHADER, File::GetFullPath(path,pTCFile));
				SLOG("Shader file %s - %s",pTCFile, aShader->getShaderInfoLog(IProgram::TESS_CONTROL_SHADER).c_str());
			}
			if (pTEFile) {
				aShader->loadShader(IProgram::TESS_EVALUATION_SHADER, File::GetFullPath(path,pTEFile));
				SLOG("Shader file %s - %s",pTEFile, aShader->getShaderInfoLog(IProgram::TESS_EVALUATION_SHADER).c_str());
			}
			if (pGSFile) {
				aShader->loadShader(IProgram::GEOMETRY_SHADER, File::GetFullPath(path,pGSFile));
				SLOG("Shader file %s - %s",pGSFile, aShader->getShaderInfoLog(IProgram::GEOMETRY_SHADER).c_str());
			}
			aShader->linkProgram();
			SLOG("Linker: %s", aShader->getProgramInfoLog().c_str());
		}
	}
}

/* -----------------------------------------------------------------------------
MATERIALCOLOR

	<color>
		<ambient r="0.2" g="0.2" b="0.2" a="1.0" />
		<diffuse r="0.8" g="0.8" b="0.8" a="0.8" />
		<specular r="0.0" g="0.0" b="0.0" a="1.0" />
		<emission r="0.0" g="0.0" b="0.0" a="1.0" />
		<shininess value="0" />
	</color>


	
-----------------------------------------------------------------------------*/
void 
ProjectLoader::loadMaterialColor(TiXmlHandle handle, MaterialLib *aLib, std::shared_ptr<Material> &aMat)
{
	TiXmlElement *pElemAux;
	pElemAux = handle.FirstChild ("color").Element();
	if (!pElemAux)
		return;

	std::vector<std::string> excluded;
	std::string s = aMat->getName() + ": color";
	ColorMaterial *cm = &(aMat->getColor());
	readChildTags(s, (AttributeValues *)cm, ColorMaterial::Attribs, excluded, pElemAux);
}


/* -----------------------------------------------------------------------------
MATERIAL IMAGE TEXTURE

			<imageTextures>
				<imageTexture UNIT=0  texture="texName">
					<ACCESS value="WRITE_ONLY" />
					<LEVEL value=0 />
				<imageTexture />
			</imageTextures>

All fields are optional except name UNIT and texture
texture refers to a previously defined texture or render target 

The name can refer to a texture in another lib, in which case the syntax is lib_name::tex_name
	
-----------------------------------------------------------------------------*/

void 
ProjectLoader::loadMaterialImageTextures(TiXmlHandle handle, MaterialLib *aLib, std::shared_ptr<Material> &aMat)
{
	TiXmlElement *pElemAux;
	pElemAux = handle.FirstChild ("imageTextures").FirstChild ("imageTexture").Element();
	for ( ; 0 != pElemAux; pElemAux = pElemAux->NextSiblingElement()) {
		//const char *pTextureName = pElemAux->GetText();

		if (!APISupport->apiSupport(IAPISupport::IMAGE_TEXTURE))
			NAU_THROW("MatLib %s\nMaterial %s: Image Texture Not Supported with OpenGL Version %d", aLib->getName().c_str(), aMat->getName().c_str(), APISupport->getVersion());

		int unit;
		if (TIXML_SUCCESS != pElemAux->QueryIntAttribute ("UNIT", &unit)) {
			NAU_THROW("MatLib %s\nMaterial %s: Image Texture has no unit", aLib->getName().c_str(),  aMat->getName().c_str());
		}

		const char *pTextureName = pElemAux->Attribute ("texture");
		if (0 == pTextureName) {
			NAU_THROW("MatLib %s\nMaterial %s: Texture has no name in image texture", aLib->getName().c_str(),  aMat->getName().c_str());
		}
		if (!strchr(pTextureName, ':') )
			sprintf(s_pFullName, "%s::%s", aLib->getName().c_str(), pTextureName);
		else
			sprintf(s_pFullName, "%s", pTextureName);
		if (!RESOURCEMANAGER->hasTexture(s_pFullName))
			NAU_THROW("MatLib %s\nMaterial %s: Texture %s in image texture is not defined", aLib->getName().c_str(),  aMat->getName().c_str(), pTextureName);
		
		ITexture *t = RESOURCEMANAGER->getTexture(s_pFullName);
		int texID = t->getPropi(ITexture::ID);

		aMat->attachImageTexture(t->getLabel(), unit, texID);
		// Reading Image ITexture Attributes

		std::vector<std::string> excluded;
		std::string s = aMat->getName() + ": image texture unit " + TextUtil::ToString(unit);
		readChildTags(s, (AttributeValues *)aMat->getImageTexture(unit), IImageTexture::Attribs, excluded, pElemAux);
	}
}

/* -----------------------------------------------------------------------------
MATERIAL BUFFERS

<buffers>
	<buffer name="bla" />
		<TYPE="SHADER_STORAGE" />
		<BINDING_POINT value="1">
	</buffer>
</buffers>

All fields are optional. By default the type is SHADER_STORAGE, and binding point is 0

The name can refer to a buffer in another lib, in which case the syntax is lib_name::buffer_name

-----------------------------------------------------------------------------*/

void
ProjectLoader::loadMaterialBuffers(TiXmlHandle handle, MaterialLib *aLib, std::shared_ptr<Material> &aMat)
{
	TiXmlElement *pElemAux;
	pElemAux = handle.FirstChild("buffers").FirstChild("buffer").Element();
	for (; 0 != pElemAux; pElemAux = pElemAux->NextSiblingElement()) {
		const char *pName = pElemAux->Attribute("name");
		if (0 == pName) {
			NAU_THROW("MatLib %s\nMaterial %s: Buffer has no name", aLib->getName().c_str(), aMat->getName().c_str());
		}
		if (!strchr(pName, ':'))
			sprintf(s_pFullName, "%s::%s", aLib->getName().c_str(), pName);
		else
			sprintf(s_pFullName, "%s", pName);
		if (!RESOURCEMANAGER->hasBuffer(s_pFullName))
			NAU_THROW("MatLib %s\nMaterial %s: Buffer %s is not defined", aLib->getName().c_str(), aMat->getName().c_str(), pName);

		IBuffer *buffer = RESOURCEMANAGER->getBuffer(s_pFullName);
		IMaterialBuffer *imb = IMaterialBuffer::Create(buffer);

		std::vector<std::string> excluded;
		std::string s = aMat->getName() + ": buffers";
		readChildTags(s, (AttributeValues *)imb, IMaterialBuffer::Attribs, excluded, pElemAux);

		aMat->attachBuffer(imb);	
	}
}

/* -----------------------------------------------------------------------------
MATERIALTEXTURE

		<textures>
			<texture name="shadowMap"  UNIT="0">
				<COMPARE_MODE value="COMPARE_REF_TO_TEXTURE" />
				<COMPARE_FUNC value="LEQUAL" />
				<MIN_FILTER value="LINEAR" />
				<MAG_FILTER value="LINEAR" />
			</texture>	
			...
		</textures>

All fields are optional except name and UNIT
name refers to a previously defined texture or render target 

The name can refer to a texture in another lib, in which case the syntax is lib_name::tex_name
	
-----------------------------------------------------------------------------*/

void 
ProjectLoader::loadMaterialTextures(TiXmlHandle handle, MaterialLib *aLib, std::shared_ptr<Material> &aMat)
{
	TiXmlElement *pElemAux;
	pElemAux = handle.FirstChild ("textures").FirstChild ("texture").Element();
	for ( ; 0 != pElemAux; pElemAux = pElemAux->NextSiblingElement()) {
		//const char *pTextureName = pElemAux->GetText();

		const char *pTextureName = pElemAux->Attribute ("name");
		if (0 == pTextureName) {
			NAU_THROW("MatLib %s\nMaterial %s\nTexture has no name", aLib->getName().c_str(),  aMat->getName().c_str());
		}
		
		int unit;
		if (TIXML_SUCCESS != pElemAux->QueryIntAttribute ("UNIT", &unit)) {
			NAU_THROW("MatLib %s\nMaterial %s\nTexture has no unit", aLib->getName().c_str(),  aMat->getName().c_str());
		}

		if (!strchr(pTextureName, ':') )
			sprintf(s_pFullName, "%s::%s", aLib->getName().c_str(), pTextureName);
		else
			sprintf(s_pFullName, "%s", pTextureName);
		if (!RESOURCEMANAGER->hasTexture(s_pFullName))
			NAU_THROW("MatLib %s\nMaterial %s\nTexture %s is not defined", aLib->getName().c_str(),  aMat->getName().c_str(), pTextureName);
			

		aMat->attachTexture (unit, s_pFullName);

		// Reading ITexture Sampler Attributes
		std::vector<std::string> excluded;
		readChildTags(pTextureName, (AttributeValues *)aMat->getTextureSampler(unit), ITextureSampler::Attribs, excluded, pElemAux);
	}
}


/* -----------------------------------------------------------------------------
MATERIALSHADER

	<shader name="perpixel-color-shadow" >
		<values>
			<valueof uniform="lightPosition" type="LIGHT" context="Sun" component="POSITION" /> 
		</values>
	</shader>
	
-----------------------------------------------------------------------------*/
void 
ProjectLoader::loadMaterialShader(TiXmlHandle handle, MaterialLib *aLib, std::shared_ptr<Material> &aMat)
{
	TiXmlElement *pElemAux, *pElemAux2;

	pElemAux = handle.FirstChild ("shader").Element();
	if (0 != pElemAux) {

		const char *pShaderName = pElemAux->Attribute("name");

		//pElemAux2 = hShader.FirstChild ("name").Element();
		//const char *pShaderName = pElemAux2->GetText();

		if (0 == pShaderName) {
			NAU_THROW("MatLib %s\nMaterial %s\nShader has no name", aLib->getName().c_str(), aMat->getName().c_str());
		}
		sprintf(s_pFullName, "%s::%s",aLib->getName().c_str(),pShaderName);
		if (!RESOURCEMANAGER->hasProgram(s_pFullName))
			NAU_THROW("MatLib %s\nMaterial %s\nShader %s is not defined", aLib->getName().c_str(), aMat->getName().c_str(), pShaderName);

		
		aMat->attachProgram (s_pFullName);
		aMat->clearProgramValues();

		TiXmlHandle hShader (pElemAux);
		pElemAux2 = hShader.FirstChild ("values").FirstChild ("valueof").Element();
		for ( ; 0 != pElemAux2; pElemAux2 = pElemAux2->NextSiblingElement()) {
			const char *pUniformName = pElemAux2->Attribute ("uniform");
			const char *pComponent = pElemAux2->Attribute ("component");
			const char *pContext = pElemAux2->Attribute("context");
			const char *pType = pElemAux2->Attribute("type");
			const char *pBlock = pElemAux2->Attribute("block");
			//const char *pId = pElemAux2->Attribute("id");

			if (0 == pUniformName) {
				NAU_THROW("MatLib %s\nMaterial %s\nNo uniform name", 
					aLib->getName().c_str(), aMat->getName().c_str());
			}
			if (0 == pType) {
				NAU_THROW("MatLib %s\nMaterial %s\nNo type found for uniform %s", 
					aLib->getName().c_str(), aMat->getName().c_str(), pUniformName);
			}
			if (0 == pContext) {
				NAU_THROW("MatLib %s\nMaterial %s\nNo context found for uniform %s", 
					aLib->getName().c_str(), aMat->getName().c_str(), pUniformName);
			}
			if (0 == pComponent) {
				NAU_THROW("MatLib %s\nMaterial %s\nNo component found for uniform %s", 
					aLib->getName().c_str(), aMat->getName().c_str(), pUniformName);
			}
			std::string message;
			validateObjectTypeAndComponent(pType, pComponent, &message);
			if ( message != "")
				NAU_THROW("MatLib %s\nMaterial %s\nUniform %s is not valid\n%s",
					aLib->getName().c_str(), aMat->getName().c_str(), pUniformName, message.c_str());

			//if (!NAU->validateShaderAttribute(pType, pContext, pComponent))
			//	NAU_THROW("MatLib %s\nMaterial %s\nUniform %s is not valid", 
			//		aLib->getName().c_str(), aMat->getName().c_str(), pUniformName);

			int id = 0;
			if ((strcmp(pContext,"CURRENT") == 0) && 
						((strcmp(pType,"LIGHT") == 0) || (0 == strcmp(pType,"TEXTURE_BINDING")) || (0 == strcmp(pType,"IMAGE_TEXTURE"))) 
						&&  (0 != strcmp(pComponent,"COUNT"))) {
				if (TIXML_SUCCESS != pElemAux2->QueryIntAttribute ("id", &id))
					NAU_THROW("MatLib %s\nMaterial %s\nUniform %s - id is required for type %s ", 
						aLib->getName().c_str(), aMat->getName().c_str(), pUniformName, pType);
				if (id < 0)
					NAU_THROW("MatLib %s\nMaterial %s\nUniform %s - id must be non negative", 
						aLib->getName().c_str(), aMat->getName().c_str(), pUniformName);

				if (0 == strcmp(pType, "TEXTURE_BINDING") && !aMat->getTexture(id)) {
					SLOG("MatLib %s\nMaterial %s\nUniform %s - id should refer to an assigned texture unit", 
						aLib->getName().c_str(), aMat->getName().c_str(), pUniformName);
				}
				else if (0 == strcmp(pType, "IMAGE_TEXTURE") && !aMat->getImageTexture(id)) {
					SLOG("MatLib %s\nMaterial %s\nUniform %s - id should refer to an assigned image texture unit", 
						aLib->getName().c_str(), aMat->getName().c_str(), pUniformName);
				}
			}
			std::string s(pType);

			if (s == "TEXTURE" && strcmp(pContext, "CURRENT")) {
				sprintf(s_pFullName, "%s::%s", aLib->getName().c_str(),pContext);
				if (!RESOURCEMANAGER->hasTexture(s_pFullName)) {
					NAU_THROW("MatLib %s\nMaterial %s\nTexture %s is not defined", 
						aLib->getName().c_str(), aMat->getName().c_str(), s_pFullName);
				}
				else {
					if (pBlock)
						aMat->addProgramBlockValue(pBlock, pUniformName, 
							ProgramBlockValue(pUniformName, pBlock, pType, s_pFullName, pComponent, id));
					else
						aMat->addProgramValue(pUniformName, 
							ProgramValue(pUniformName, pType, s_pFullName, pComponent, id));
				}
			}

			else if (s == "CAMERA" && strcmp(pContext, "CURRENT")) {
				// Must consider that a camera can be defined internally in a pass, example:lightcams
				/*if (!RENDERMANAGER->hasCamera(pContext))
					NAU_THROW("Camera %s is not defined in the project file", pContext);*/
					if (pBlock)
						aMat->addProgramBlockValue(pBlock, pUniformName, 
							ProgramBlockValue(pUniformName, pBlock, pType, pContext, pComponent, id));
					else
						aMat->addProgramValue(pUniformName, 
							ProgramValue(pUniformName, pType, pContext, pComponent, id));
			}
			else if (s == "LIGHT" && strcmp(pContext, "CURRENT")) {
				if (!RENDERMANAGER->hasLight(pContext))
					NAU_THROW("MatLib %s\nMaterial %s\nUniform %s: Light %s is not defined in the project file", 
						aLib->getName().c_str(), aMat->getName().c_str(), pUniformName, pContext);
				if (pBlock)
					aMat->addProgramBlockValue(pBlock, pUniformName, 
						ProgramBlockValue(pUniformName, pBlock, pType, pContext, pComponent, id));
				else
					aMat->addProgramValue(pUniformName, 
						ProgramValue(pUniformName, pType, pContext, pComponent, id));

			}
			else {
				if (pBlock)
					aMat->addProgramBlockValue(pBlock, pUniformName, 
						ProgramBlockValue(pUniformName, pBlock, pType, pContext, pComponent, id));
				else
					aMat->addProgramValue(pUniformName, 
						ProgramValue(pUniformName, pType, pContext, pComponent, id));
			}
		}
	}
	aMat->checkProgramValuesAndUniforms();
}

/* -----------------------------------------------------------------------------
MATERIALSTATE


	<state name="bla" />

where bla is previously defined in the mat lib.
	

-----------------------------------------------------------------------------*/

void 
ProjectLoader::loadMaterialState(TiXmlHandle handle, MaterialLib *aLib, std::shared_ptr<Material> &aMat)
{
	TiXmlElement *pElemAux;

	pElemAux = handle.FirstChild("state").Element();
	if (0 != pElemAux) {
		const char *pStateName = pElemAux->Attribute ("name");
		//definition by ref
		if (0 == pStateName) {
			NAU_THROW("MatLib %s\nMaterial %s\nState requires a name and must be previously defined", aLib->getName().c_str(), aMat->getName().c_str());
		}
		else {
			std::string fullName = aLib->getName() + "::" + pStateName;
			if (!RESOURCEMANAGER->hasState(fullName))
				NAU_THROW("MatLib %s\nMaterial %s\nState %s is not defined", aLib->getName().c_str(), aMat->getName().c_str(), pStateName);

			aMat->setState(RESOURCEMANAGER->getState(fullName));
		}
	}
	else {
		std::string fullName = aLib->getName() + "::" + aMat->getName();
		aMat->setState(RESOURCEMANAGER->createState(fullName));
	}
}

/* -----------------------------------------------------------------------------
MATERIAL LIBS     

	<?xml version="1.0" ?>
	<materiallib name="CubeMap">
		<shaders>
			... // see loadMatLibShaders
		</shaders>
		<textures>
			... // see loadMatLibTextures
		</textures>
		<states>
			... // see loadMatLibStates
		</states>
		<materials>
			<material name = "bla">
				<color>
					... see loadMaterialColor
				</color>
				<textures>
					... see loadMaterialTextures
				</textures>
				<state>
					... see loadMaterialState
				</state>
				<shader>
					... see loadMaterialShader
				</shader>
			</material>
		</materials>
	...
	</materiallib>


-----------------------------------------------------------------------------*/

void 
ProjectLoader::loadMatLib (std::string file)
{
	std::string path = File::GetPath(file);
	//std::map<std::string,IState *> states;

	TiXmlDocument doc (file.c_str());
	bool loadOkay = doc.LoadFile();

	MaterialLib *aLib = 0;

	if (!loadOkay) 
		NAU_THROW("Parsing Error -%s- Line(%d) Column(%d) in file: %s", doc.ErrorDesc(), doc.ErrorRow(), doc.ErrorCol(),file.c_str());

	TiXmlHandle hDoc (&doc);
	TiXmlHandle hRoot (0);
	TiXmlElement *pElem;

	{ //root
		pElem = hDoc.FirstChildElement().Element();
		if (0 == pElem) 
			NAU_THROW("Parse Error in mat lib file %s", file.c_str());
		hRoot = TiXmlHandle (pElem);
	}

	pElem = hRoot.Element();
	const char* pName = pElem->Attribute ("name");

	if (0 == pName) 
		NAU_THROW("Material lib has no name in file %s", file.c_str());

	SLOG("Material Lib Name: %s", pName);

	aLib = MATERIALLIBMANAGER->getLib (pName);

	std::string aux = s_File;
	s_File = file;

	loadMatLibRenderTargets(hRoot, aLib, path);
	loadMatLibTextures(hRoot, aLib,path);
	loadMatLibStates(hRoot, aLib);
	loadMatLibShaders(hRoot, aLib, path);
	loadMatLibBuffers(hRoot, aLib, path);
	pElem = hRoot.FirstChild ("materials").FirstChild ("material").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {

		TiXmlHandle handle (pElem);

		const char *pMaterialName = pElem->Attribute ("name");

		if (0 == pMaterialName) 
			NAU_THROW("MatLib %s\nMaterial has no name", pName);

		if (MATERIALLIBMANAGER->hasMaterial(pName, pMaterialName))
			NAU_THROW("MatLib %s\nMaterial %s redefinition", pName, pMaterialName);

		SLOG("Material: %s", pMaterialName);

		//Material *mat = new Material;
		//mat->setName (pMaterialName);
		std::shared_ptr<Material> &mat = MATERIALLIBMANAGER->createMaterial(pName,pMaterialName);

		loadMaterialColor(handle,aLib,mat);
		loadMaterialTextures(handle,aLib,mat);
		loadMaterialImageTextures(handle, aLib, mat);
		loadMaterialShader(handle,aLib,mat);
		loadMaterialState(handle,aLib,mat);
		loadMaterialBuffers(handle, aLib, mat);

		//aLib->addMaterial (mat);
	}
	s_File = aux;
}

/* ----------------------------------------------------------------
Specification of the debug:

Configuration of attributes are in projectloaderdebuglinler.cpp
see void initGLInterceptFunctions() and initAttributeListMaps() for
details.

Generic values:
"bool" means a true or false string
"uint" means a a positive integer string
"string" means any string goes

glilog attribute in debug tag is an optional value, if not present
it'll default to true, creates glilog.log which output's glIntercept's
errors.

	<debug glilog="bool">
		<functionlog>
			... see loadDebugFunctionlog
		</functionlog>
		<logperframe>
			... see loadDebugLogperframe
		</logperframe>
		<errorchecking>
			... see loadDebugErrorchecking
		</errorchecking>
		<imagelog>
			... see loadDebugImagelog
		</imagelog>
		<shaderlog>
			... see loadDebugShaderlog
		</shaderlog>
		<displaylistlog>
			... see loadDebugDisplaylistlog
		</displaylistlog>
		<framelog>
			... see loadDebugFramelog
		</framelog>
		<timerlog>
			... see loadDebugTimerlog
		</timerlog>
		<plugins>
			<plugin>
				... see loadDebugPlugins
			</plugin>
			...
		</plugins>
	</assets>
----------------------------------------------------------------- */
//void
//ProjectLoader::loadDebug (TiXmlHandle &hRoot)
//{
//#ifdef GLINTERCEPTDEBUG
//	
//	TiXmlElement *pElem;
//	TiXmlHandle handle (hRoot.FirstChild ("debug").Element());
//	bool startGliLog = true;
//
//	if (handle.Element()){
//		pElem = handle.Element();
//
//		initGLInterceptFunctions();
//	
//		if (pElem->Attribute("glilog")){
//			pElem->QueryBoolAttribute("glilog",&startGliLog);
//		}
//		if (startGliLog){
//			startGlilog();
//		}
//
//		activateGLI();
//
//		//TiXmlElement *pElem;
//		loadDebugFunctionlog(handle);
//		loadDebugLogperframe(handle);
//		loadDebugErrorchecking(handle);
//		loadDebugImagelog(handle);
//		loadDebugShaderlog(handle);
////		loadDebugDisplaylistlog(handle);
//		loadDebugFramelog(handle);
//		loadDebugTimerlog(handle);
//		loadDebugPlugins(handle);
//		startGLIConfiguration();
//	}
//#endif
//}

/* ----------------------------------------------------------------
Specification of the functionlog:

		<functionlog>
			<enabled value="bool"/> (Automatic)
			<logxmlformat value="bool"/> (REMOVED)
			<logflush value="bool"/> (REMOVED)
			<logmaxframeloggingenabled value="bool"/>
			<logmaxnumlogframes value="uint"/>
			<logpath value="bool"/>
			<logname value="bool"/>
			<xmlformat>
				... see loadDebugFunctionlogXmlFormat (REMOVED)
			</xmlformat>
		</functionlog>
----------------------------------------------------------------- */
//void
//ProjectLoader::loadDebugFunctionlog (TiXmlHandle &hRoot){
//	TiXmlHandle handle (hRoot.FirstChild ("functionlog").Element());
//
//#ifdef GLINTERCEPTDEBUG
//	if (handle.Element()){
//		bool defaultValue = true;
//		void *functionSetPointer = getGLIFunction("enabled", "functionlog");
//		useGLIFunction(functionSetPointer, &defaultValue);
//	}
//#endif
//
//	loadDebugConfigData(handle,"functionlog");
//
//	//loadDebugFunctionlogXmlFormat(hRoot);
//}



/* ----------------------------------------------------------------
Specification of the xmlformat: (REMOVED)

logxslfile requires an existing .xsl

logxslbasedir will specify where the logxslfile exists

with GL intercept 1.2 installed the default locations should be
	xslfile = "gliIntercept_DHTML2.xsl"
	xslfilesource = "C:\Program Files\GLIntercept_1_2_0\XSL"

			<xmlformat>
				<logxslfile value="xslfile"/>
				<logxslbasedir value="xslfilesource"/>
			</xmlformat>
----------------------------------------------------------------- */
//void
//ProjectLoader::loadDebugFunctionlogXmlFormat (TiXmlHandle &hRoot){
//	TiXmlHandle handle (hRoot.FirstChild ("xmlformat").Element());
//
//	loadDebugConfigData(handle,"functionlogxmlformat");
//}

/* ----------------------------------------------------------------
Specification of the logperframe:

in logFrameKeys each item uses a string value, for example 
	<item value="ctrl"/>
	<item value="F"/>
will enable log the frame using ctrl+F

		<logperframe>
			<logperframe value="bool"/>
			<logoneframeonly value="bool"/>
			<logframekeys>
				<item value="key"/>
				<item value="key"/>
				...
			</logframekeys>
		</logperframe>
----------------------------------------------------------------- */
//void
//ProjectLoader::loadDebugLogperframe (TiXmlHandle &hRoot){
//	TiXmlHandle handle (hRoot.FirstChild ("logperframe").Element());
//
//	loadDebugConfigData(handle,"logperframe");
//}

/* ----------------------------------------------------------------
Specification of the errorchecking:

		<errorchecking>
			<errorgetopenglchecks value="bool"/>
			<errorthreadchecking value="bool"/>
			<errorbreakonerror value="bool"/>
			<errorlogonerror value="bool"/>
			<errorextendedlogerror value="bool"/>
			<errordebuggererrorlog value="bool"/>
		</errorchecking>
----------------------------------------------------------------- */
//void
//ProjectLoader::loadDebugErrorchecking (TiXmlHandle &hRoot){
//	TiXmlHandle handle (hRoot.FirstChild ("errorchecking").Element());
//
//	loadDebugConfigData(handle,"errorchecking");
//
//}

/* ----------------------------------------------------------------
Specification of the imagelog:

imagesavepng, imagesavetga and imagesavejpg can be used simultaneosly

		<imagelog>
			<imagerendercallstatelog value="bool">
			<imagesavepng value="bool"/>
			<imagesavetga value="bool"/>
			<imagesavejpg value="bool"/>
			<imageflipxaxis value="bool"/>
			<imagecubemaptile value="bool"/>
			<imagesave1d value="bool"/>
			<imagesave2d value="bool"/>
			<imagesave3d value="bool"/>
			<imagesavecube value="bool"/>
			<imagesavepbuffertex value="bool"/>
			<imageicon>
				... see loadDebugImagelogimageicon
			</imageicon>
		</imagelog>
----------------------------------------------------------------- */
//void
//ProjectLoader::loadDebugImagelog (TiXmlHandle &hRoot){
//	TiXmlHandle handle (hRoot.FirstChild ("imagelog").Element());
//
//	loadDebugConfigData(handle,"imagelog");
//
//
//	loadDebugImagelogimageicon(handle);
//}

/* ----------------------------------------------------------------
Specification of the imageicon:

imageiconformat is the format of the save icon images (TGA,PNG or JPG)
only one format at a time

			<imageicon>
				<imagesaveicon value="bool"/>
				<imageiconsize value="uint"/>
				<imageiconformat value="png"/>
			</imageicon>
----------------------------------------------------------------- */
//void
//ProjectLoader::loadDebugImagelogimageicon (TiXmlHandle &hRoot){
//	TiXmlHandle handle (hRoot.FirstChild ("imageicon").Element());
//
//	loadDebugConfigData(handle,"imagelogimageicon");
//}

/* ----------------------------------------------------------------
Specification of the shaderlog:

		<shaderlog>
			<enabled value="bool"/> (Automatic)
			<shaderrendercallstatelog value="bool"/>
			<shaderattachlogstate value="bool"/>
			<shadervalidateprerender value="bool"/>
			<shaderloguniformsprerender value="bool"/>
		</shaderlog>
----------------------------------------------------------------- */
//void
//ProjectLoader::loadDebugShaderlog (TiXmlHandle &hRoot){
//	TiXmlHandle handle(hRoot.FirstChild("shaderlog").Element());
//
//#ifdef GLINTERCEPTDEBUG
//	if (handle.Element()){
//		bool defaultValue = true;
//		void *functionSetPointer = getGLIFunction("enabled", "shaderlog");
//		useGLIFunction(functionSetPointer, &defaultValue);
//	}
//#endif
//
//	loadDebugConfigData(handle,"shaderlog");
//}



/* ----------------------------------------------------------------
Specification of the framelog:

According to the original gliConfig regarding framestencilcolors:
When saving the stencil buffer, it can be useful to save the buffer with color codes.
(ie stencil value 1 = red) This array supplies index color pairs for each stencil 
value up to 255. The indices must be in order and the colors are in the format 
AABBGGRR. If an index is missing, it will take the value of the index as the color.
(ie. stencil index 128 = (255, 128,128,128) = greyscale values)

		<framelog>
			<enabled value="bool"/> (Automatic)
			<frameimageformat value="string"/>
			<framestencilcolors>
				<item value="uint"/>
				<item value="uint"/>
				...
			</frameStencilColors>
			<frameprecolorsave value="bool"/>
			<framepostcolorsave value="bool"/>
			<framediffcolorsave value="bool"/>
			<framepredepthsave value="bool"/>
			<framepostdepthsave value="bool"/>
			<framediffdepthsave value="bool"/>
			<frameprestencilsave value="bool"/>
			<framepoststencilsave value="bool"/>
			<framediffstencilsave value="bool"/>
			<frameAdditionalRenderCalls> (REMOVED, UNSAFE)
				<item value="string"/>
				<item value="string"/>
				...
			</frameAdditionalRenderCalls>
			<frameicon>
				<frameiconsave value="bool"/>
				<frameiconsize value="uint"/>
				<frameiconimageformat value="png"/>
			</frameicon>
			<framemovie>
				<framemovieenabled value="bool"/>
				<framemoviewidth value="uint"/>
				<framemovieheight value="uint"/>
				<framemovierate value="uint"/>
				<frameMovieCodecs>
					<item value="string"/>
					<item value="string"/>
					...
				</frameMovieCodecs>
			</framemovie>
		</framelog>

----------------------------------------------------------------- */
//void
//ProjectLoader::loadDebugFramelog (TiXmlHandle &hRoot){
//	TiXmlHandle handle (hRoot.FirstChild ("framelog").Element());
//
//#ifdef GLINTERCEPTDEBUG
//	if (handle.Element()){
//		bool defaultValue = true;
//		void *functionSetPointer = getGLIFunction("enabled", "framelog");
//		useGLIFunction(functionSetPointer, &defaultValue);
//	}
//#endif
//
//	loadDebugConfigData(handle,"framelog");
//
//	
//	loadDebugFramelogFrameicon(handle);
//	loadDebugFramelogFramemovie(handle);
//}

/* ----------------------------------------------------------------
Specification of the frameicon:

frameiconimageformat is tee format of the save icon images (TGA,PNG or JPG)
only one format at a time

			<frameicon>
				<frameiconsave value="bool"/>
				<frameiconsize value="uint"/>
				<frameiconimageformat value="png"/>
			</frameicon>
----------------------------------------------------------------- */
//void
//ProjectLoader::loadDebugFramelogFrameicon (TiXmlHandle &hRoot){
//	TiXmlHandle handle (hRoot.FirstChild ("frameicon").Element());
//
//	loadDebugConfigData(handle,"framelogframeicon");
//
//}
//

/* ----------------------------------------------------------------
Specification of the framemovie:

			<framemovie>
				<framemovieenabled value="bool"/> (Automatic)
				<framemoviewidth value="uint"/>
				<framemovieheight value="uint"/>
				<framemovierate value="uint"/>
				<frameMovieCodecs>
					<item value="string"/>
					<item value="string"/>
					...
				</frameMovieCodecs>
			</framemovie>
----------------------------------------------------------------- */
//void
//ProjectLoader::loadDebugFramelogFramemovie (TiXmlHandle &hRoot){
//	TiXmlHandle handle (hRoot.FirstChild ("framemovie").Element());
//
//#ifdef GLINTERCEPTDEBUG
//	if (handle.Element()){
//		bool defaultValue = true;
//		void *functionSetPointer = getGLIFunction("framemovieenabled", "framelogframemovie");
//		useGLIFunction(functionSetPointer, &defaultValue);
//	}
//#endif
//
//	loadDebugConfigData(handle,"framelogframemovie");
//
//}

/* ----------------------------------------------------------------
Specification of the timerlog:

		<timerlog>
			<enabled value="bool"/> (Automatic)
			<timerlogcutoff value="uint"/>
		</timerlog>
----------------------------------------------------------------- */
//void
//ProjectLoader::loadDebugTimerlog (TiXmlHandle &hRoot){
//	TiXmlHandle handle (hRoot.FirstChild ("timerlog").Element());
//
//#ifdef GLINTERCEPTDEBUG
//	if (handle.Element()){
//		bool defaultValue = true;
//		void *functionSetPointer = getGLIFunction("enabled", "timerlog");
//		useGLIFunction(functionSetPointer, &defaultValue);
//	}
//#endif
//
//	loadDebugConfigData(handle,"timerlog");
//
//}

/* ----------------------------------------------------------------
Specification of the plugins:

Some plugins can fit extra parameters (for example extension override)
the extra parameters should be placed as mentioned in the example.

Pay in mind that most GLIntercept plugins require the plugin name to
match a certain name, for example:

<plugin name="OpenGLShaderEdit" dll="GLShaderEdit/GLShaderEdit.dll"/>

You can usually see which names are required in the standard
gliConfig.ini, for example the plugin mentioned above in the ini becomes:

OpenGLShaderEdit = ("GLShaderEdit/GLShaderEdit.dll")

if you give the plugin a different name it may not work at all.


			<plugins>
				<plugin name="name string" dll="dllpath string">
					extraparameter1 = "extraparameter1 value";
					extraparameter2 = "extraparameter2 value";
					...
				<plugin>
				...
			</plugins>
----------------------------------------------------------------- */
//void
//ProjectLoader::loadDebugPlugins (TiXmlHandle &hRoot)
//{
//#ifdef GLINTERCEPTDEBUG
//	TiXmlElement *pElem;
//	TiXmlHandle handle (hRoot.FirstChild("plugins").Element());
//	
//	string name;
//	string dllpath;
//	string data="";
//
//	pElem = handle.FirstChild().Element();
//	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
//			pElem->QueryStringAttribute("name",&name);
//			pElem->QueryStringAttribute("dll",&dllpath);
//			if(pElem->GetText()){
//				data=pElem->GetText();
//			}
//			addPlugin(name.c_str(), dllpath.c_str(), data.c_str());
//		
//	}
//#endif
//}

/* ----------------------------------------------------------------
Helper function, reads sub attributes.
				<configcategory>
					<configattribute value="...">
					...
				</configcategory>
----------------------------------------------------------------- */
//void
//ProjectLoader::loadDebugConfigData (TiXmlHandle &handle, const char *configMapName)
//{
//#ifdef GLINTERCEPTDEBUG
//
//	TiXmlElement *pElem;
//	void *functionSetPointer;
//	pElem = handle.FirstChild().Element();
//	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
//		const char *functionName = pElem->Value();
//		functionSetPointer = getGLIFunction(functionName,configMapName);
//		switch (getGLIFunctionType(functionSetPointer))
//		{
//		case GLIEnums::FunctionType::BOOL:{
//				bool functionValue; 
//				pElem->QueryBoolAttribute("value",&functionValue);
//				useGLIFunction(functionSetPointer,&functionValue);
//				break;
//			}
//		case GLIEnums::FunctionType::INT:{
//				int functionValue; 
//				pElem->QueryIntAttribute("value",&functionValue);
//				useGLIFunction(functionSetPointer,&functionValue);
//				break;
//			}
//		case GLIEnums::FunctionType::UINT:{
//				unsigned int functionValue; 
//				pElem->QueryUnsignedAttribute("value",&functionValue);
//				useGLIFunction(functionSetPointer,&functionValue);
//				break;
//			}
//		case GLIEnums::FunctionType::STRING:{
//				string functionValue; 
//				pElem->QueryStringAttribute("value",&functionValue);
//				useGLIFunction(functionSetPointer,(void*)functionValue.c_str());
//				break;
//			}
//		case GLIEnums::FunctionType::UINTARRAY:
//		case GLIEnums::FunctionType::STRINGARRAY:
//			useGLIClearFunction(functionSetPointer);
//			loadDebugArrayData(handle,functionName,functionSetPointer);
//			break;
//		default:
//			break;
//		}
//	}
//#endif
//}

/* ----------------------------------------------------------------
Specification of the configdata:
				<arrayconfigname>
					<item value="...">
					...
				</arrayconfigname>
----------------------------------------------------------------- */
//void
//ProjectLoader::loadDebugArrayData (TiXmlHandle &hRoot, const char *functionName, void *functionSetPointer)
//{
//#ifdef GLINTERCEPTDEBUG
//	TiXmlElement *pElem;
//	TiXmlHandle handle (hRoot.FirstChild(functionName).Element());
//	unsigned int functionType = getGLIFunctionType(functionSetPointer);
//
//	pElem = handle.FirstChild().Element();
//	switch (functionType)
//	{
//	case GLIEnums::FunctionType::UINTARRAY:{
//		for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
//				unsigned int functionValue; 
//				pElem->QueryUnsignedAttribute("value",&functionValue);
//				useGLIFunction(functionSetPointer,&functionValue);
//				break;
//			}
//		}
//	case GLIEnums::FunctionType::STRINGARRAY:{
//		for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
//				string functionValue; 
//				pElem->QueryStringAttribute("value",&functionValue);
//				useGLIFunction(functionSetPointer,(void*)functionValue.c_str());
//				break;
//			}
//		}
//	default:
//		break;
//	}
//#endif
//}
//

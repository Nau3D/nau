#include <nau/loader/projectloader.h>

#include <nau.h>

#include <nau/material/programvalue.h>
#include <nau/render/pipeline.h>
#include <nau/render/passfactory.h>
#ifdef NAU_OPTIX_PRIME
#include <nau/render/passoptixprime.h>
#endif
#ifdef NAU_OPTIX
#include <nau/render/passOptix.h>
#endif

#include <nau/render/passCompute.h>

#include <nau/system/textutil.h>
//
#include <nau/event/sensorfactory.h>
#include <nau/event/interpolatorFactory.h>
#include <nau/scene/sceneobjectfactory.h>
#include <nau/event/route.h>
#include <nau/event/objectAnimation.h>
#include <nau/render/rendertarget.h>
#include <nau/scene/geometryobject.h>
#include <nau/geometry/primitive.h>
#include <nau/math/transformfactory.h>

#include <nau/slogger.h>

#include <nau/config.h>

#ifdef NAU_PLATFORM_WIN32
#include <nau/system/dirent.h>
#else
#include <dirent.h>
#include <sys/types.h>
#endif

#ifdef GLINTERCEPTDEBUG
#include <nau/loader/projectloaderdebuglinker.h>
#endif


using namespace nau::loader;
using namespace nau::math;
using namespace nau::material;
using namespace nau::render;
using namespace nau::scene;
using namespace nau::geometry;
//
using namespace nau::event_;
//

std::string ProjectLoader::s_Path = "";
std::string ProjectLoader::s_File = "";
std::string ProjectLoader::s_Dummy;

char ProjectLoader::s_pFullName[256] = "";

vec4 ProjectLoader::s_Dummy_vec4;
bvec4 ProjectLoader::s_Dummy_bvec4;
float ProjectLoader::s_Dummy_float;
int ProjectLoader::s_Dummy_int;
bool ProjectLoader::s_Dummy_bool;


std::string 
ProjectLoader::toLower(std::string strToConvert) {

	s_Dummy = strToConvert;
   for (std::string::iterator p = s_Dummy.begin(); s_Dummy.end() != p; ++p)
       *p = tolower(*p);

   return s_Dummy;

}

void *
ProjectLoader::readAttr(std::string pName, TiXmlElement *p, Enums::DataType type, AttribSet attribs) {

	std::string s;

	switch (type) {
	
		case Enums::FLOAT:
			if (TIXML_SUCCESS != p->QueryFloatAttribute("value", &s_Dummy_float))
				NAU_THROW("File %s: Element %s: Attribute %s without a value", ProjectLoader::s_File.c_str(),pName.c_str(), p->Value()); 
			return &s_Dummy_float;
			break;
		case Enums::VEC4:
			if ((TIXML_SUCCESS == p->QueryFloatAttribute("x", &(s_Dummy_vec4.x)) || TIXML_SUCCESS == p->QueryFloatAttribute("r", &(s_Dummy_vec4.x))) 
				&& ((TIXML_SUCCESS == p->QueryFloatAttribute("y", &(s_Dummy_vec4.y)) || TIXML_SUCCESS == p->QueryFloatAttribute("g", &(s_Dummy_vec4.y))) 
				&& ((TIXML_SUCCESS == p->QueryFloatAttribute("z", &(s_Dummy_vec4.z)) || TIXML_SUCCESS == p->QueryFloatAttribute("b", &(s_Dummy_vec4.z)))))) {
			
				if (TIXML_SUCCESS != p->QueryFloatAttribute("w", &(s_Dummy_vec4.w)))
					p->QueryFloatAttribute("a", &(s_Dummy_vec4.w));
				return &s_Dummy_vec4;
			}
			else
				NAU_THROW("File %s: Element %s: Attribute %s has absent or incomplete value (x,y and z are required, w is optional)", ProjectLoader::s_File.c_str(),pName.c_str(),p->Value()); 
			break;

		case Enums::BVEC4:
			if (TIXML_SUCCESS == p->QueryBoolAttribute("x", &(s_Dummy_bvec4.x)) 
				&& TIXML_SUCCESS == p->QueryBoolAttribute("y", &(s_Dummy_bvec4.y))
				&& TIXML_SUCCESS == p->QueryBoolAttribute("z", &(s_Dummy_bvec4.z))
				&& TIXML_SUCCESS == p->QueryBoolAttribute("w", &(s_Dummy_bvec4.w))) {
			
				return &s_Dummy_bvec4;
			}
			else
				NAU_THROW("File %s: Element %s: Attribute %s has absent or incomplete value (x,y,z and w are required)", ProjectLoader::s_File.c_str(),pName.c_str(),p->Value()); 
			break;

		case Enums::INT:
			if (TIXML_SUCCESS != p->QueryIntAttribute("value", &s_Dummy_int))
				NAU_THROW("File %s: Element %s: Attribute %s without a value", ProjectLoader::s_File.c_str(),pName.c_str(), p->Value()); 
			return &s_Dummy_int;
			break;
		case Enums::UINT:
			if (TIXML_SUCCESS != p->QueryIntAttribute("value", &s_Dummy_int))
				NAU_THROW("File %s: Element %s: Attribute %s without a value", ProjectLoader::s_File.c_str(),pName.c_str(), p->Value()); 
			return &s_Dummy_int;
			break;
		case Enums::BOOL:
			if (TIXML_SUCCESS != p->QueryBoolAttribute("value", &s_Dummy_bool))
				NAU_THROW("File %s: Element %s: Attribute %s without a value", ProjectLoader::s_File.c_str(),pName.c_str(), p->Value()); 
			return &s_Dummy_bool;
			break;
		case Enums::ENUM:
			if (TIXML_SUCCESS != p->QueryStringAttribute("value", &s))
				NAU_THROW("File %s: Element %s: Attribute %s without a value", ProjectLoader::s_File.c_str(),pName.c_str(), p->Value()); 
			if (!attribs.isValid(p->Value(), s))
				NAU_THROW("File %s: Element %s: Attribute %s has an invalid value", ProjectLoader::s_File.c_str(),pName.c_str(), p->Value()); 
			s_Dummy_int = attribs.getListValueOp(attribs.getID(p->Value()), s); 
			return &s_Dummy_int;
			break;
		default:
			assert(false && "Missing attribute type in function ProjectLoader::readAttr");
	}
	return NULL;
}


/*--------------------------------------------------------------------
Project Specification

<?xml version="1.0" ?>
<project name="teste1-shadows">
	<assets>
		...
	</assets>
	<pipelines>
		...
	</pipelines>
</project>

-------------------------------------------------------------------*/

void
ProjectLoader::load (std::string file, int *width, int *height, bool *tangents, bool *triangleIDs)
{
	ProjectLoader::s_Path = FileUtil::GetPath(file);
	ProjectLoader::s_File = file;

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


	try {
		*width = 0;
		*height = 0;
		if (TIXML_SUCCESS == pElem->QueryIntAttribute("width",width) &&
			TIXML_SUCCESS == pElem->QueryIntAttribute("height",height)) {
				if (*width <= 0 || *height <= 0) {
					*width = 0;
					*height = 0;
				}
				NAU->setWindowSize(*width, *height);
		}

		const char *pUseTangents = pElem->Attribute("useTangents");
		if (pUseTangents)
			*tangents = !strcmp(pUseTangents, "yes");
		else
			*tangents = false;

		const char *pUseTriangleIDs = pElem->Attribute("useTriangleIDs");
		if (pUseTriangleIDs)
			*triangleIDs = !strcmp(pUseTriangleIDs, "yes");
		else
			*triangleIDs = false;

		//bool core;
		//const char *pCoreProfile = pElem->Attribute("core");
		//if (pCoreProfile)
		//	core = !strcmp(pCoreProfile, "yes");
		//else
		//	core = false;
		//RENDERER->setCore(core);
		
#ifdef GLINTERCEPTDEBUG
		loadDebug(hRoot);
#endif
		loadAssets (hRoot, matLibs);
		loadPipelines (hRoot);
	}
	catch(std::string &s) {
		throw(s);
	}
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
type see readAttr()

----------------------------------------------------------------- */


void 
ProjectLoader::loadUserAttrs(TiXmlHandle handle) 
{
	TiXmlElement *pElem;

	pElem = handle.FirstChild ("attributes").FirstChild ("attribute").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
		const char *pContext = pElem->Attribute("context");
		const char *pName = pElem->Attribute ("name");
		const char *pType = pElem->Attribute ("type");
			
		if (0 == pContext)
			NAU_THROW("File %s: Attribute without a context", ProjectLoader::s_File.c_str()); 					

		if (!NAU->validateUserAttribContext(pContext))
			NAU_THROW("File %s: Attribute with an invalid context %s", ProjectLoader::s_File.c_str(), pContext); 					

		if (0 == pName) 
			NAU_THROW("File %s: Attribute without a name", ProjectLoader::s_File.c_str()); 

		if (!NAU->validateUserAttribName(pContext, pName))
			NAU_THROW("File %s: Attribute name %s is already in use in context %s", ProjectLoader::s_File.c_str(), pName, pContext);
		
		if (0 == pType) 
			NAU_THROW("File %s: Attribute without a type", ProjectLoader::s_File.c_str()); 					

		if (!Attribute::isValidUserAttrType(pType))
			NAU_THROW("File %s: Attribute with na invalid type", ProjectLoader::s_File.c_str()); 					

		AttribSet *attribs = NAU->getAttribs(pContext);
		Enums::DataType dt = Enums::getType(pType);
		void *v = readAttr(pName, pElem, dt, *attribs);
		Attribute a = Attribute(attribs->getNextFreeID(), pName, dt, false, v);
		attribs->add(a);	
		std::string s;
		SLOG("User Attribute : %s::%s", pContext, pName);
				
	}
}




/* ----------------------------------------------------------------
Specification of the scenes:

		<scenes>
			<scene name="MainScene" type="Octree" filename = "aScene.cbo" param="SWAP_YZ">
				<file>..\ntg-bin-3\fonte-finallambert.dae</file>
				<folder>..\ntg-bin-pl3dxiv</folder>
			</scene>
			...
		</scenes>

scenes can have multiple "scene" defined
each scene can have files and folders OR a single file containing a scene.
the path may be relative to the project file or absolute
type see sceneFactory

param is passed to the loader
	3DS loader: SWAP_YZ to indicate that ZY axis should be swaped 
	( by default they are swapped)
----------------------------------------------------------------- */


void 
ProjectLoader::loadScenes(TiXmlHandle handle) 
{
	TiXmlElement *pElem;

	pElem = handle.FirstChild ("scenes").FirstChild ("scene").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
		const char *pName = pElem->Attribute ("name");
		const char *pType = pElem->Attribute ("type");
		const char *pFilename = pElem->Attribute("filename");
		const char *pParam = pElem->Attribute("param");
			
		std::string s;

		if (0 == pName) {
			NAU_THROW("File %s: scene has no name", ProjectLoader::s_File.c_str()); 					
		}

		SLOG("Scene : %s", pName);
			
		if (pParam == NULL)
			s = "";
		else
			s = pParam;

		IScene *is;
		if (0 == pType) 
			is = RENDERMANAGER->createScene (pName);
		else {
			is = RENDERMANAGER->createScene(pName, pType);
			if (is == NULL)
				NAU_THROW("Invalid type for scene %s in file %s", pName, ProjectLoader::s_File.c_str()); 	
		}
		

		const char *pTransX = pElem->Attribute("transX");
		const char *pTransY = pElem->Attribute("transY");
		const char *pTransZ = pElem->Attribute("transZ");

		const char *pScaleX = pElem->Attribute("scaleX");
		const char *pScaleY = pElem->Attribute("scaleY");
		const char *pScaleZ = pElem->Attribute("scaleZ");
		const char *pScale = pElem->Attribute("scale");

		ITransform *tis = TransformFactory::create("SimpleTransform");

		if (pTransX && pTransY && pTransZ) {
			tis->translate(nau::system::textutil::ParseFloat(pTransX),
				nau::system::textutil::ParseFloat(pTransY),
				nau::system::textutil::ParseFloat(pTransZ));

		}
		if (pScaleX && pScaleY && pScaleZ) {
			tis->scale(nau::system::textutil::ParseFloat(pScaleX),
				nau::system::textutil::ParseFloat(pScaleY),
				nau::system::textutil::ParseFloat(pScaleZ));
		}	
		if (pScale) {
			float scale = nau::system::textutil::ParseFloat(pScale);
			tis->scale(scale);
		}

		is->setTransform(tis);

		// the filename should point to a scene
		if (0 != pFilename) {

			if (!FileUtil::exists(FileUtil::GetFullPath(ProjectLoader::s_Path, pFilename)))
				NAU_THROW("Scene file %s does not exist", pFilename); 			

			try {
				nau::Nau::getInstance()->loadAsset (FileUtil::GetFullPath(ProjectLoader::s_Path, pFilename), pName, s);
			}
			catch(std::string &s) {
				throw(s);
			}
		}
		else {
			handle = TiXmlHandle (pElem);
			TiXmlElement* pElementAux;

			pElementAux = handle.FirstChild("geometry").Element();
			for ( ; 0 != pElementAux; pElementAux = pElementAux->NextSiblingElement()) {
				const char *pName = pElementAux->Attribute("name");
				const char *pPrimType = pElementAux->Attribute ("type");
				const char *pMaterial = pElementAux->Attribute("material");

				pTransX = pElementAux->Attribute("transX");
				pTransY = pElementAux->Attribute("transY");
				pTransZ = pElementAux->Attribute("transZ");

				pScaleX = pElementAux->Attribute("scaleX");
				pScaleY = pElementAux->Attribute("scaleY");
				pScaleZ = pElementAux->Attribute("scaleZ");
				pScale = pElementAux->Attribute("scale");

				if (pPrimType == NULL)
					NAU_THROW("Scene %s has no type", pName); 			

				GeometricObject *go = (GeometricObject *)nau::scene::SceneObjectFactory::create("Geometry");
				
				if (go == NULL)
					NAU_THROW("Scene %s has invalid type type", pName); 
				if (pName)
					go->setName(pName);

				Primitive *p = (Primitive *)RESOURCEMANAGER->createRenderable(pPrimType, pName);
				std::string n = p->getParamfName(0);
				unsigned int i = 0;
				while (Primitive::NoParam != n) {

					float value;
					if (TIXML_SUCCESS == pElementAux->QueryFloatAttribute (n.c_str(), &value)) 
						p->setParam(i,value);
					++i;
					n = p->getParamfName(i);
				}
				if (i)
					p->build();
								
				ITransform *t = TransformFactory::create("SimpleTransform");

				if (pTransX && pTransY && pTransZ) {
					t->translate(nau::system::textutil::ParseFloat(pTransX),
						nau::system::textutil::ParseFloat(pTransY),
						nau::system::textutil::ParseFloat(pTransZ));

				}
				if (pScaleX && pScaleY && pScaleZ) {
					t->scale(nau::system::textutil::ParseFloat(pScaleX),
						nau::system::textutil::ParseFloat(pScaleY),
						nau::system::textutil::ParseFloat(pScaleZ));
				}	
				if (pScale) {
					float scale = nau::system::textutil::ParseFloat(pScale);
					t->scale(scale);
				}
				go->setTransform(t);
				go->setRenderable(p);

				if (pMaterial) {
					if (!MATERIALLIBMANAGER->hasMaterial(DEFAULTMATERIALLIBNAME,pMaterial)) {
						Material *mat = MATERIALLIBMANAGER->createMaterial(pMaterial);
					}
					go->setMaterial(pMaterial);
				}
				is ->add(go);

			}


			pElementAux = handle.FirstChild ("file").Element();
			for ( ; 0 != pElementAux; pElementAux = pElementAux->NextSiblingElement()) {
				const char * pFileName = pElementAux->GetText();

				if (!FileUtil::exists(FileUtil::GetFullPath(ProjectLoader::s_Path, pFileName)))
					NAU_THROW("Scene file %s does not exist", pFileName); 			

				nau::Nau::getInstance()->loadAsset (FileUtil::GetFullPath(ProjectLoader::s_Path, pFileName), pName, s);
			}

			pElementAux = handle.FirstChild ("folder").Element();
			for ( ; 0 != pElementAux; pElementAux = pElementAux->NextSiblingElement()) {
				
				DIR *dir;

				struct dirent *ent;

				const char * pDirName = pElementAux->GetText();

				dir = opendir (FileUtil::GetFullPath(ProjectLoader::s_Path, pDirName).c_str());

				if (!dir)
					NAU_THROW("Scene folder %s does not exist", pDirName); 			

				if (0 != dir) {

					int count = 0;
					while ((ent = readdir (dir)) != 0) {
						char file [1024];

#ifdef NAU_PLATFORM_WIN32
						sprintf (file, "%s\\%s", (char *)FileUtil::GetFullPath(ProjectLoader::s_Path, pDirName).c_str(), ent->d_name);
#else
						sprintf (file, "%s/%s", pDirName, ent->d_name);						
#endif
						try {
							nau::Nau::getInstance()->loadAsset (file, pName,s);
						}
						catch(std::string &s) {
							closedir(dir);
							throw(s);
						}
						++count;
					}
					if (count < 3 )
						NAU_THROW("Scene folder %s is empty", pDirName); 			

				closedir (dir);
				}
			}
		}
	}
}

/* ----------------------------------------------------------------
Specification of the atomic semantics:

		<atomics>
			<atomic id=0 semantics="Red Pixels"/>
			...
		</atomics>

Each atomic must have an id and a name.
----------------------------------------------------------------- */

#if NAU_OPENGL_VERSION >= 400

void
ProjectLoader::loadAtomicSemantics(TiXmlHandle handle) 
{
	TiXmlElement *pElem;

	pElem = handle.FirstChild ("atomics").FirstChild ("atomic").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
		const char *pName = pElem->Attribute ("semantics");
		int id;
		int failure = pElem->QueryIntAttribute("id", &id);

		if (failure) {
			NAU_THROW("Atomic has no id, in file %s", ProjectLoader::s_File.c_str());
		}

		if (0 == pName) {
			NAU_THROW("Atomic %d has no semantics, in file %s", id, ProjectLoader::s_File.c_str());
		}

		SLOG("Atomic : %d %s", id, pName);

		RENDERER->addAtomic(id,pName);
	} 
}

#endif
/* ----------------------------------------------------------------
Specification of the viewport:

		<viewports>
			<viewport name="MainViewport">
				<CLEAR_COLOR r = 0.0 g = 0.0 b = 0.3 />
				<ORIGIN x = 0.66 y = 0 />
				<SIZE width= 0.33 ratio = 1.0 />
			</viewport>
			...
		</viewports>

		or

		<viewports>
			<viewport name="MainViewport">
				<CLEAR_COLOR r = 0.0 g = 0.0 b = 0.3 />
				<ORIGIN x = 0.66 y = 0 />
				<SIZE width=0.33 height = 1.0 />
			</viewport>
			...
		</viewports>

CLEAR_COLOR is optional, if not specified it will be black

geometry can be relative or absolute, if values are smaller than, or equal to 1
it is assumed to be relative.

geometry is specified with ORIGIN and SIZE

ratio can be used instead of height, in which case height = width*ratio
----------------------------------------------------------------- */

void
ProjectLoader::loadViewports(TiXmlHandle handle) 
{
	TiXmlElement *pElem;
	Viewport *v;

	pElem = handle.FirstChild ("viewports").FirstChild ("viewport").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
		const char *pName = pElem->Attribute ("name");

		if (0 == pName) {
			NAU_THROW("File %s: Viewport has no name", ProjectLoader::s_File.c_str());
		}

		SLOG("Viewport : %s", pName);

		TiXmlElement *pElemAux = 0;
		pElemAux = pElem->FirstChildElement ("CLEAR_COLOR");

		if (0 == pElemAux) {
			// no color is specified
			v = nau::Nau::getInstance()->createViewport(pName, vec4(0.0f, 0.0f, 0.0f, 1.0f));				
		}
		else {
			vec4 *v4;
			//// clear color for viewport
			v4 = (vec4 *)readAttr(pName, pElemAux, Enums::VEC4, Viewport::Attribs);
			v = nau::Nau::getInstance()->createViewport (pName, vec4 (v4->x, v4->y, v4->z, 1.0f));
		}

		pElemAux = pElem->FirstChildElement("SIZE");
		if (!pElemAux) {
			v->setProp(Viewport::FULL, true);
		}
		else {
			float width,height,ratio;
			if (TIXML_SUCCESS != pElemAux->QueryFloatAttribute ("width", &width) ||
				(TIXML_SUCCESS != pElemAux->QueryFloatAttribute ("height", &height) &&
				TIXML_SUCCESS != pElemAux->QueryFloatAttribute("ratio", &ratio))){

				NAU_THROW("File %s: Element %s: SIZE definition error", ProjectLoader::s_File.c_str(), pName);					
			}
			
			if (TIXML_SUCCESS == pElemAux->QueryFloatAttribute("ratio", &ratio))
				v->setProp(Viewport::RATIO, ratio);

			v->setProp(Viewport::SIZE, vec2(width, height));
		}

		pElemAux = pElem->FirstChildElement("ORIGIN");
		if (pElemAux)
		{
			float x,y;
			if (TIXML_SUCCESS != pElemAux->QueryFloatAttribute ("x", &x) ||
				TIXML_SUCCESS != pElemAux->QueryFloatAttribute ("y", &y)){

				NAU_THROW("File %s: Element %s: ORIGIN definition error", ProjectLoader::s_File.c_str(), pName);					
			}		
			v->setProp(Viewport::ORIGIN, vec2(x, y));
		}
		else
			v->setProp(Viewport::ORIGIN, vec2(0.0f, 0.0f));

		// Reading remaining viewport attributes
		std::map<std::string, Attribute> attribs = Viewport::Attribs.getAttributes();
		TiXmlElement *p = pElem->FirstChildElement();
		Attribute a; 	
		void *value;

		while (p) {
			// skip previously processed elements
			if (strcmp(p->Value(), "ORIGIN") && strcmp(p->Value(), "SIZE") && strcmp(p->Value(), "CLEAR_COLOR")) {
				// trying to define an attribute that does not exist?		
				if (attribs.count(p->Value()) == 0)
					NAU_THROW("File %s: Element %s: %s is not an attribute", ProjectLoader::s_File.c_str(), pName, p->Value());
				// trying to set the value of a read only attribute?
				a = attribs[p->Value()];
				if (a.mReadOnlyFlag)
					NAU_THROW("File %s: Element %s: %s is a read-only attribute", ProjectLoader::s_File.c_str(), pName, p->Value());

				value = readAttr(pName, p, a.mType, Light::Attribs);
				v->setProp(a.mId, a.mType, value);
			}
			p = p->NextSiblingElement();
		}

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

	pElem = handle.FirstChild ("cameras").FirstChild ("camera").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
		const char *pName = pElem->Attribute ("name");

		if (0 == pName) {
			NAU_THROW("File %s: Camera has no name", ProjectLoader::s_File.c_str());
		}

		SLOG("Camera: %s", pName);

		if (RENDERMANAGER->hasCamera(pName))
			NAU_THROW("File %s: Camera %s is already defined", ProjectLoader::s_File.c_str(), pName);

		Camera *aNewCam = RENDERMANAGER->getCamera (pName);

		TiXmlElement *pElemAux = 0;
		std::string s;

		// Read Viewport
		pElemAux = pElem->FirstChildElement ("viewport");
		Viewport *v = 0;
		if (0 == pElemAux) {
			v = nau::Nau::getInstance()->getDefaultViewport ();
		} else {
			if (TIXML_SUCCESS != pElemAux->QueryStringAttribute("name", &s))
				NAU_THROW("File %s: Element %s: viewport name is required", ProjectLoader::s_File.c_str(), pName);

			// Check if previously defined
			v = nau::Nau::getInstance()->getViewport (s);
			if (!v)
				NAU_THROW("File %s: Element %s: viewport %s is not previously defined", ProjectLoader::s_File.c_str(), pName, s.c_str());

			aNewCam->setViewport (v);
		}

		// read projection values
		pElemAux = pElem->FirstChildElement("projection");
		if (pElemAux == NULL)
			NAU_THROW("File %s: Element %s: projection definition missing", ProjectLoader::s_File.c_str(), pName);

		if (TIXML_SUCCESS != pElemAux->QueryStringAttribute("TYPE", &s))
			NAU_THROW("File %s: Element %s: projection type is not defined", ProjectLoader::s_File.c_str(), pName);

		if (s == "PERSPECTIVE") {

			float fov, nearPlane, farPlane;
			if (TIXML_SUCCESS != pElemAux->QueryFloatAttribute ("FOV", &fov) || 
				TIXML_SUCCESS != pElemAux->QueryFloatAttribute ("NEAR", &nearPlane) ||
				TIXML_SUCCESS != pElemAux->QueryFloatAttribute ("FAR", &farPlane)) {

				NAU_THROW("File %s: Element %s: perspective definition error (FOV, NEAR and FAR are required)", ProjectLoader::s_File.c_str(), pName);
			}	
			aNewCam->setPerspective (fov, nearPlane, farPlane);
		}
		else if (s == "ORTHO") {
			float left, right, bottom, top, nearPlane, farPlane;

			if (TIXML_SUCCESS != pElemAux->QueryFloatAttribute ("LEFT", &left) || 
				TIXML_SUCCESS != pElemAux->QueryFloatAttribute ("RIGHT", &right) ||
				TIXML_SUCCESS != pElemAux->QueryFloatAttribute ("BOTTOM", &bottom) ||
				TIXML_SUCCESS != pElemAux->QueryFloatAttribute ("TOP", &top) || 
				TIXML_SUCCESS != pElemAux->QueryFloatAttribute ("NEAR", &nearPlane) ||
				TIXML_SUCCESS != pElemAux->QueryFloatAttribute ("FAR", &farPlane)) {

				NAU_THROW("File %s: Element %s: ortho definition error (LEFT, RIGHT, BOTTOM, TOP, NEAR and FAR are required)", ProjectLoader::s_File.c_str(), pName);
			}
			aNewCam->setOrtho (left, right, bottom, top, nearPlane, farPlane);
		}
		else
			NAU_THROW("File %s: Element %s: projection type is not valid", ProjectLoader::s_File.c_str(), pName);

		// Reading remaining camera attributes
		std::map<std::string, Attribute> attribs = Camera::Attribs.getAttributes();
		TiXmlElement *p = pElem->FirstChildElement();
		Attribute a; 	
		void *value;

		while (p) {
			// skip previously processed elements
			if (strcmp(p->Value(), "projection") && strcmp(p->Value(), "viewport")) {
				// trying to define an attribute that does not exist?		
				if (attribs.count(p->Value()) == 0) {
					NAU_THROW("File %s: Element %s: %s is not an attribute", ProjectLoader::s_File.c_str(), pName, p->Value());
				}
				// trying to set the value of a read only attribute?
				a = attribs[p->Value()];
				if (a.mReadOnlyFlag)
					NAU_THROW("File %s: Element %s: %s is a read-only attribute", ProjectLoader::s_File.c_str(), pName, p->Value());

				value = readAttr(pName, p, a.mType, Light::Attribs);
				aNewCam->setProp(a.mId, a.mType, value);
			}
			p = p->NextSiblingElement();
		}
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
	Light *l;
	void *value;

	pElem = handle.FirstChild ("lights").FirstChild ("light").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
		const char *pName = pElem->Attribute ("name");
		const char *pClass = pElem->Attribute("class"); 
		
		if (0 == pName) 
			NAU_THROW("Light has no name in file %s", ProjectLoader::s_File.c_str());

		SLOG("Light: %s", pName);


		if (RENDERMANAGER->hasLight(pName))
			NAU_THROW("Light %s is already defined, in file %s", pName, ProjectLoader::s_File.c_str());

		if (0 == pClass) 
			l = RENDERMANAGER->getLight (pName);
		else
			l = RENDERMANAGER->getLight (pName, pClass);
		
		// Reading Light Attributes

		std::map<std::string, Attribute> attribs = Light::Attribs.getAttributes();
		TiXmlElement *p = pElem->FirstChildElement();
		Attribute a;
		while (p) {
			// trying to define an attribute that does not exist		
			if (attribs.count(p->Value()) == 0)
				NAU_THROW("Light %s: %s is not an attribute, in file %s", pName, p->Value(),ProjectLoader::s_File.c_str());
			// trying to set the value of a read only attribute
			a = attribs[p->Value()];
			if (a.mReadOnlyFlag)
				NAU_THROW("Light %s: %s is a read-only attribute, in file %s", pName, p->Value(),ProjectLoader::s_File.c_str());

			value = readAttr(pName, p, a.mType, Light::Attribs);
			l->setProp(a.mId, a.mType, value);
			p = p->NextSiblingElement();
		}

	}//End of lights
}


/* ----------------------------------------------------------------
Specification of the assets:

	<assets>
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

	loadUserAttrs(handle);
	loadScenes(handle);
	loadViewports(handle);
	loadCameras(handle);
	loadLights(handle);	
	loadEvents(handle);
#if NAU_OPENGL_VERSION >= 400
	loadAtomicSemantics(handle);
#endif
	pElem = handle.FirstChild ("materialLibs").FirstChild ("mlib").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
		const char *pFilename = pElem->Attribute ("filename");

		if (0 == pFilename) {
			NAU_THROW("No file specified for material lib in file %s", ProjectLoader::s_File.c_str());
		}

		try {
			SLOG("Loading Material Lib from file : %s", FileUtil::GetFullPath(ProjectLoader::s_Path,pFilename).c_str());
			loadMatLib(FileUtil::GetFullPath(ProjectLoader::s_Path,pFilename));
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

	<camera>testCamera</camera>

Specifies a previously defined camera (in the assets part of the file)
-----------------------------------------------------------------------------*/

void 
ProjectLoader::loadPassCamera(TiXmlHandle hPass, Pass *aPass) 
{
	TiXmlElement *pElem;

	pElem = hPass.FirstChild ("camera").Element();
	if (0 == pElem && aPass->getClassName() != "compute") {
		NAU_THROW("No camera element found in pass: %s", aPass->getName().c_str());
	}
	else if (pElem != 0) {
		if (!RENDERMANAGER->hasCamera(pElem->GetText()))
			NAU_THROW("Camera %s is not defined, in pass: %s", pElem->GetText(), aPass->getName().c_str());

		aPass->setCamera (pElem->GetText());
	}
}

/* -----------------------------------------------------------------------------
LIGHTS

	<lights>
		<light>Sun</light>
	</lights>

Specifies a previously defined light (in the assets part of the file)
-----------------------------------------------------------------------------*/

void 
ProjectLoader::loadPassLights(TiXmlHandle hPass, Pass *aPass) 
{
	TiXmlElement *pElem;

	pElem = hPass.FirstChild ("lights").FirstChild ("light").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
		const char *pName = pElem->GetText();

		if (0 == pName) {
			NAU_THROW("Light has no name in pass: %s", aPass->getName().c_str());
		}
		if (!RENDERMANAGER->hasLight(pName))
			NAU_THROW("Light %s is not defined, in pass: %s", pName, aPass->getName().c_str());

		aPass->addLight (pName);
	}//End of lights
}



/* -----------------------------------------------------------------------------
SCENES

	<scenes>
		<scene>MainScene</scene>
		...
	</scenes>

Specifies a previously defined scene (in the assets part of the file)
-----------------------------------------------------------------------------*/



void 
ProjectLoader::loadPassScenes(TiXmlHandle hPass, Pass *aPass) 
{
	TiXmlElement *pElem;

	pElem = hPass.FirstChild ("scenes").FirstChild ("scene").Element();
	if (0 == pElem && aPass->getClassName() != "compute") {
		NAU_THROW("No Scene element found in pass: %s", aPass->getName().c_str());
	}
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {

		if (!RENDERMANAGER->hasScene(pElem->GetText()))
				NAU_THROW("Scene %s is not defined, in pass: %s", pElem->GetText(),aPass->getName().c_str());

		aPass->addScene (pElem->GetText());
	} //End of scenes
}


/* -----------------------------------------------------------------------------
CLEAR DEPTH AND COLOR

	<color clear=true />
	<depth clear=true clearValue=1.0 test=true write=true/>

	<stencil clear=true clearValue=0 test=true mask=255>
		<stencilFunc func=ALWAYS ref=1 mask=255/>
		<stencilOp sfail=KEEP dfail=KEEP dpass=KEEP />
	</stencil>

By default these fields will be true and can be omitted.
Clear color is the one from the viewport
-----------------------------------------------------------------------------*/

void 
ProjectLoader::loadPassClearDepthAndColor(TiXmlHandle hPass, Pass *aPass)
{
	TiXmlElement *pElem;
	bool *b = new bool();

	// Clear Color and Depth
	pElem = hPass.FirstChild ("depth").Element();
	if (0 != pElem) {
		float vf;

		const char *pEnable = pElem->Attribute ("test");
		if (pEnable != NULL) {
			if (!strcmp(pEnable, "false"))
				*b = false;
			else
				*b = true;
			aPass->setProp(Pass::DEPTH_ENABLE, Enums::BOOL, b);
		}
		const char *pClear = pElem->Attribute ("clear");
		if (pClear != NULL) {
			if (!strcmp(pClear, "false"))
				*b = false;
			else
				*b = true;
			aPass->setProp(Pass::DEPTH_CLEAR, Enums::BOOL, b);
		}
		const char *pWrite = pElem->Attribute ("write");
		if (pWrite != NULL) {
			if (!strcmp(pWrite, "false"))
				*b = false;
			else
				*b = true;
			aPass->setProp(Pass::DEPTH_MASK, Enums::BOOL, b);
		}
		if (TIXML_SUCCESS == pElem->QueryFloatAttribute ("clearValue",&vf))
			aPass->setDepthClearValue(vf);

		const char *pFunc = pElem->Attribute("func");
		if (pFunc) {
			int enumFunc = Pass::Attribs.getListValueOp(Pass::DEPTH_FUNC, pFunc);//IState::translateStringToFuncEnum(pFunc);
			if (enumFunc != -1)
				aPass->setDepthFunc(enumFunc);
		}

	}

	pElem = hPass.FirstChild ("stencil").Element();
	if (0 != pElem) {
		float vf;
		const char *pEnable = pElem->Attribute ("test");
		if (pEnable != NULL) {
			if (!strcmp(pEnable, "false"))
				*b = false;
			else
				*b = true;
			aPass->setProp(Pass::STENCIL_ENABLE, Enums::BOOL, b);
		}
		const char *pClear = pElem->Attribute ("clear");
		if (pClear != NULL) {
			if (!strcmp(pClear, "false"))
				*b = false;
			else
				*b = true;
			aPass->setProp(Pass::STENCIL_CLEAR, Enums::BOOL, b);
		}
		if (TIXML_SUCCESS == pElem->QueryFloatAttribute ("clearValue",&vf))
			aPass->setStencilClearValue(vf);
		//if (TIXML_SUCCESS == pElem->QueryIntAttribute ("maskValue",&vi))
		//	aPass->setStencilMaskValue(vf);

		TiXmlElement *pElemAux = pElem->FirstChildElement("stencilFunc");
		int ref, mask;
		if (pElemAux != NULL) {
			const char *pFunc = pElemAux->Attribute("func");
			if ((pFunc != NULL) && (Pass::Attribs.isValid("STENCIL_FUNC", pFunc) /*IRenderer::translateStringToStencilFunc(pFunc) != -1*/) &&
				(TIXML_SUCCESS == pElemAux->QueryIntAttribute ("ref",&ref)) &&
				(TIXML_SUCCESS == pElemAux->QueryIntAttribute ("mask",&mask)) && (mask >= 0))

					aPass->setStencilFunc((Pass::StencilFunc)Pass::Attribs.getListValueOp(Pass::STENCIL_FUNC, pFunc)/* (IRenderer::StencilFunc)IRenderer::translateStringToStencilFunc(pFunc)*/, 
											ref, (unsigned int)mask);
		}
		pElemAux = pElem->FirstChildElement("stencilOp");
		if (pElemAux != NULL) {

			const char *pSFail = pElemAux->Attribute("sfail");
			const char *pDFail = pElemAux->Attribute("dfail");
			const char *pDPass = pElemAux->Attribute("dpass");
			if (pSFail != NULL && pDFail != NULL && pDPass != NULL && 
				Pass::Attribs.isValid("STENCIL_FAIL", pSFail) &&
				Pass::Attribs.isValid("STENCIL_DEPTH_FAIL", pDFail) &&
				Pass::Attribs.isValid("STENCIL_DEPTH_PASS", pDPass))
				//IRenderer::translateStringToStencilOp(pSFail) != -1 &&
				//IRenderer::translateStringToStencilOp(pDFail) != -1 &&
				//IRenderer::translateStringToStencilOp(pDPass) != -1 )

					aPass->setStencilOp(
						(Pass::StencilOp)Pass::Attribs.getListValueOp(Pass::STENCIL_FAIL, pSFail),
						(Pass::StencilOp)Pass::Attribs.getListValueOp(Pass::STENCIL_DEPTH_FAIL, pDFail),
						(Pass::StencilOp)Pass::Attribs.getListValueOp(Pass::STENCIL_DEPTH_PASS, pDPass));
				//aPass->setStencilOp(
				//	(IRenderer::StencilOp)IRenderer::translateStringToStencilOp(pSFail),
				//	(IRenderer::StencilOp)IRenderer::translateStringToStencilOp(pDFail),
				//	(IRenderer::StencilOp)IRenderer::translateStringToStencilOp(pDPass));

		}
	}

	pElem = hPass.FirstChild ("color").Element();
	if (0 != pElem) {
		const char *pEnable = pElem->Attribute ("clear");
		if (pEnable != NULL) {
			if (!strcmp(pEnable, "false"))
				*b = false;
			else
				*b = true;
			aPass->setProp(Pass::COLOR_CLEAR, Enums::BOOL, b);
		}
	}
}

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

		// CHECK IF EXISTS
		aPass->setViewport (nau::Nau::getInstance()->getViewport (pElem->GetText()));
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
		const char *pName = pElem->Attribute ("name");
		const char *pLib = pElem->Attribute("fromLibrary");
		
		if (!pName )
			NAU_THROW("Texture without name in pass: %s", aPass->getName().c_str());
		if (!pLib) 
			sprintf(s_pFullName, "%s", pName);
		else
			sprintf(s_pFullName, "%s::%s", pLib, pName);

		if (!RESOURCEMANAGER->hasTexture(s_pFullName))
				NAU_THROW("Texture %s is not defined, in pass: %s", s_pFullName,aPass->getName().c_str());


		Material *srcMat, *dstMat;
		srcMat = MATERIALLIBMANAGER->getDefaultMaterial("__Quad");
		dstMat = srcMat->clone();
		dstMat->attachTexture(0,s_pFullName);
		MATERIALLIBMANAGER->addMaterial(aPass->getName(),dstMat);
		aPass->remapMaterial ("__Quad", aPass->getName(), "__Quad");

		/*Material *srcMat = new Material();
		srcMat->clone()
		srcMat->setName("__Quad");
		srcMat->getColor().setDiffuse(1.0,1.0,1.0,1.0);

		srcMat->attachTexture(0,s_pFullName);
		MATERIALLIBMANAGER->addMaterial(aPass->getName(),srcMat);

		aPass->remapMaterial ("__Quad", aPass->getName(), "__Quad");*/

	}
}
/* -----------------------------------------------------------------------------
PARAMS

	<userParams>
		<paramname1  value = 2 />
	</params>

Some passes may take other parameters which can be specified in here.
The available types so far are int and float.
-----------------------------------------------------------------------------*/

void 
ProjectLoader::loadPassParams(TiXmlHandle hPass, Pass *aPass)
{
	void *value;

	std::map<std::string, Attribute> attribs = Pass::Attribs.getAttributes();
	TiXmlElement *p = hPass.FirstChild("userAttributes").FirstChild().Element();
	Attribute a;
	while (p) {
		// trying to define an attribute that does not exist		
		if (attribs.count(p->Value()) == 0)
			NAU_THROW("Pass %s: %s is not an attribute, in file %s", aPass->getName().c_str() , p->Value(), ProjectLoader::s_File.c_str());
		// trying to set the value of a read only attribute
		a = attribs[p->Value()];
		if (a.mReadOnlyFlag)
			NAU_THROW("Pass %s: %s is a read-only attribute, in file %s", aPass->getName().c_str(), p->Value(), ProjectLoader::s_File.c_str());

		value = readAttr(aPass->getName(), p, a.mType, Light::Attribs);
		aPass->setProp(a.mId, a.mType, value);
		p = p->NextSiblingElement();
	}






	//int vi;
	//float vf;
	//std::string s;

	//pElem = hPass.FirstChild("params").FirstChild("param").Element();
	//for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {

	//	const char *pName = pElem->Attribute ("name");
	//	
	//	if (!pName)
	//		NAU_THROW("Param without name in pass: %s", aPass->getName().c_str());

	//	const char *pValue = pElem->Attribute ("string");
	//	if (TIXML_SUCCESS == pElem->QueryIntAttribute ("int",&vi))
	//		aPass->setParam(pName,vi);
	//	else if (TIXML_SUCCESS == pElem->QueryFloatAttribute ("float",&vf))
	//		aPass->setParam(pName,vf);
	//	//else if (pValue != NULL)
	//	//		aPass->setParam(pName,pValue);
	//	else {
	//		NAU_THROW("Param %s without value in pass: %s", pName, aPass->getName().c_str());
	//	}
	//}
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


	pElem = hPass.FirstChild ("rendertarget").Element();

	if (0 != pElem) {
		
		const char* pSameAs = pElem->Attribute ("sameas");
		const char *pName = pElem->Attribute("name");
		const char *pLib = pElem->Attribute("fromLibrary");

		if (0 != pSameAs) {
			if (passMapper.count (pSameAs) > 0) {
				aPass->setRenderTarget (passMapper[pSameAs]->getRenderTarget());
			} else {
				NAU_THROW("Render Target Definition in %s: Pass %s is not defined", aPass->getName().c_str(), pSameAs);
			}
		} 
		else if (0 != pName && 0 != pLib) {
	
			sprintf(s_pFullName, "%s::%s", pLib, pName);
			RenderTarget *rt = RESOURCEMANAGER->getRenderTarget(s_pFullName);
			if (rt != NULL)
				aPass->setRenderTarget(rt);
			else
				NAU_THROW("Render Target %s is not defined in material lib %s, in pass %s", pName, pLib, aPass->getName().c_str(), pSameAs);
			
		}
		else {
			NAU_THROW("Render Target Definition error in %s", aPass->getName().c_str());
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
		<valueof optixVar="lightDir" type="CURRENT" context="LIGHT" id=0 component="DIRECTION" />
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
		const char *pFile = pElem->Attribute ("file");
		const char *pProc = pElem->Attribute ("proc");
		
		if (!pType || (0 != strcmp(pType, "RayGen") && 0 != strcmp(pType, "Exception")))
			NAU_THROW("Invalid Optix Entry Point Type in pass %s", aPass->getName().c_str());

		if (!pFile)
			NAU_THROW("Missing Optix Entry Point File in pass %s", aPass->getName().c_str());

		if (!pProc)
			NAU_THROW("Missing Optix Entry Point Proc in pass %s", aPass->getName().c_str());

		if (!strcmp(pType, "RayGen"))
			p->setOptixEntryPointProcedure(nau::render::optixRender::OptixRenderer::RAY_GEN, pFile, pProc);
		else
			p->setOptixEntryPointProcedure(nau::render::optixRender::OptixRenderer::EXCEPTION, pFile, pProc);
	}
	pElem = hPass.FirstChild("optixDefaultMaterial").FirstChildElement("optixProgram").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {

		const char *pType = pElem->Attribute ("type");
		const char *pFile = pElem->Attribute ("file");
		const char *pProc = pElem->Attribute ("proc");
		const char *pRay  = pElem->Attribute ("ray");
		
		if (!pType || (0 != strcmp(pType, "Closest_Hit") && 0 != strcmp(pType, "Any_Hit")  && 0 != strcmp(pType, "Miss")))
			NAU_THROW("Invalid Optix Default Material Proc Type in pass %s", aPass->getName().c_str());

		if (!pFile)
			NAU_THROW("Missing Optix Default Material Proc File in pass %s", aPass->getName().c_str());

		if (!pProc)
			NAU_THROW("Missing Optix Default Material Proc in pass %s", aPass->getName().c_str());

		if (!pRay)
			NAU_THROW("Missing Optix Default Material Ray in pass %s", aPass->getName().c_str());
		
		if (!strcmp("Closest_Hit", pType)) 
			p->setDefaultMaterialProc(nau::render::optixRender::OptixMaterialLib::CLOSEST_HIT, pRay, pFile, pProc);
		else if (!strcmp("Any_Hit", pType)) 
			p->setDefaultMaterialProc(nau::render::optixRender::OptixMaterialLib::ANY_HIT, pRay, pFile, pProc);
		else if (!strcmp("Miss", pType)) 
			p->setDefaultMaterialProc(nau::render::optixRender::OptixMaterialLib::MISS, pRay, pFile, pProc);
	}

	pElem = hPass.FirstChild("optixMaterialMap").FirstChildElement("optixMap").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {

		const char *pName = pElem->Attribute("to");

		pElemAux = pElem->FirstChildElement("optixProgram");
		for ( ; 0 != pElemAux; pElemAux = pElemAux->NextSiblingElement()) {

		const char *pType = pElemAux->Attribute ("type");
		const char *pFile = pElemAux->Attribute ("file");
		const char *pProc = pElemAux->Attribute ("proc");
		const char *pRay  = pElemAux->Attribute ("ray");

		if (!pType || (0 != strcmp(pType, "Closest_Hit") && 0 != strcmp(pType, "Any_Hit")))
			NAU_THROW("Invalid Optix Material Proc Type in pass %s", aPass->getName().c_str());

		if (!pFile)
			NAU_THROW("Missing Optix Material Proc File in pass %s", aPass->getName().c_str());

		if (!pProc)
			NAU_THROW("Missing Optix Material Proc in pass %s", aPass->getName().c_str());

		if (!pRay)
			NAU_THROW("Missing Optix Material Ray in pass %s", aPass->getName().c_str());
		
		if (!strcmp("Closest_Hit", pType)) 
			p->setMaterialProc(pName, nau::render::optixRender::OptixMaterialLib::CLOSEST_HIT, pRay, pFile, pProc);
		else if (!strcmp("Any_Hit", pType)) 
			p->setMaterialProc(pName, nau::render::optixRender::OptixMaterialLib::ANY_HIT, pRay, pFile, pProc);
		}
	}

	pElem = hPass.FirstChild("optixInput").FirstChildElement("buffer").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {

		const char *pVar = pElem->Attribute ("var");
		const char *pTexture = pElem->Attribute ("texture");
		
		if (!pVar)
			NAU_THROW("Optix Variable required in Input Definition, in pass %s", aPass->getName().c_str());

		if (!pTexture)
			NAU_THROW("Missing texture in Optix Input Definitiont, in pass %s", aPass->getName().c_str());

		if (!RESOURCEMANAGER->hasTexture(pTexture))
				NAU_THROW("Texture %s is not defined, in pass: %s", pTexture,aPass->getName().c_str());

		p->setInputBuffer(pVar, pTexture);
	}

	pElem = hPass.FirstChild("optixOutput").FirstChildElement("buffer").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {

		const char *pVar = pElem->Attribute ("var");
		const char *pTexture = pElem->Attribute ("texture");
		
		if (!pVar)
			NAU_THROW("Optix Variable required in Input Definition, in pass %s", aPass->getName().c_str());

		if (!pTexture)
			NAU_THROW("Missing texture in Optix Input Definitiont, in pass %s", aPass->getName().c_str());

		if (!RESOURCEMANAGER->hasTexture(pTexture))
				NAU_THROW("Texture %s is not defined, in pass: %s", pTexture,aPass->getName().c_str());

		p->setOutputBuffer(pVar, pTexture);
	}

	pElem = hPass.FirstChild("optixGeometryProgram").FirstChildElement("optixProgram").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {

		const char *pType = pElem->Attribute ("type");
		const char *pFile = pElem->Attribute ("file");
		const char *pProc = pElem->Attribute ("proc");
		
		if (!pType || (0 != strcmp(pType, "Geometry_Intersection") && 0 != strcmp(pType, "Bounding_Box")))
			NAU_THROW("Invalid Optix Geometry Program in pass %s", aPass->getName().c_str());

		if (!pFile)
			NAU_THROW("Missing Optix Geometry Program File in pass %s", aPass->getName().c_str());

		if (!pProc)
			NAU_THROW("Missing Optix Geometry Program Proc in pass %s", aPass->getName().c_str());

		if (!strcmp(pType, "Geometry_Intersection"))
			p->setGeometryIntersectProc(pFile, pProc);
		else
			p->setBoundingBoxProc(pFile, pProc);
	}

	pElem = hPass.FirstChild("optixVertexAttributes").FirstChildElement("attribute").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {

		const char *pType = pElem->Attribute ("name");
		unsigned int vi = VertexData::getAttribIndex(pType);
		if (!pType || (VertexData::MaxAttribs == vi ))
			NAU_THROW("Invalid Optix Vertex Attribute in pass %s", aPass->getName().c_str());
	
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
			NAU_THROW("No optix variable name, in pass %s", aPass->getName().c_str());
		}
		if (0 == pType) {
			NAU_THROW("No type found for optix variable %s, in pass %s", pUniformName, aPass->getName().c_str());
		}
		if (0 == pContext) {
			NAU_THROW("No context found for optix variable %s, in pass %s", pUniformName, aPass->getName().c_str());
		}
		if (0 == pComponent) {
			NAU_THROW("No component found for optix variable %s, in pass %s", pUniformName, aPass->getName().c_str());
		}
		if (!ProgramValue::Validate(pType, pContext, pComponent))
			NAU_THROW("Optix variable %s is not valid, in pass %s", pUniformName, aPass->getName().c_str());

		int id = 0;
		if (((strcmp(pContext,"LIGHT") == 0) || (0 == strcmp(pContext,"TEXTURE"))) &&  (0 != strcmp(pComponent,"COUNT"))) {
			if (TIXML_SUCCESS != pElemAux2->QueryIntAttribute ("id", &id))
				NAU_THROW("No id found for optix variable %s, in pass %s", pUniformName, aPass->getName().c_str());
			if (id < 0)
				NAU_THROW("id must be non negative, in optix variable %s, in pass %s", pUniformName, aPass->getName().c_str());
		}
		std::string s(pType);
		if (s == "TEXTURE") {
			if (!RESOURCEMANAGER->hasTexture(pContext)) {
				NAU_THROW("Texture %s is not defined, in pass %s", pContext, aPass->getName().c_str());
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
				NAU_THROW("Light %s is not defined in the project file", pContext);
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
			NAU_THROW("No optix variable name, in pass %s", aPass->getName().c_str());
		}
		if (0 == pType) {
			NAU_THROW("No type found for optix variable %s, in pass %s", pUniformName, aPass->getName().c_str());
		}
		if (0 == pContext) {
			NAU_THROW("No context found for optix variable %s, in pass %s", pUniformName, aPass->getName().c_str());
		}
		if (0 == pComponent) {
			NAU_THROW("No component found for optix variable %s, in pass %s", pUniformName, aPass->getName().c_str());
		}
		if (!ProgramValue::Validate(pType, pContext, pComponent))
			NAU_THROW("Optix variable %s is not valid, in pass %s", pUniformName, aPass->getName().c_str());

		int id = 0;
		if (((strcmp(pContext,"LIGHT") == 0) || (0 == strcmp(pContext,"TEXTURE"))) &&  (0 != strcmp(pComponent,"COUNT"))) {
			if (TIXML_SUCCESS != pElemAux2->QueryIntAttribute ("id", &id))
				NAU_THROW("No id found for optix variable %s, in pass %s", pUniformName, aPass->getName().c_str());
			if (id < 0)
				NAU_THROW("id must be non negative, in optix variable %s, in pass %s", pUniformName, aPass->getName().c_str());
		}
		std::string s(pType);
		if (s == "TEXTURE") {
			if (!RESOURCEMANAGER->hasTexture(pContext)) {
				NAU_THROW("Texture %s is not defined, in pass %s", pContext, aPass->getName().c_str());
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
				NAU_THROW("Light %s is not defined in the project file", pContext);
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
			NAU_THROW("No optix variable name, in pass %s", aPass->getName().c_str());
		}
		if (0 == pType) {
			NAU_THROW("No type found for optix variable %s, in pass %s", pUniformName, aPass->getName().c_str());
		}
		if (0 == pContext) {
			NAU_THROW("No context found for optix variable %s, in pass %s", pUniformName, aPass->getName().c_str());
		}
		if (0 == pComponent) {
			NAU_THROW("No component found for optix variable %s, in pass %s", pUniformName, aPass->getName().c_str());
		}
		if (!ProgramValue::Validate(pType, pContext, pComponent))
			NAU_THROW("Optix variable %s is not valid, in pass %s", pUniformName, aPass->getName().c_str());

		int id = 0;
		if (((strcmp(pContext,"LIGHT") == 0) || (0 == strcmp(pContext,"TEXTURE"))) &&  (0 != strcmp(pComponent,"COUNT"))) {
			if (TIXML_SUCCESS != pElemAux2->QueryIntAttribute ("id", &id))
				NAU_THROW("No id found for optix variable %s, in pass %s", pUniformName, aPass->getName().c_str());
			if (id < 0)
				NAU_THROW("id must be non negative, in optix variable %s, in pass %s", pUniformName, aPass->getName().c_str());
		}
		std::string s(pType);
		if (s == "TEXTURE") {
			if (!RESOURCEMANAGER->hasTexture(pContext)) {
				NAU_THROW("Texture %s is not defined, in pass %s", pContext, aPass->getName().c_str());
			}
			else
				po->addGlobalAttribute (pUniformName, ProgramValue (pUniformName,pType, pContext, pComponent, id));
		}

		else if (s == "CAMERA") {
			// Must consider that a camera can be defined internally in a pass, example:lightcams
			/*if (!RENDERMANAGER->hasCamera(pContext))
				NAU_THROW("Camera %s is not defined in the project file", pContext);*/
			po->addGlobalAttribute (pUniformName, ProgramValue (pUniformName,pType, pContext, pComponent, id));
		}
		else if (s == "LIGHT") {
			if (!RENDERMANAGER->hasLight(pContext))
				NAU_THROW("Light %s is not defined in the project file", pContext);
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
#ifdef NAU_OPTIX_PRIME 

void
ProjectLoader::loadPassOptixPrimeSettings(TiXmlHandle hPass, Pass *aPass) {

#if NAU_OPENGL_VERSION >= 420
	TiXmlElement *pElem;
	PassOptixPrime *p = (PassOptixPrime *)aPass;

	pElem = hPass.FirstChildElement("scene").Element();
	if (pElem != NULL) {
		const char *pSceneName = pElem->Attribute("name");

		if (pSceneName != NULL) {

			if (!RENDERMANAGER->hasScene(pSceneName)) {
				NAU_THROW("Pass %s: Scene %s is not defined", aPass->getName().c_str(), pSceneName);
			}
			else
				p->addScene(pSceneName);
		}
	}
	else {
		NAU_THROW("Pass %s: Scene is not defined", aPass->getName().c_str());
	}

	pElem = hPass.FirstChildElement("rays").Element();
	if (pElem != NULL) {
		const char *pBufferName = pElem->Attribute("buffer");
		const char *pQueryType = pElem->Attribute("queryType");

		if (pBufferName != NULL) {

			if (!RESOURCEMANAGER->hasBuffer(pBufferName)) {
				NAU_THROW("Pass %s: Ray buffer %s is not defined", aPass->getName().c_str(), pBufferName);
			}
			else
				p->addRayBuffer(RESOURCEMANAGER->getBuffer(pBufferName));
		}
		else {
			NAU_THROW("Pass %s: Ray buffer has no name", aPass->getName().c_str());
		}
		if (pQueryType != NULL) {
			p->setQueryType(pQueryType);
		}
		else {
			NAU_THROW("Pass %s: Ray buffer %s: Query Type not defined", aPass->getName().c_str(), pBufferName);
		}
	}
	else {
		NAU_THROW("Pass %s: Ray buffer is not defined", aPass->getName().c_str());
	}

	pElem = hPass.FirstChildElement("hits").Element();
	if (pElem != NULL) {
		const char *pBufferName = pElem->Attribute("buffer");

		if (pBufferName != NULL) {

			if (!RESOURCEMANAGER->hasBuffer(pBufferName)) {
				NAU_THROW("Pass %s: Hit Buffer %s is not defined", aPass->getName().c_str(), pBufferName);
			}
			else
				p->addHitBuffer(RESOURCEMANAGER->getBuffer(pBufferName));
		}
		else {
			NAU_THROW("Pass %s: Hit buffer has no name", aPass->getName().c_str(), pBufferName);
		}
	}
	else {
		NAU_THROW("Pass %s: Hit buffer is not defined", aPass->getName().c_str());
	}

#endif
}


#endif
/* ----------------------------------------------------------------------------

	COMPUTE SETTINGS

			<pass class="compute" name="test">
				<material mat="computeShader" lib="myLib" WdimX=256, WdimY=256, WdimZ=0/>
				<!-- may have params, camera, viewport, lights -->
			</pass>	
-------------------------------------------------------------------------------*/



void
ProjectLoader::loadPassComputeSettings(TiXmlHandle hPass, Pass *aPass) {

	TiXmlElement *pElem;
	PassCompute *p = (PassCompute *)aPass;

	pElem = hPass.FirstChildElement("material").Element();
	if (pElem != NULL) {
		const char *pMatName = pElem->Attribute ("mat");
		const char *pLibName = pElem->Attribute ("lib");
		if (pMatName != NULL && pLibName != NULL) {
			if (!MATERIALLIBMANAGER->hasMaterial(pLibName,pMatName))
				NAU_THROW("Pass %s: Material %s::%s is not defined", aPass->getName().c_str(), pLibName, pMatName);
		}
		else 
			NAU_THROW("Pass %s: Material not defined", aPass->getName().c_str());

		int dimX, dimY, dimZ;
		if (TIXML_SUCCESS != pElem->QueryIntAttribute("dimX", &dimX)) 
			NAU_THROW("Pass %s: dimX is not defined", aPass->getName().c_str());

		if (TIXML_SUCCESS != pElem->QueryIntAttribute("dimY", &dimY))
			dimY = 1;
		if (TIXML_SUCCESS != pElem->QueryIntAttribute("dimZ", &dimZ))
			dimZ = 1;

		p->setMaterialName (pLibName, pMatName);
		p->setDimension( dimX, dimY, dimZ);	
	}
	else
		NAU_THROW("Pass %s: Missing material", aPass->getName().c_str());
	
}

/* -----------------------------------------------------------------------------
MATERIAL LIBRAY RENDERTARGET DEFINITION

	<rendertargets>
		<rendertarget name = "test" />
			<size width=2048 height=2048 samples=16 layers = 2/>
			<clear r=0.0 g=0.0 b=0.0 />
			<colors>
				<color name="offscreenrender" internalFormat="RGBA32F" />
			</colors>
			<depth name="shadowMap1"  internalFormat="DEPTH_COMPONENT24"  />
				or
			<depthStencil name="bli" />
		</rendertarget>
	</rendertargets>

The names of both color and depth RTs will be available for other passes to use
	as textures

layers is optional, by default regular (non-array) textures are built.

There can be multiple color, but only one depth.
depth and depthStencil can't be both defined in the same render target 
Depth and Color can be omitted, but at least one of them must be present.
Setting samples to 0 or 1 implies no multisampling
-----------------------------------------------------------------------------*/

void 
ProjectLoader::loadMatLibRenderTargets(TiXmlHandle hRoot, MaterialLib *aLib, std::string path)
{
	TiXmlElement *pElem;
	RenderTarget *m_RT;
	int rtWidth, rtHeight, rtSamples = 0, rtLayers = 0; 

	pElem = hRoot.FirstChild ("rendertargets").FirstChild ("rendertarget").Element();
	for ( ; 0 != pElem; pElem=pElem->NextSiblingElement()) {
		const char *pRTName = pElem->Attribute ("name");

		if (!pRTName)
			NAU_THROW("Library %s: Render Target has no name", aLib->getName());

		// Render Target size
		TiXmlElement *pElemSize;
		pElemSize = pElem->FirstChildElement ("size");
			
		if (0 != pElemSize) {
			if (TIXML_SUCCESS != pElemSize->QueryIntAttribute ("width", &rtWidth)) {
				NAU_THROW("Library %s: Render Target %s: WIDTH is required", aLib->getName().c_str(), pRTName);
			}

			if (TIXML_SUCCESS != pElemSize->QueryIntAttribute ("height",  &rtHeight)) {
				NAU_THROW("Library %s: Render Target %s: HEIGHT is required", aLib->getName().c_str(), pRTName);
			}

			sprintf(s_pFullName, "%s::%s", aLib->getName().c_str(), pRTName);
			m_RT = RESOURCEMANAGER->createRenderTarget (s_pFullName, rtWidth, rtHeight);	
			
			if (TIXML_SUCCESS == pElemSize->QueryIntAttribute ("samples", &rtSamples)) {
				m_RT->setSampleCount(rtSamples);
			}

			if (TIXML_SUCCESS == pElemSize->QueryIntAttribute ("layers", &rtLayers)) {
				m_RT->setLayerCount(rtLayers);
			}
		} 
		else {
			NAU_THROW("Library %s: Render Target %s: No size element found", aLib->getName().c_str(), pRTName);
		} //End of  rendertargets size


		// Render Target clear values
		TiXmlElement *pElemClear;
		pElemClear = pElem->FirstChildElement ("clear");
			
		if (0 != pElemClear) {

			float r = 0.0f, g = 0.0f, b = 0.0f, a = 0.0f; 
			pElemClear->QueryFloatAttribute ("r", &r);
			pElemClear->QueryFloatAttribute ("g", &g);
			pElemClear->QueryFloatAttribute ("b", &b);
			pElemClear->QueryFloatAttribute ("a", &a);

			m_RT->setClearValues(r,g,b,a);
		} //End of  rendertargets clear


		TiXmlElement *pElemDepth;
		pElemDepth = pElem->FirstChildElement ("depth");

		if (0 != pElemDepth) {
			const char *pNameDepth = pElemDepth->Attribute ("name");
			const char *internalFormat = pElemDepth->Attribute("internalFormat");

			if (0 == pNameDepth) {
				NAU_THROW("Library %s: Render Target %s: Depth rendertarget has no name", aLib->getName().c_str(), pRTName);							
			}

			if (internalFormat == 0) {
				NAU_THROW("Library %s: Render Target %s: Depth rendertarget %s has no internal format", aLib->getName().c_str(), pRTName, pNameDepth);
			}

			if (!Texture::Attribs.isValid("INTERNAL_FORMAT", internalFormat)) //isValidInternalFormat(internalFormat))
					NAU_THROW("Library %s: Render Target %s: Depth rendertarget's internal format %s is invalid", aLib->getName().c_str(), pRTName, internalFormat);
			//int intFormat = Texture::Attribs.getListValueOp(Texture::INTERNAL_FORMAT, internalFormat);
			//if (intFormat < 0) 
			//	NAU_THROW("Library %s: Render Target %s: Depth rendertarget's internal format %s is invalid", aLib->getName().c_str(), pRTName, internalFormat);

			sprintf(s_pFullName, "%s::%s", aLib->getName().c_str(), pNameDepth);
			m_RT->addDepthTarget (s_pFullName, internalFormat);
		}

		//pElemDepth = pElem->FirstChildElement ("stencil");

		//if (0 != pElemDepth) {
		//	const char *pNameDepth = pElemDepth->Attribute ("name");

		//	if (0 == pNameDepth) {
		//		NAU_THROW("Stencil rendertarget has no name, in render target %s,in material lib: %s", pRTName, aLib->getName().c_str());							
		//	}
		//				
		//	sprintf(s_pFullName, "%s::%s", aLib->getName().c_str(), pNameDepth);
		//	m_RT->addStencilTarget (s_pFullName);
		//}

		pElemDepth = pElem->FirstChildElement ("depthStencil");

		if (0 != pElemDepth) {
			const char *pNameDepth = pElemDepth->Attribute ("name");

			if (0 == pNameDepth) {
				NAU_THROW("Depth/Stencil rendertarget has no name, in render target %s,in material lib: %s", pRTName, aLib->getName().c_str());							
			}
						
			sprintf(s_pFullName, "%s::%s", aLib->getName().c_str(), pNameDepth);
			m_RT->addDepthStencilTarget (s_pFullName);
		}

		TiXmlElement *pElemColor;	
		TiXmlNode *pElemColors;
		pElemColors = pElem->FirstChild("colors");
		if (pElemColors != NULL) {
			pElemColor = pElemColors->FirstChildElement("color");

			for ( ; 0 != pElemColor; pElemColor = pElemColor->NextSiblingElement()) {

				const char *pNameColor = pElemColor->Attribute ("name");
				const char *internalFormat = pElemColor->Attribute("internalFormat");
				int layers = 0;

				if (pElemColor->QueryIntAttribute("layers", &layers))
					if (layers < 2 && layers != 0)
						NAU_THROW("Number of layers must be greater or equal to 2, in render target %s,in material lib: %s", pRTName, aLib->getName().c_str());	

				if (0 == pNameColor) {
					NAU_THROW("Color rendertarget has no name, in render target %s,in material lib: %s", pRTName, aLib->getName().c_str());							
				}

				if (internalFormat == 0) {
					NAU_THROW("Color rendertarget %s has no internal format, in render target %s, in material lib: %s", pNameColor, pRTName, aLib->getName().c_str());
				}
				if (!Texture::Attribs.isValid("INTERNAL_FORMAT", internalFormat)) //isValidInternalFormat(internalFormat))
					NAU_THROW("Library %s: Render Target %s: Color rendertarget %s internal format %s is invalid", aLib->getName().c_str(), pRTName, pNameColor,internalFormat);

				sprintf(s_pFullName, "%s::%s", aLib->getName().c_str(), pNameColor);
					
				m_RT->addColorTarget (s_pFullName, internalFormat);
			}//End of rendertargets color
		}
		SLOG("Render Target : %s width:%d height:%d samples:%d layers:%d", pRTName, rtWidth, rtHeight, rtSamples, rtLayers);

		if (!m_RT->checkStatus()) {
			NAU_THROW("Render target is not OK, in render target %s,in material lib: %s", pRTName, aLib->getName().c_str());							

		}
	}//End of rendertargets


}

/* -----------------------------------------------------------------------------
EVENTS

THEY NEED TO BE CHECKED IF EVERY ASSET REQUIRED IS DEFINED	

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
			NAU_THROW("Sensor %d has no class in file %s", pName, ProjectLoader::s_File.c_str());
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


	SceneObject *o;
	pElem = handle.FirstChild ("moveableobjects").FirstChild ("moveableobject").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
		const char *pName = pElem->Attribute ("name");
		const char *pObject = pElem->Attribute("object");

		if (0 == pName) 
			NAU_THROW("MoveableObject has no name in file %s ", ProjectLoader::s_File.c_str());

		if (0 == pObject) {
			o=0;
		}

		o = RENDERMANAGER->getScene("MainScene")->getSceneObject(pObject); // substituir o MainScene, pode ter mais do q uma Cena?

		nau::event_::ObjectAnimation *oa= new nau::event_::ObjectAnimation(pName, o);

		//in->init((char *) pName, o, (char *)pKey, (char *)pKeyValue); 
		
	}
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

		//int pPriority; 
		//if (TIXML_SUCCESS != pElem->QueryIntAttribute("fromPriority",&pPriority))
		//	pPriority = -1;

		//if (-1 != pPriority && pFromMaterial)

		//	NAU_THROW("A Material Map can be used for materials or priorities, not both simultaneously. In pass: %s", aPass->getName().c_str());

		if (/*-1 != pPriority && */(pToMaterial == 0)) {

		    NAU_THROW("Material map error in pass : %s", aPass->getName().c_str());
		}
		if (/*pPriority == -1 && */( 
		    (0 != pFromMaterial && 0 == pToMaterial) ||
		    (0 == pFromMaterial && 0 != pToMaterial))) {
		  
		    NAU_THROW("Material map error in pass: %s", aPass->getName().c_str());
		}

		//if (pPriority != -1) {

		//	if (MATERIALLIBMANAGER->hasMaterial(library, pToMaterial))
		//		aPass->remapAllFromPriority(pPriority, library, pToMaterial);
		//	else
		//		NAU_THROW("Material Map Error, destination material (%s,%s) is not defined, in  pass: %s", library, pToMaterial,aPass->getName().c_str());
		//}
		
		else if (0 == pFromMaterial) {
			if (MATERIALLIBMANAGER->hasLibrary(library))
				aPass->remapAll (library);
			else
				NAU_THROW("Material Map Error, destination library %s is not defined, in  pass: %s", library.c_str(), aPass->getName().c_str());
		}
		else if (0 == strcmp (pFromMaterial, "*")) {
			if (MATERIALLIBMANAGER->hasMaterial(library, pToMaterial))
				aPass->remapAll (library, pToMaterial);
			else
				NAU_THROW("Material Map Error, destination material (%s,%s) is not defined, in  pass: %s", library.c_str(), pToMaterial,aPass->getName().c_str());
		}
		else {
			if (MATERIALLIBMANAGER->hasMaterial(library, pToMaterial))
				aPass->remapMaterial (pFromMaterial, library, pToMaterial);
			else
				NAU_THROW("Material Map Error, destination material (%s,%s) is not defined, in  pass: %s", library.c_str(), pToMaterial,aPass->getName().c_str());
		}
	} //End of map

}

/* -----------------------------------------------------------------------------
SHADERMAPS

	<shadermaps>
		<attach shader="perpixel-color" fromLibrary="bla" toMaterial="c"/>
	</shadermaps>

Creates a new material in the pass material lib, cloning the default's
library material "c" and adding the shader to it

NOTE: This would more usefull if allowed to attach shaders and textures and states
to a material
-----------------------------------------------------------------------------*/
//
//void 
//ProjectLoader::loadPassShaderMaps(TiXmlHandle hPass, Pass *aPass)
//{
//	TiXmlElement *pElem;
//
//	pElem = hPass.FirstChild("shadermaps").FirstChild("attach").Element();
//	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
//		const char *pShader = pElem->Attribute ("shader");
//		const char *pLib = pElem->Attribute("fromLibrary");
//		const char *pToMaterial = pElem->Attribute ("toMaterial");
//
//		if (0 == pShader || 0 == pToMaterial || 0 == pLib) {
//		  NAU_THROW("Shader map error in pass: %s", aPass->getName().c_str());
//		}
//
//		sprintf(s_pFullName, "%s::%s", pLib, pShader);
//		if (!RESOURCEMANAGER->hasProgram(s_pFullName))
//			NAU_THROW("Shader %s is not defined in lib %s, in pass: %s", pShader, pLib,aPass->getName().c_str());
//
//		//aPass->attachShader(pShader, pToMaterial);
//
//		MaterialLib *aLib = MATERIALLIBMANAGER->getLib(aPass->getName());
//		MaterialLib *defLib = MATERIALLIBMANAGER->getLib(DEFAULTMATERIALLIBNAME);
//
//		std::vector<std::string> *names = defLib->getMaterialNames(pToMaterial);
//		std::vector<std::string>::iterator iter;
//
//		for(iter = names->begin(); iter != names->end(); ++iter) {
//
//			std::string name = *iter;
//			Material *srcMat = defLib->getMaterial(name);
//			Material *dstMat = srcMat->clone();
//			MATERIALLIBMANAGER->addMaterial(aPass->getName(), dstMat);
//			dstMat->attachProgram(s_pFullName);
//
//			aPass->remapMaterial (name, aPass->getName(), name);
//		}
//		delete names;
//	}
//
//}

/* -----------------------------------------------------------------------------
STATEMAPS

	<statemaps>
		<set inMaterial="Grade*" state="Grades" />
	</statemaps>

Creates a new material in the passe's library with the same name, and the same 
values of the original material, and attaches a state to it.

An * can be used at the end of the material, and it works as a wildcard.

-----------------------------------------------------------------------------*/

//void 
//ProjectLoader::loadPassStateMaps(TiXmlHandle hPass, Pass *aPass)
//{
//	TiXmlElement *pElem;
//
//	pElem = hPass.FirstChild ("statemaps").FirstChild ("set").Element();
//	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
//		const char *pInMaterial = pElem->Attribute ("inMaterial");
//		const char *pState = pElem->Attribute ("state");
//		const char *pLib = pElem->Attribute ("fromLibrary");
//
//		if (0 == pInMaterial || 0 == pState || 0 == pLib) {
//		  NAU_THROW("State map error in pass: %s", aPass->getName().c_str());
//		}
//
//		sprintf(s_pFullName, "%s::%s", pLib, pState);
//		if (!RESOURCEMANAGER->hasState(s_pFullName))
//			NAU_THROW("State %s is not defined in lib %s, in pass: %s", pState, pLib,aPass->getName().c_str());
//
//		MaterialLib *aLib = MATERIALLIBMANAGER->getLib(aPass->getName());
//		MaterialLib *defLib = MATERIALLIBMANAGER->getLib(DEFAULTMATERIALLIBNAME);
//
//		std::vector<std::string> *names = defLib->getMaterialNames(pInMaterial);
//		std::vector<std::string>::iterator iter;
//
//		for(iter = names->begin(); iter != names->end(); ++iter) {
//
//			std::string name = *iter;
//			Material *srcMat = defLib->getMaterial(name);
//			Material *dstMat = srcMat->clone();
//			MATERIALLIBMANAGER->addMaterial(aPass->getName(), dstMat);
//			dstMat->setState(RESOURCEMANAGER->getState(s_pFullName));
//
//			aPass->remapMaterial (name, aPass->getName(), name);
//		}
//		delete names;
//	} //End of map
//}


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

	pElem = hPass.FirstChild ("injectionMaps").FirstChild ("map").Element();
	for ( ; pElem != NULL; pElem = pElem->NextSiblingElement()) {
	
		const char* pMat = pElem->Attribute("toMaterial");

		if (0 == pMat)
			NAU_THROW("Map error: a name is required for the material, in pass: %s", aPass->getName().c_str());
	
		MaterialLib *defLib = MATERIALLIBMANAGER->getLib(DEFAULTMATERIALLIBNAME);
		std::vector<std::string> *names = defLib->getMaterialNames(pMat);

		if (names->size() == 0)
			NAU_THROW("No materials match %s in map definition, in pass: %s", pMat, aPass->getName().c_str());

		Material *dstMat, *srcMat;

		std::vector<std::string>::iterator iter;
		for(iter = names->begin(); iter != names->end(); ++iter) {

			std::string name = *iter;
			Material *srcMat = defLib->getMaterial(name);
			dstMat = srcMat->clone();
			MATERIALLIBMANAGER->addMaterial(aPass->getName(), dstMat);

			aPass->remapMaterial (name, aPass->getName(), name);
		}


		pElemAux = pElem->FirstChildElement("state");
		if (pElemAux) {
	
			const char *pState = pElemAux->Attribute ("name");
			const char *pLib = pElemAux->Attribute ("fromLibrary");

			if (0 == pState || 0 == pLib) {
			  NAU_THROW("State map error in pass: %s", aPass->getName().c_str());
			}

			sprintf(s_pFullName, "%s::%s", pLib, pState);
			if (!RESOURCEMANAGER->hasState(s_pFullName))
				NAU_THROW("State %s is not defined in lib %s, in pass: %s", pState, pLib,aPass->getName().c_str());
		
			std::vector<std::string>::iterator iter;
			for(iter = names->begin(); iter != names->end(); ++iter) {

				std::string name = *iter;
				dstMat = MATERIALLIBMANAGER->getMaterial(aPass->getName(), name);
				dstMat->setState(RESOURCEMANAGER->getState(s_pFullName));
			}
		}
	
		pElemAux = pElem->FirstChildElement("shader");
		if (pElemAux) {

			const char *pMat = pElemAux->Attribute ("fromMaterial");
			const char *pLib = pElemAux->Attribute ("fromLibrary");

			if (0 == pMat || 0 == pLib) {
			  NAU_THROW("State map error in pass: %s", aPass->getName().c_str());
			}

			if (!MATERIALLIBMANAGER->hasMaterial(pLib,pMat))
				NAU_THROW("Material %s is not defined in lib %s, in pass: %s", pMat, pLib,aPass->getName().c_str());

			std::vector<std::string>::iterator iter;
			for(iter = names->begin(); iter != names->end(); ++iter) {

				std::string name = *iter;
				dstMat = MATERIALLIBMANAGER->getMaterial(aPass->getName(), name);
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
				  NAU_THROW("Texture map error in pass: %s", aPass->getName().c_str());
				}

				int unit;
				if (TIXML_SUCCESS != pElemAux->QueryIntAttribute("toUnit", &unit))
				  NAU_THROW("Texture unit not specified in material map, in pass: %s", aPass->getName().c_str());

				sprintf(s_pFullName, "%s::%s", pLib, pName);
				if (!RESOURCEMANAGER->hasTexture(s_pFullName))
					NAU_THROW("Texture %s is not defined in lib %s, in pass: %s", pName, pLib,aPass->getName().c_str());

				std::vector<std::string>::iterator iter;
				for(iter = names->begin(); iter != names->end(); ++iter) {

					std::string name = *iter;
					dstMat = MATERIALLIBMANAGER->getMaterial(aPass->getName(), name);
					dstMat->attachTexture(unit, RESOURCEMANAGER->getTexture(s_pFullName));
					std::map<std::string, Attribute> attribs = TextureSampler::Attribs.getAttributes();
					TiXmlElement *p = pElemAux->FirstChildElement();
					Attribute a;
					void *value;
					while (p) {
						// trying to define an attribute that does not exist		
						if (attribs.count(p->Value()) == 0)
							NAU_THROW("Pass %s: Texture %s: %s is not an attribute", aPass->getName().c_str(),  s_pFullName, p->Value());
						// trying to set the value of a read only attribute
						a = attribs[p->Value()];
						if (a.mReadOnlyFlag)
							NAU_THROW("Pass %s: Texture %s: %s is a read-only attribute, in file %s", aPass->getName().c_str(),  s_pFullName, p->Value());

						value = readAttr(s_pFullName, p, a.mType, TextureSampler::Attribs);
						dstMat->getTextureSampler(unit)->setProp(a.mId, a.mType, value);
						p = p->NextSiblingElement();
					}
				}

		
			}
		}
#if NAU_OPENGL_VERSION >= 420
		pElemNode = pElem->FirstChild("buffers");
		if (pElemNode) {
			pElemAux = pElemNode->FirstChildElement("buffer");
			for (; pElemAux != NULL; pElemAux = pElemAux->NextSiblingElement()) {

				const char *pName = pElemAux->Attribute("name");
				const char *pLib = pElemAux->Attribute("fromLibrary");


				if (0 == pName || 0 == pLib) {
					NAU_THROW("Buffer map error in pass: %s", aPass->getName().c_str());
				}

				sprintf(s_pFullName, "%s::%s", pLib, pName);
				if (!RESOURCEMANAGER->hasBuffer(s_pFullName))
					NAU_THROW("Buffer %s is not defined in lib %s, in pass: %s", pName, pLib, aPass->getName().c_str());

				std::vector<std::string>::iterator iter;
				for (iter = names->begin(); iter != names->end(); ++iter) {

					std::string name = *iter;
					dstMat = MATERIALLIBMANAGER->getMaterial(aPass->getName(), name);
					IBuffer *buffer = RESOURCEMANAGER->getBuffer(s_pFullName);
					IBuffer *b = buffer->clone();

					std::map<std::string, Attribute> attribs = IBuffer::Attribs.getAttributes();
					TiXmlElement *p = pElemAux->FirstChildElement();
					Attribute a;
					void *value;
					while (p) {
						// trying to define an attribute that does not exist		
						if (attribs.count(p->Value()) == 0)
							NAU_THROW("Pass %s: Buffer %s: %s is not an attribute", aPass->getName().c_str(), s_pFullName, p->Value());
						// trying to set the value of a read only attribute
						a = attribs[p->Value()];
						if (a.mReadOnlyFlag)
							NAU_THROW("Pass %s: Buffer %s: %s is a read-only attribute, in file %s", aPass->getName().c_str(), s_pFullName, p->Value());

						value = readAttr(s_pFullName, p, a.mType, IBuffer::Attribs);
						b->setProp(a.mId, a.mType, value);
						p = p->NextSiblingElement();
					}
					dstMat->attachBuffer(b);
				}


			}
		}

#endif
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
			  NAU_THROW("State map error in pass: %s", aPass->getName().c_str());
			}

			if (!MATERIALLIBMANAGER->hasMaterial(pLib,pMat))
				NAU_THROW("Material %s is not defined in lib %s, in pass: %s", pMat, pLib,aPass->getName().c_str());

			srcMat = MATERIALLIBMANAGER->getMaterial(pLib,pMat);
			std::vector<std::string>::iterator iter;
			for(iter = names->begin(); iter != names->end(); ++iter) {

				std::string name = *iter;
				dstMat = MATERIALLIBMANAGER->getMaterial(aPass->getName(), name);
				if (!pAmbient && !pDiffuse && !pSpecular && !pEmission && !pShininess)
					dstMat->getColor().clone(srcMat->getColor());
				if (pAmbient && !strcmp("true",pAmbient))
					dstMat->getColor().setProp(ColorMaterial::AMBIENT, srcMat->getColor().getPropf4(ColorMaterial::AMBIENT));
				if (pDiffuse && !strcmp("true",pDiffuse))
					dstMat->getColor().setProp(ColorMaterial::DIFFUSE, srcMat->getColor().getPropf4(ColorMaterial::DIFFUSE));
				if (pSpecular && !strcmp("true",pSpecular))
					dstMat->getColor().setProp(ColorMaterial::SPECULAR, srcMat->getColor().getPropf4(ColorMaterial::SPECULAR));
				if (pEmission && !strcmp("true",pEmission))
					dstMat->getColor().setProp(ColorMaterial::EMISSION, srcMat->getColor().getPropf4(ColorMaterial::EMISSION));
				if (pShininess && !strcmp("true",pShininess))
					dstMat->getColor().setProp(ColorMaterial::SHININESS, srcMat->getColor().getPropf(ColorMaterial::SHININESS));
			}

		}
		delete names;
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
ProjectLoader::loadPipelines (TiXmlHandle &hRoot)
{
	TiXmlElement *pElem;
	TiXmlHandle handle (0);
	std::map<std::string, Pass*> passMapper;


	char activePipeline[256];

	memset (activePipeline, 0, 256);

	pElem = hRoot.FirstChild ("pipelines").FirstChild ("pipeline").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
		const char *pNamePip = pElem->Attribute ("name");
		const char *pDefault = pElem->Attribute ("default");
		const char *pDefaultCamera = pElem->Attribute("defaultCamera");

		if (0 == pNamePip) 
			NAU_THROW("Pipeline has no name");

		if (pDefaultCamera && !(RENDERMANAGER->hasCamera(pDefaultCamera)))
			NAU_THROW("Camera %s, defined as default in pipeline %s, is not defined", pDefaultCamera, pNamePip);

		if (RENDERMANAGER->hasPipeline(pNamePip))
			NAU_THROW("Pipeline %s is already defined in file %s", pNamePip, ProjectLoader::s_File.c_str());
		Pipeline *aPipeline = RENDERMANAGER->getPipeline (pNamePip);
		
		// if no default pipeline is set, then the first pipeline will be the default
		if (0 == strcmp (pDefault, "true")) {
			strcpy (activePipeline, pNamePip);
		}

		handle = TiXmlHandle (pElem);
		TiXmlElement *pElemPass;


		pElemPass = handle.FirstChild ("pass").Element();
		for ( ; 0 != pElemPass; pElemPass = pElemPass->NextSiblingElement()) {
			
			TiXmlHandle hPass (pElemPass);
			const char *pName = pElemPass->Attribute ("name");
			const char *pClass = pElemPass->Attribute ("class");

			if (0 == pName) 
				NAU_THROW("Pass has no name in file %s", ProjectLoader::s_File.c_str());

			if (RENDERMANAGER->hasPass(pNamePip, pName))
				NAU_THROW("Pass %s is defined more than once, in pipeline %s, in file %s", pName, pNamePip, ProjectLoader::s_File.c_str());

			Pass *aPass = 0;
			if (0 == pClass) {
				aPass = aPipeline->createPass(pName);
			} else {
				if (PassFactory::isClass(pClass))
					aPass = aPipeline->createPass (pName, pClass);
				else
					NAU_THROW("Class %s is not available, in pass (%s,%s), in file %s", pClass, pNamePip, pName, ProjectLoader::s_File.c_str());

			}
			passMapper[pName] = aPass;
					
			if (0 != strcmp(pClass, "optixPrime") && 0 != strcmp(pClass, "quad") && 0 != strcmp(pClass, "profiler")) {

				loadPassScenes(hPass,aPass);
				loadPassCamera(hPass,aPass);			
			}
			else
				loadPassTexture(hPass,aPass);
#ifdef NAU_OPTIX
			if (0 == strcmp("optix", pClass))
				loadPassOptixSettings(hPass, aPass);
#endif
			if (0 == strcmp("compute", pClass))
				loadPassComputeSettings(hPass, aPass);
			
#ifdef NAU_OPTIX_PRIME
			if (0 == strcmp("optixPrime", pClass))
				loadPassOptixPrimeSettings(hPass, aPass);
#endif

			loadPassParams(hPass, aPass);
			loadPassViewport(hPass, aPass);
			loadPassClearDepthAndColor(hPass, aPass);
			loadPassRenderTargets(hPass, aPass, passMapper);
			loadPassLights(hPass, aPass);

			loadPassInjectionMaps(hPass, aPass);
			loadPassMaterialMaps(hPass, aPass);
			//loadPassStateMaps(hPass, aPass);
			//loadPassShaderMaps(hPass, aPass);
		} //End of pass
		if (pDefaultCamera)
			aPipeline->setDefaultCamera(pDefaultCamera);

		//else
		//	aPipeline->setDefaultCamera(aPipeline->getLastPassCameraName());
	} //End of pipeline
	
	if (strlen (activePipeline) > 0) {
		RENDERMANAGER->setActivePipeline (activePipeline);
	} else {
		NAU_THROW("No default pipeline");
	}
}


/* -----------------------------------------------------------------------------
BUFFERS

<buffers>
	<buffer name="bla" size=123 />
</buffers>

All fields are required. Size is in bytes
-----------------------------------------------------------------------------*/


void
ProjectLoader::loadMatLibBuffers(TiXmlHandle hRoot, MaterialLib *aLib, std::string path)
{
#if NAU_OPENGL_VERSION >= 420
	TiXmlElement *pElem;
	int layers = 0;
	pElem = hRoot.FirstChild("buffers").FirstChild("buffer").Element();
	for (; 0 != pElem; pElem = pElem->NextSiblingElement()) {
		const char* pName = pElem->Attribute("name");

		if (0 == pName) {
			NAU_THROW("Mat Lib %s: Buffer has no name", aLib->getName().c_str());
		}
		sprintf(s_pFullName, "%s::%s", aLib->getName().c_str(), pName);

		SLOG("Buffer : %s", s_pFullName);

		if (RESOURCEMANAGER->hasBuffer(s_pFullName)) {
			NAU_THROW("Mat Lib %s: Buffer %s is already defined", aLib->getName().c_str(), s_pFullName);
		}
		int size;
		if (pElem->QueryIntAttribute("size", &size)) {
			NAU_THROW("Mat Lib %s: Buffer %s: has no size", aLib->getName().c_str(), pName);
		}

		if (size < 0) {
			NAU_THROW("Mat Lib %s: Buffer %s : size must be greater than zero", aLib->getName().c_str(), pName);
		}

		RESOURCEMANAGER->createBuffer(s_pFullName, size);
	}
#endif
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
		<texture name ="Bla"
			width=512 height = 512 internalFormat="RGBA" layers = 2/>
	</textures>

Layers are an optional field. If specified a 2D texture array will be created.
The paths may be relative to the material lib file, or absolute.
-----------------------------------------------------------------------------*/
void 
ProjectLoader::loadMatLibTextures(TiXmlHandle hRoot, MaterialLib *aLib, std::string path)
{
	TiXmlElement *pElem;
	int layers = 0;
	pElem = hRoot.FirstChild ("textures").FirstChild ("texture").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
		const char* pTextureName = pElem->Attribute ("name");
		const char* pFilename = pElem->Attribute ("filename");
		const char *internalFormat = pElem->Attribute("internalFormat");

		// layers
		if (!pElem->QueryIntAttribute("layers", &layers))
			layers = 0;
		else if (layers < 2 && layers != 0)
			NAU_THROW("Mat Lib &s - Texture %s - Layers must be equal or greater than 2", aLib->getName().c_str(), s_pFullName);
			
		//const char *type = pElem->Attribute("type");
		//const char *format = pElem->Attribute("format");
		int mipmap=0;
		const char *pMipMap = pElem->Attribute("mipmap", &mipmap);



		if (0 == pTextureName) {
			NAU_THROW("Texture has no name in Mat Lib %s", aLib->getName().c_str());
		} 

		sprintf(s_pFullName,"%s::%s", aLib->getName().c_str(), pTextureName);

		SLOG("Texture : %s", s_pFullName);

		if (RESOURCEMANAGER->hasTexture(s_pFullName))
			NAU_THROW("Texture %s is already defined, in Mat Lib %s", s_pFullName, aLib->getName().c_str());

		if (0 == pFilename && 0 == internalFormat /*&& 0 == type && 0 == format */) {
			NAU_THROW("Incomplete texture %s definition Mat Lib %s", pTextureName, aLib->getName().c_str());
		}

		// empty texture
		if (0 == pFilename) {
			if (0 == internalFormat  /*|| 0 == type || 0 == format*/) {
				NAU_THROW("Incomplete texture %s definition Mat Lib %s", pTextureName, aLib->getName().c_str());		
			}
			if (!Texture::Attribs.isValid("INTERNAL_FORMAT", internalFormat))//isValidInternalFormat(internalFormat))
				NAU_THROW("Texture %s internal format is invalid, in material lib: %s", pTextureName, aLib->getName().c_str());

		/*	if (!Texture::Attribs.isValid("FORMAT", format))//isValidFormat(format))
				NAU_THROW("Texture %s format is invalid, in material lib: %s", pTextureName, aLib->getName().c_str());
				*/
			int width, height;
			if (pElem->QueryIntAttribute("width", &width) || pElem->QueryIntAttribute("height", &height))
				NAU_THROW("Texture %s dimensions are missing or invalid, in material lib: %s", pTextureName, aLib->getName().c_str());

			RESOURCEMANAGER->createTexture (s_pFullName, internalFormat, width, height, layers);
		}

		else {
			bool mipmap = false;
			if (pMipMap != 0 && strcmp(pMipMap,"1") == 0)
				mipmap = true;

			RESOURCEMANAGER->addTexture (FileUtil::GetFullPath(path,pFilename), s_pFullName, mipmap);
		}
	}
	pElem = hRoot.FirstChild ("textures").FirstChild ("cubeMap").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
		const char* pTextureName = pElem->Attribute ("name");
		const char* pFilePosX = pElem->Attribute ("filePosX");
		const char* pFileNegX = pElem->Attribute ("fileNegX");
		const char* pFilePosY = pElem->Attribute ("filePosY");
		const char* pFileNegY = pElem->Attribute ("fileNegY");
		const char* pFilePosZ = pElem->Attribute ("filePosZ");
		const char* pFileNegZ = pElem->Attribute ("fileNegZ");
		const char *pMipMap = pElem->Attribute("mipmap");

		if (0 == pTextureName) {
			NAU_THROW("Library %s: Cube Map texture has no name", aLib->getName().c_str());
		} 

		if (!(pFilePosX && pFileNegX && pFilePosY && pFileNegY && pFilePosZ && pFileNegZ)) {
			NAU_THROW("Library %s: Cube Map is not complete", aLib->getName().c_str());
		}

		sprintf(s_pFullName,"%s::%s", aLib->getName().c_str(), pTextureName);
		if (RESOURCEMANAGER->hasTexture(s_pFullName))
			NAU_THROW("Library %s: Cube Map %s is already defined", aLib->getName().c_str(), s_pFullName);

		bool mipmap = false;
		if (pMipMap != 0 && strcmp(pMipMap,"1") == 0)
			mipmap = true;


		std::vector<std::string> files(6);
		files[0] = FileUtil::GetFullPath(path,pFilePosX);
		files[1] = FileUtil::GetFullPath(path,pFileNegX);
		files[2] = FileUtil::GetFullPath(path,pFilePosY);
		files[3] = FileUtil::GetFullPath(path,pFileNegY);
		files[4] = FileUtil::GetFullPath(path,pFilePosZ);
		files[5] = FileUtil::GetFullPath(path,pFileNegZ);

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

	std::map<std::string, Attribute> attribs = IState::Attribs.getAttributes();
	TiXmlElement *p = pElemAux->FirstChildElement();
	Attribute a;
	void *value;
	while (p) {
		// trying to define an attribute that does not exist		
		if (attribs.count(p->Value()) == 0)
			NAU_THROW("Library %s: State %s: %s is not an attribute", aLib->getName().c_str(), s->getName().c_str(), p->Value());
		// trying to set the value of a read only attribute
		a = attribs[p->Value()];
		if (a.mReadOnlyFlag)
			NAU_THROW("Library %s: State %s: %s is a read-only attribute, in file %s", aLib->getName().c_str(),  s->getName().c_str(), p->Value());

		value = readAttr(s->getName(), p, a.mType, IState::Attribs);
		s->setProp(a.mId, a.mType, value);
		//aMat->getTextureSampler(unit)->setProp(a.mId, a.mType, value);
		p = p->NextSiblingElement();
	}
}


void 
ProjectLoader::loadMatLibStates(TiXmlHandle hRoot, MaterialLib *aLib)
{
	TiXmlElement *pElem;
	pElem = hRoot.FirstChild("states").FirstChild("state").Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
		const char* pStateName = pElem->Attribute ("name");
		
		if (0 == pStateName) {
			NAU_THROW("State has no name in Mat Lib %s", aLib->getName().c_str());
		}

		SLOG("State: %s", pStateName);

		sprintf(s_pFullName, "%s::%s", aLib->getName().c_str(), pStateName);
		if (RESOURCEMANAGER->hasState(s_pFullName))
			NAU_THROW("State %s is already defined, in Mat Lib %s", pStateName,aLib->getName().c_str());


		IState *s = IState::create();
		s->setName(s_pFullName);

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
			NAU_THROW("Shader has no name in Mat Lib %s", aLib->getName().c_str());
		}

		SLOG("Shader : %s", pProgramName);

		sprintf(s_pFullName, "%s::%s", aLib->getName().c_str(), pProgramName); 
		if (RESOURCEMANAGER->hasProgram(s_pFullName))
			NAU_THROW("Shader %s is already defined, in Mat Lib %s", pProgramName,aLib->getName().c_str());


		const char *pVSFile = pElem->Attribute ("vs");
		const char *pPSFile = pElem->Attribute ("ps");
		const char *pGSFile = pElem->Attribute("gs");
		const char *pTCFile = pElem->Attribute("tc");
		const char *pTEFile = pElem->Attribute("te");
		const char *pCSFile = pElem->Attribute("cs");

		if ((0 == pCSFile) && (0 == pVSFile || (0 == pPSFile && 0 == pGSFile))) {
			NAU_THROW("Shader %s missing files in Mat Lib %s", pProgramName, aLib->getName().c_str());
		}

		if (0 != pCSFile && (0 != pVSFile || 0 != pPSFile || 0 != pGSFile || 0 != pTEFile || 0 != pTCFile)) 
			NAU_THROW("Mat Lib %s: Shader %s: Mixing Compute Shader with other shader stages",aLib->getName().c_str(), pProgramName);

		if (pCSFile) {
#if (NAU_OPENGL_VERSION >= 430)
			std::string CSFileName(FileUtil::GetFullPath(path, pCSFile));
			if (!FileUtil::exists(CSFileName))
				NAU_THROW("Shader file %s in MatLib %s does not exist", pCSFile, aLib->getName().c_str());
			SLOG("Program %s", pProgramName);
			IProgram *aShader = RESOURCEMANAGER->getProgram (s_pFullName);
			aShader->loadShader(IProgram::COMPUTE_SHADER, FileUtil::GetFullPath(path,pCSFile));
			aShader->linkProgram();
			SLOG("Linker: %s", aShader->getProgramInfoLog());
#else
			NAU_THROW("Mat Lib %s: Shader %s: Compute shader is not allowed with OpenGL < 4.3",aLib->getName().c_str(), pProgramName);
#endif
		}
		else {
			std::string GSFileName;
			std::string FSFileName;
			std::string TEFileName, TCFileName;

			std::string VSFileName(FileUtil::GetFullPath(path,pVSFile));
			if (!FileUtil::exists(VSFileName))
				NAU_THROW("Shader file %s in MatLib %s does not exist", pVSFile, aLib->getName().c_str());

			if (pGSFile) {
#if NAU_OPENGL_VERSION >= 320
				GSFileName = FileUtil::GetFullPath(path,pGSFile);
				if (!FileUtil::exists(GSFileName))
					NAU_THROW("Shader file %s in MatLib %s does not exist", pGSFile, aLib->getName().c_str());
#else
				NAU_THROW("Mat Lib %s: Shader %s: Geometry Shader shader is not allowed with OpenGL < 3.2",aLib->getName().c_str(), pProgramName);
#endif
			}
			if (pPSFile) {
				FSFileName = FileUtil::GetFullPath(path,pPSFile);
				if (!FileUtil::exists(FSFileName))
					NAU_THROW("Shader file %s in MatLib %s does not exist", FSFileName.c_str(), aLib->getName().c_str());
			}
			if (pTEFile) {
#if NAU_OPENGL_VERSION >= 400
				TEFileName = FileUtil::GetFullPath(path,pTEFile);
				if (!FileUtil::exists(TEFileName))
					NAU_THROW("Shader file %s in MatLib %s does not exist", TEFileName.c_str(), aLib->getName().c_str());
#else
				NAU_THROW("Mat Lib %s: Shader %s: Tesselation shaders are not allowed with OpenGL < 4.0",aLib->getName().c_str(), pProgramName);
#endif
			}
			if (pTCFile) {
#if NAU_OPENGL_VERSION >= 400
				TCFileName = FileUtil::GetFullPath(path,pTCFile);
				if (!FileUtil::exists(TCFileName))
					NAU_THROW("Shader file %s in MatLib %s does not exist", TCFileName.c_str(), aLib->getName().c_str());
#else
				NAU_THROW("Mat Lib %s: Shader %s: Tesselation shaders are not allowed with OpenGL < 4.0",aLib->getName().c_str(), pProgramName);
#endif
			}

	
			SLOG("Program %s", pProgramName);
			IProgram *aShader = RESOURCEMANAGER->getProgram (s_pFullName);
			aShader->loadShader(IProgram::VERTEX_SHADER, FileUtil::GetFullPath(path,pVSFile));
			SLOG("Shader file %s - %s",pVSFile, aShader->getShaderInfoLog(IProgram::VERTEX_SHADER).c_str());
			if (pPSFile) {
				aShader->loadShader(IProgram::FRAGMENT_SHADER, FileUtil::GetFullPath(path,pPSFile));
				SLOG("Shader file %s - %s",pPSFile, aShader->getShaderInfoLog(IProgram::FRAGMENT_SHADER).c_str());
			}
#if NAU_OPENGL_VERSION >= 400			
			if (pTCFile) {
				aShader->loadShader(IProgram::TESS_CONTROL_SHADER, FileUtil::GetFullPath(path,pTCFile));
				SLOG("Shader file %s - %s",pTCFile, aShader->getShaderInfoLog(IProgram::TESS_CONTROL_SHADER).c_str());
			}
			if (pTEFile) {
				aShader->loadShader(IProgram::TESS_EVALUATION_SHADER, FileUtil::GetFullPath(path,pTEFile));
				SLOG("Shader file %s - %s",pTEFile, aShader->getShaderInfoLog(IProgram::TESS_EVALUATION_SHADER).c_str());
			}
#endif
#if NAU_OPENGL_VERSION >= 320
			if (pGSFile) {
				aShader->loadShader(IProgram::GEOMETRY_SHADER, FileUtil::GetFullPath(path,pGSFile));
				SLOG("Shader file %s - %s",pGSFile, aShader->getShaderInfoLog(IProgram::GEOMETRY_SHADER).c_str());
			}
#endif
			aShader->linkProgram();
			SLOG("Linker: %s", aShader->getProgramInfoLog());
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
ProjectLoader::loadMaterialColor(TiXmlHandle handle, MaterialLib *aLib, Material *aMat)
{
	TiXmlElement *pElemAux;
	pElemAux = handle.FirstChild ("color").Element();
	if (!pElemAux)
		return;

	std::map<std::string, Attribute> attribs = ColorMaterial::Attribs.getAttributes();
	TiXmlElement *p = pElemAux->FirstChildElement();
	Attribute a;
	void *value;
	while (p) {
		// trying to define an attribute that does not exist		
		if (attribs.count(p->Value()) == 0)
			NAU_THROW("Library %s: Material %s: Color - %s is not an attribute", aLib->getName().c_str(),  aMat->getName().c_str(), p->Value());
		// trying to set the value of a read only attribute
		a = attribs[p->Value()];
		if (a.mReadOnlyFlag)
			NAU_THROW("Library %s: Material %s: Color - %s is a read-only attribute", aLib->getName().c_str(),  aMat->getName().c_str(), p->Value());

		value = readAttr("", p, a.mType, ColorMaterial::Attribs);
		aMat->getColor().setProp(a.mId, a.mType, value);
		p = p->NextSiblingElement();
	}



	//if (0 != pElemAux) {

	//	float r, g, b, a;

	//	pElemAux2 = pElemAux->FirstChildElement ("ambient");
	//	if (pElemAux2) {
	//		if (TIXML_SUCCESS != pElemAux2->QueryFloatAttribute ("r", &r) || 
	//			TIXML_SUCCESS != pElemAux2->QueryFloatAttribute ("g", &g) ||
	//			TIXML_SUCCESS != pElemAux2->QueryFloatAttribute ("b", &b) ||
	//			TIXML_SUCCESS != pElemAux2->QueryFloatAttribute ("a", &a)){

	//			NAU_THROW("Color ambient definition error in library %s in material %s", aLib->getName().c_str(), aMat->getName().c_str());
	//		}
	//		aMat->getColor().setProp(ColorMaterial::AMBIENT, r, g, b, a);
	//	}

	//	pElemAux2 = pElemAux->FirstChildElement ("diffuse");
	//	if (pElemAux2) {
	//		if (TIXML_SUCCESS != pElemAux2->QueryFloatAttribute ("r", &r) || 
	//			TIXML_SUCCESS != pElemAux2->QueryFloatAttribute ("g", &g) ||
	//			TIXML_SUCCESS != pElemAux2->QueryFloatAttribute ("b", &b) ||
	//			TIXML_SUCCESS != pElemAux2->QueryFloatAttribute ("a", &a)){

	//			NAU_THROW("Color diffuse definition error in library %s in material %s", aLib->getName().c_str(), aMat->getName().c_str());
	//		}
	//		aMat->getColor().setProp(ColorMaterial::DIFFUSE, r, g, b, a);
	//	}

	//	pElemAux2 = pElemAux->FirstChildElement ("specular");
	//	if (pElemAux2) {
	//		if (TIXML_SUCCESS != pElemAux2->QueryFloatAttribute ("r", &r) || 
	//			TIXML_SUCCESS != pElemAux2->QueryFloatAttribute ("g", &g) ||
	//			TIXML_SUCCESS != pElemAux2->QueryFloatAttribute ("b", &b) ||
	//			TIXML_SUCCESS != pElemAux2->QueryFloatAttribute ("a", &a)){

	//			NAU_THROW("Color specular definition error in library %s in material %s", aLib->getName().c_str(), aMat->getName().c_str());
	//		}
	//		aMat->getColor().setProp(ColorMaterial::SPECULAR, r, g, b, a);
	//	}
	//
	//	pElemAux2 = pElemAux->FirstChildElement ("emission");
	//	if (pElemAux2) {
	//		if (TIXML_SUCCESS != pElemAux2->QueryFloatAttribute ("r", &r) || 
	//			TIXML_SUCCESS != pElemAux2->QueryFloatAttribute ("g", &g) ||
	//			TIXML_SUCCESS != pElemAux2->QueryFloatAttribute ("b", &b) ||
	//			TIXML_SUCCESS != pElemAux2->QueryFloatAttribute ("a", &a)){

	//			NAU_THROW("Color emission definition error in library %s in material %s", aLib->getName().c_str(), aMat->getName().c_str());
	//		}
	//		aMat->getColor().setProp(ColorMaterial::EMISSION, r, g, b, a);
	//	}

	//	float value;
	//	pElemAux2 = pElemAux->FirstChildElement ("shininess");
	//	if (pElemAux2) {
	//		if (TIXML_SUCCESS != pElemAux2->QueryFloatAttribute ("value", &value)){

	//			NAU_THROW("Color shininess definition error in library %s in material %s", aLib->getName().c_str(), aMat->getName().c_str());
	//		}
	//		aMat->getColor().setProp(ColorMaterial::SHININESS, value);
	//	}
	//}
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
ProjectLoader::loadMaterialImageTextures(TiXmlHandle handle, MaterialLib *aLib, Material *aMat)
{
	TiXmlElement *pElemAux;
	pElemAux = handle.FirstChild ("imageTextures").FirstChild ("imageTexture").Element();
	for ( ; 0 != pElemAux; pElemAux = pElemAux->NextSiblingElement()) {
#if NAU_OPENGL_VERSION >= 420
		//const char *pTextureName = pElemAux->GetText();

		int unit;
		if (TIXML_SUCCESS != pElemAux->QueryIntAttribute ("UNIT", &unit)) {
			NAU_THROW("Library %s: Material %s: Image Texture has no unit", aLib->getName().c_str(),  aMat->getName().c_str());
		}

		const char *pTextureName = pElemAux->Attribute ("texture");
		if (0 == pTextureName) {
			NAU_THROW("Library %s: Material %s: Texture has no name in image texture", aLib->getName().c_str(),  aMat->getName().c_str());
		}
		if (!strchr(pTextureName, ':') )
			sprintf(s_pFullName, "%s::%s", aLib->getName().c_str(), pTextureName);
		else
			sprintf(s_pFullName, "%s", pTextureName);
		if (!RESOURCEMANAGER->hasTexture(s_pFullName))
			NAU_THROW("Library %s: Material %s: Texture %s in image texture is not defined", aLib->getName().c_str(),  aMat->getName().c_str(), pTextureName);
		
		Texture *t = RESOURCEMANAGER->getTexture(s_pFullName);
		int texID = t->getPropi(Texture::ID);

		aMat->attachImageTexture(t->getLabel(), unit, texID);
		// Reading Image Texture Attributes

		std::map<std::string, Attribute> attribs = ImageTexture::Attribs.getAttributes();
		TiXmlElement *p = pElemAux->FirstChildElement();
		Attribute a;
		void *value;
		while (p) {
			// trying to define an attribute that does not exist		
			if (attribs.count(p->Value()) == 0)
				NAU_THROW("Library %s: Material %s: ImageTexture - %s is not an attribute", aLib->getName().c_str(),  aMat->getName().c_str(), p->Value());
			// trying to set the value of a read only attribute
			a = attribs[p->Value()];
			if (a.mReadOnlyFlag)
				NAU_THROW("Library %s: Material %s: ImageTexture - %s is a read-only attribute", aLib->getName().c_str(),  aMat->getName().c_str(), p->Value());

			value = readAttr("", p, a.mType, ImageTexture::Attribs);
			aMat->getImageTexture(unit)->setProp(a.mId, a.mType, value);
			p = p->NextSiblingElement();
		}
#else
		NAU_THROW("Library %s: Material %s: Buffers Not Supported with OpenGL Version %d", aLib->getName().c_str(), aMat->getName().c_str(), NAU_OPENGL_VERSION);
#endif

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
ProjectLoader::loadMaterialBuffers(TiXmlHandle handle, MaterialLib *aLib, Material *aMat)
{

	TiXmlElement *pElemAux;
	pElemAux = handle.FirstChild("buffers").FirstChild("buffer").Element();
	for (; 0 != pElemAux; pElemAux = pElemAux->NextSiblingElement()) {
#if NAU_OPENGL_VERSION >=  430
		const char *pName = pElemAux->Attribute("name");
		if (0 == pName) {
			NAU_THROW("Library %s: Material %s: Buffer has no name", aLib->getName().c_str(), aMat->getName().c_str());
		}
		if (!strchr(pName, ':'))
			sprintf(s_pFullName, "%s::%s", aLib->getName().c_str(), pName);
		else
			sprintf(s_pFullName, "%s", pName);
		if (!RESOURCEMANAGER->hasBuffer(s_pFullName))
			NAU_THROW("Library %s: Material %s: Buffer %s is not defined", aLib->getName().c_str(), aMat->getName().c_str(), pName);

		IBuffer *buffer = RESOURCEMANAGER->getBuffer(s_pFullName);
		IBuffer *b = buffer->clone();
		// Reading Buffer Attributes

		std::map<std::string, Attribute> attribs = IBuffer::Attribs.getAttributes();
		TiXmlElement *p = pElemAux->FirstChildElement();
		Attribute a;
		void *value;
		while (p) {
			// trying to define an attribute that does not exist		
			if (attribs.count(p->Value()) == 0)
				NAU_THROW("Library %s: Material %s: Buffer - %s is not an attribute", aLib->getName().c_str(), aMat->getName().c_str(), p->Value());
			// trying to set the value of a read only attribute
			a = attribs[p->Value()];
			if (a.mReadOnlyFlag)
				NAU_THROW("Library %s: Material %s: Buffer - %s is a read-only attribute", aLib->getName().c_str(), aMat->getName().c_str(), p->Value());

			value = readAttr("", p, a.mType, IBuffer::Attribs);
			b->setProp(a.mId, a.mType, value);
			p = p->NextSiblingElement();
		}
		aMat->attachBuffer(b);
#else
		NAU_THROW("Library %s: Material %s: Buffers Not Supported with OpenGL Version %d", aLib->getName().c_str(), aMat->getName().c_str(), NAU_OPENGL_VERSION);

#endif	
	
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
ProjectLoader::loadMaterialTextures(TiXmlHandle handle, MaterialLib *aLib, Material *aMat)
{
	TiXmlElement *pElemAux;
	pElemAux = handle.FirstChild ("textures").FirstChild ("texture").Element();
	for ( ; 0 != pElemAux; pElemAux = pElemAux->NextSiblingElement()) {
		//const char *pTextureName = pElemAux->GetText();

		const char *pTextureName = pElemAux->Attribute ("name");
		if (0 == pTextureName) {
			NAU_THROW("Library %s: Material %s: Texture has no name", aLib->getName().c_str(),  aMat->getName().c_str());
		}
		
		int unit;
		if (TIXML_SUCCESS != pElemAux->QueryIntAttribute ("UNIT", &unit)) {
			NAU_THROW("Library %s: Material %s: Texture has no unit", aLib->getName().c_str(),  aMat->getName().c_str());
		}

		if (!strchr(pTextureName, ':') )
			sprintf(s_pFullName, "%s::%s", aLib->getName().c_str(), pTextureName);
		else
			sprintf(s_pFullName, "%s", pTextureName);
		if (!RESOURCEMANAGER->hasTexture(s_pFullName))
			NAU_THROW("Library %s: Material %s: Texture %s is not defined", aLib->getName().c_str(),  aMat->getName().c_str(), pTextureName);
			

		aMat->attachTexture (unit, s_pFullName);

		// Reading Texture Sampler Attributes

		std::map<std::string, Attribute> attribs = TextureSampler::Attribs.getAttributes();
		TiXmlElement *p = pElemAux->FirstChildElement();
		Attribute a;
		void *value;
		while (p) {
			// trying to define an attribute that does not exist		
			if (attribs.count(p->Value()) == 0)
				NAU_THROW("Library %s: Material %s: Texture %s: %s is not an attribute", aLib->getName().c_str(),  aMat->getName().c_str(), pTextureName, p->Value());
			// trying to set the value of a read only attribute
			a = attribs[p->Value()];
			if (a.mReadOnlyFlag)
				NAU_THROW("Library %s: Material %s: Texture %s: %s is a read-only attribute, in file %s", aLib->getName().c_str(),  aMat->getName().c_str(), pTextureName, p->Value());

			value = readAttr(pTextureName, p, a.mType, TextureSampler::Attribs);
			aMat->getTextureSampler(unit)->setProp(a.mId, a.mType, value);
			p = p->NextSiblingElement();
		}
	}
}


/* -----------------------------------------------------------------------------
MATERIALSHADER

	<shader>
		<name>perpixel-color-shadow</name>
		<values>
			<valueof uniform="lightPosition" type="LIGHT" context="Sun" component="POSITION" /> 
		</values>
	</shader>
	
-----------------------------------------------------------------------------*/
void 
ProjectLoader::loadMaterialShader(TiXmlHandle handle, MaterialLib *aLib, Material *aMat)
{
	TiXmlElement *pElemAux, *pElemAux2;

	pElemAux = handle.FirstChild ("shader").Element();
	if (0 != pElemAux) {
		TiXmlHandle hShader (pElemAux);

		pElemAux2 = hShader.FirstChild ("name").Element();
		const char *pShaderName = pElemAux2->GetText();

		if (0 == pShaderName) {
			NAU_THROW("Shader has no target in library %s in material %s", aLib->getName().c_str(), aMat->getName().c_str());
		}
		sprintf(s_pFullName, "%s::%s",aLib->getName().c_str(),pShaderName);
		if (!RESOURCEMANAGER->hasProgram(s_pFullName))
			NAU_THROW("Shader %s is not defined in library %s in material %s", pShaderName,aLib->getName().c_str(), aMat->getName().c_str());

		
		aMat->attachProgram (s_pFullName);
		aMat->clearProgramValues();

		pElemAux2 = hShader.FirstChild ("values").FirstChild ("valueof").Element();
		for ( ; 0 != pElemAux2; pElemAux2 = pElemAux2->NextSiblingElement()) {
			const char *pUniformName = pElemAux2->Attribute ("uniform");
			const char *pComponent = pElemAux2->Attribute ("component");
			const char *pContext = pElemAux2->Attribute("context");
			const char *pType = pElemAux2->Attribute("type");
			//const char *pId = pElemAux2->Attribute("id");

			if (0 == pUniformName) {
				NAU_THROW("No uniform name, in library %s in material %s", aLib->getName().c_str(), aMat->getName().c_str());
			}
			if (0 == pType) {
				NAU_THROW("No type found for uniform %s, in library %s in material %s", pUniformName, aLib->getName().c_str(), aMat->getName().c_str());
			}
			if (0 == pContext) {
				NAU_THROW("No context found for uniform %s, in library %s in material %s", pUniformName, aLib->getName().c_str(), aMat->getName().c_str());
			}
			if (0 == pComponent) {
				NAU_THROW("No component found for uniform %s, in library %s in material %s", pUniformName, aLib->getName().c_str(), aMat->getName().c_str());
			}
			if (!ProgramValue::Validate(pType, pContext, pComponent))
				NAU_THROW("Uniform %s is not valid, in library %s in material %s", pUniformName, aLib->getName().c_str(), aMat->getName().c_str());

			int id = 0;
			if (((strcmp(pContext,"LIGHT") == 0) || (0 == strcmp(pContext,"TEXTURE")) || (0 == strcmp(pContext,"IMAGE_TEXTURE"))) &&  (0 != strcmp(pComponent,"COUNT"))) {
				if (TIXML_SUCCESS != pElemAux2->QueryIntAttribute ("id", &id))
					NAU_THROW("No id found for uniform %s, in library %s in material %s", pUniformName, aLib->getName().c_str(), aMat->getName().c_str());
				if (id < 0)
					NAU_THROW("id must be non negative, in uniform %s, in library %s in material %s", pUniformName, aLib->getName().c_str(), aMat->getName().c_str());
			}
			std::string s(pType);
			if (s == "TEXTURE") {
				sprintf(s_pFullName, "%s::%s", aLib->getName().c_str(),pContext);
				if (!RESOURCEMANAGER->hasTexture(s_pFullName)) {
					NAU_THROW("Texture %s is not defined, in library %s in material %s", s_pFullName, aLib->getName().c_str(), aMat->getName().c_str());
				}
				else
					aMat->addProgramValue (pUniformName, ProgramValue (pUniformName,pType, s_pFullName, pComponent, id));
			}

			else if (s == "CAMERA") {
				// Must consider that a camera can be defined internally in a pass, example:lightcams
				/*if (!RENDERMANAGER->hasCamera(pContext))
					NAU_THROW("Camera %s is not defined in the project file", pContext);*/
				aMat->addProgramValue (pUniformName, ProgramValue (pUniformName,pType, pContext, pComponent, id));
			}
			else if (s == "LIGHT") {
				if (!RENDERMANAGER->hasLight(pContext))
					NAU_THROW("Light %s is not defined in the project file", pContext);
				aMat->addProgramValue (pUniformName, ProgramValue (pUniformName,pType, pContext, pComponent, id));

			}
			else
				aMat->addProgramValue (pUniformName, ProgramValue (pUniformName,pType, pContext, pComponent, id));

				//sprintf(s_pFullName, "%s(%s,%s)",pType,pContext,pComponent );

			
		}
	}
	aMat->checkProgramValuesAndUniforms();
}

/* -----------------------------------------------------------------------------
MATERIALSTATE

	<state>
		<alphatest alphaFunc="GREATER" alphaRef="0.25" />
		<blend src="ONE" dst="ZERO" />
		<cull value="0" />
		<order value="1" />
	</state>	

alphaFunc: ALPHA_NEVER, ALPHA_ALWAYS, ALPHA_LESS, ALPHA_LEQUAL,
				ALPHA_EQUAL, ALPHA_GEQUAL, ALPHA_GREATER, ALPHA_NOT_EQUAL

order: is a value that defines the order of rendering. Higher values are drawn later.

src and dst: ZERO,ONE,SRC_COLOR,ONE_MINUS_SRC_COLOR,DST_COLOR,
				ONE_MINUS_DST_COLOR, SRC_ALPHA, ONE_MINUS_SRC_ALPHA, DST_ALPHA,
				ONE_MINUS_DST_ALPHA, SRC_ALPHA_SATURATE, CONSTANT_COLOR, 
				ONE_MINUS_CONSTANT_COLOR, CONSTANT_ALPHA, ONE_MINUS_CONSTANT_ALPHA

	OR

	<state name="bla" />

where bla is previously defined in the mat lib.
	

-----------------------------------------------------------------------------*/

void 
ProjectLoader::loadMaterialState(TiXmlHandle handle, MaterialLib *aLib, Material *aMat)
{
	TiXmlElement *pElemAux;

	pElemAux = handle.FirstChild("state").Element();
	if (0 != pElemAux) {
		const char *pStateName = pElemAux->Attribute ("name");
		//definition by ref
		if (0 != pStateName) {

			sprintf(s_pFullName, "%s::%s", aLib->getName().c_str(), pStateName);
			if (!RESOURCEMANAGER->hasState(s_pFullName))
				NAU_THROW("State %s is not defined in library %s, in material %s", pStateName,aLib->getName().c_str(), aMat->getName().c_str());


			aMat->setState(RESOURCEMANAGER->getState(s_pFullName));
		}
		else { // definition inline
			loadState(pElemAux,aLib,aMat,aMat->getState());
		}
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
	std::string path = FileUtil::GetPath(file);
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
			NAU_THROW("Parse problem in mat lib file %s", file.c_str());
		hRoot = TiXmlHandle (pElem);
	}

	pElem = hRoot.Element();
	const char* pName = pElem->Attribute ("name");

	if (0 == pName) 
		NAU_THROW("Material lib has no name in file %s", file.c_str());

	SLOG("Material Lib Name: %s", pName);

	aLib = MATERIALLIBMANAGER->getLib (pName);

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
			NAU_THROW("Material has no name in material lib %s", pName);

		if (MATERIALLIBMANAGER->hasMaterial(pName, pMaterialName))
			NAU_THROW("Material %s is already defined in library %s", pMaterialName,pName);

		SLOG("Material: %s", pMaterialName);

		//Material *mat = new Material;
		//mat->setName (pMaterialName);
		Material *mat = MATERIALLIBMANAGER->createMaterial(pName,pMaterialName);

		loadMaterialColor(handle,aLib,mat);
		loadMaterialTextures(handle,aLib,mat);
		loadMaterialShader(handle,aLib,mat);
		loadMaterialState(handle,aLib,mat);
		loadMaterialImageTextures(handle, aLib, mat);
		loadMaterialBuffers(handle, aLib, mat);

		//aLib->addMaterial (mat);
	}


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
void
ProjectLoader::loadDebug (TiXmlHandle &hRoot)
{
#ifdef GLINTERCEPTDEBUG
	
	TiXmlElement *pElem;
	TiXmlHandle handle (hRoot.FirstChild ("debug").Element());
	bool startGliLog = true;

	if (handle.Element()){
		pElem = handle.Element();

		initGLInterceptFunctions();
	
		if (pElem->Attribute("glilog")){
			pElem->QueryBoolAttribute("glilog",&startGliLog);
		}
		if (startGliLog){
			startGlilog();
		}

		//TiXmlElement *pElem;
		loadDebugFunctionlog(handle);
		loadDebugLogperframe(handle);
		loadDebugErrorchecking(handle);
		loadDebugImagelog(handle);
		loadDebugShaderlog(handle);
		loadDebugDisplaylistlog(handle);
		loadDebugFramelog(handle);
		loadDebugTimerlog(handle);
		loadDebugPlugins(handle);
		startGLIConfiguration();
	}
#endif
}

/* ----------------------------------------------------------------
Specification of the functionlog:

		<functionlog>
			<enabled value="bool"/>
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
void
ProjectLoader::loadDebugFunctionlog (TiXmlHandle &hRoot){
	TiXmlHandle handle (hRoot.FirstChild ("functionlog").Element());

	loadDebugConfigData(handle,"functionlog");

	//loadDebugFunctionlogXmlFormat(hRoot);
}



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
void
ProjectLoader::loadDebugFunctionlogXmlFormat (TiXmlHandle &hRoot){
	TiXmlHandle handle (hRoot.FirstChild ("xmlformat").Element());

	loadDebugConfigData(handle,"functionlogxmlformat");
}

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
void
ProjectLoader::loadDebugLogperframe (TiXmlHandle &hRoot){
	TiXmlHandle handle (hRoot.FirstChild ("logperframe").Element());

	loadDebugConfigData(handle,"logperframe");
}

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
void
ProjectLoader::loadDebugErrorchecking (TiXmlHandle &hRoot){
	TiXmlHandle handle (hRoot.FirstChild ("errorchecking").Element());

	loadDebugConfigData(handle,"errorchecking");

}

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
void
ProjectLoader::loadDebugImagelog (TiXmlHandle &hRoot){
	TiXmlHandle handle (hRoot.FirstChild ("imagelog").Element());

	loadDebugConfigData(handle,"imagelog");


	loadDebugImagelogimageicon(handle);
}

/* ----------------------------------------------------------------
Specification of the imageicon:

imageiconformat is tee format of the save icon images (TGA,PNG or JPG)
only one format at a time

			<imageicon>
				<imagesaveicon value="bool"/>
				<imageiconsize value="uint"/>
				<imageiconformat value="png"/>
			</imageicon>
----------------------------------------------------------------- */
void
ProjectLoader::loadDebugImagelogimageicon (TiXmlHandle &hRoot){
	TiXmlHandle handle (hRoot.FirstChild ("imageicon").Element());

	loadDebugConfigData(handle,"imagelogimageicon");
}

/* ----------------------------------------------------------------
Specification of the shaderlog:

		<shaderlog>
			<enabled value="bool"/>
			<shaderrendercallstatelog value="bool"/>
			<shaderattachlogstate value="bool"/>
			<shadervalidateprerender value="bool"/>
			<shaderloguniformsprerender value="bool"/>
		</shaderlog>
----------------------------------------------------------------- */
void
ProjectLoader::loadDebugShaderlog (TiXmlHandle &hRoot){
	TiXmlHandle handle (hRoot.FirstChild ("shaderlog").Element());

	loadDebugConfigData(handle,"shaderlog");
}

/* ----------------------------------------------------------------
Specification of the displaylistlog:

		<displaylistlog>
			<enabled value="bool"/>
		</displaylistlog>
----------------------------------------------------------------- */
void
ProjectLoader::loadDebugDisplaylistlog (TiXmlHandle &hRoot){
	TiXmlHandle handle (hRoot.FirstChild ("displaylistlog").Element());

	loadDebugConfigData(handle,"displaylistlog");
}

/* ----------------------------------------------------------------
Specification of the framelog:

According to the original gliConfig regarding framestencilcolors:
When saving the stencil buffer, it can be useful to save the buffer with color codes.
(ie stencil value 1 = red) This array supplies index color pairs for each stencil 
value up to 255. The indices must be in order and the colors are in the format 
AABBGGRR. If an index is missing, it will take the value of the index as the color.
(ie. stencil index 128 = (255, 128,128,128) = greyscale values)

		<framelog>
			<enabled value="bool"/>
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
void
ProjectLoader::loadDebugFramelog (TiXmlHandle &hRoot){
	TiXmlHandle handle (hRoot.FirstChild ("framelog").Element());

	loadDebugConfigData(handle,"framelog");

	
	loadDebugFramelogFrameicon(handle);
	loadDebugFramelogFramemovie(handle);
}

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
void
ProjectLoader::loadDebugFramelogFrameicon (TiXmlHandle &hRoot){
	TiXmlHandle handle (hRoot.FirstChild ("frameicon").Element());

	loadDebugConfigData(handle,"framelogframeicon");

}


/* ----------------------------------------------------------------
Specification of the framemovie:

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
----------------------------------------------------------------- */
void
ProjectLoader::loadDebugFramelogFramemovie (TiXmlHandle &hRoot){
	TiXmlHandle handle (hRoot.FirstChild ("framemovie").Element());

	loadDebugConfigData(handle,"framelogframemovie");

}

/* ----------------------------------------------------------------
Specification of the timerlog:

		<timerlog>
			<enabled value="bool"/>
			<timerlogcutoff value="uint"/>
		</timerlog>
----------------------------------------------------------------- */
void
ProjectLoader::loadDebugTimerlog (TiXmlHandle &hRoot){
	TiXmlHandle handle (hRoot.FirstChild ("timerlog").Element());

	loadDebugConfigData(handle,"timerlog");

}

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
void
ProjectLoader::loadDebugPlugins (TiXmlHandle &hRoot)
{
#ifdef GLINTERCEPTDEBUG
	TiXmlElement *pElem;
	TiXmlHandle handle (hRoot.FirstChild("plugins").Element());
	
	string name;
	string dllpath;
	string data="";

	pElem = handle.FirstChild().Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
			pElem->QueryStringAttribute("name",&name);
			pElem->QueryStringAttribute("dll",&dllpath);
			if(pElem->GetText()){
				data=pElem->GetText();
			}
			addPlugin(name.c_str(), dllpath.c_str(), data.c_str());
		
	}
#endif
}

/* ----------------------------------------------------------------
Helper function, reads sub attributes.
				<configcategory>
					<configattribute value="...">
					...
				</configcategory>
----------------------------------------------------------------- */
void
ProjectLoader::loadDebugConfigData (TiXmlHandle &handle, const char *configMapName)
{
#ifdef GLINTERCEPTDEBUG

	TiXmlElement *pElem;
	void *functionSetPointer;
	pElem = handle.FirstChild().Element();
	for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
		const char *functionName = pElem->Value();
		functionSetPointer = getGLIFunction(functionName,configMapName);
		switch (getGLIFunctionType(functionSetPointer))
		{
		case GLIEnums::FunctionType::BOOL:{
				bool functionValue; 
				pElem->QueryBoolAttribute("value",&functionValue);
				useGLIFunction(functionSetPointer,&functionValue);
				break;
			}
		case GLIEnums::FunctionType::INT:{
				int functionValue; 
				pElem->QueryIntAttribute("value",&functionValue);
				useGLIFunction(functionSetPointer,&functionValue);
				break;
			}
		case GLIEnums::FunctionType::UINT:{
				unsigned int functionValue; 
				pElem->QueryUnsignedAttribute("value",&functionValue);
				useGLIFunction(functionSetPointer,&functionValue);
				break;
			}
		case GLIEnums::FunctionType::STRING:{
				string functionValue; 
				pElem->QueryStringAttribute("value",&functionValue);
				useGLIFunction(functionSetPointer,(void*)functionValue.c_str());
				break;
			}
		case GLIEnums::FunctionType::UINTARRAY:
		case GLIEnums::FunctionType::STRINGARRAY:
			useGLIClearFunction(functionSetPointer);
			loadDebugArrayData(handle,functionName,functionSetPointer);
			break;
		default:
			break;
		}
	}
#endif
}

/* ----------------------------------------------------------------
Specification of the configdata:
				<arrayconfigname>
					<item value="...">
					...
				</arrayconfigname>
----------------------------------------------------------------- */
void
ProjectLoader::loadDebugArrayData (TiXmlHandle &hRoot, const char *functionName, void *functionSetPointer)
{
#ifdef GLINTERCEPTDEBUG
	TiXmlElement *pElem;
	TiXmlHandle handle (hRoot.FirstChild(functionName).Element());
	unsigned int functionType = getGLIFunctionType(functionSetPointer);

	pElem = handle.FirstChild().Element();
	switch (functionType)
	{
	case GLIEnums::FunctionType::UINTARRAY:{
		for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
				unsigned int functionValue; 
				pElem->QueryUnsignedAttribute("value",&functionValue);
				useGLIFunction(functionSetPointer,&functionValue);
				break;
			}
		}
	case GLIEnums::FunctionType::STRINGARRAY:{
		for ( ; 0 != pElem; pElem = pElem->NextSiblingElement()) {
				string functionValue; 
				pElem->QueryStringAttribute("value",&functionValue);
				useGLIFunction(functionSetPointer,(void*)functionValue.c_str());
				break;
			}
		}
	default:
		break;
	}
#endif
}

#ifndef PROJECTLOADER_H
#define PROJECTLOADER_H

#include "nau/material/materialLib.h"
#include "nau/render/pass.h"
#include "nau/system/file.h"

#include <tinyxml.h>

#include <string>
#include <vector>


using namespace nau::render;
using namespace nau::material;

namespace nau
{
	namespace loader
	{
		class ProjectLoader
		{
		public:
			// loads a project
			static void load (std::string file, int *width, int *height);
			// loads a material lib
			static void loadMatLib (std::string file);

			// loads a physics material lib
			static void loadPhysLib(std::string file);

			static std::string s_Path;
			static std::string s_File;

		private:

			typedef struct {
				std::string filename;
				int row, column;
				std::string actualValue;
				std::string objType;
			} DeferredValidation;

			static std::vector<DeferredValidation> s_DeferredVal;

			enum {
				OK,
				ITEM_NAME_NOT_SPECIFIED,
				FILE_DOES_NOT_EXIST
			};

			ProjectLoader(void);

			// add to deferred Validation list
			static void addToDefferredVal(std::string filename, int row, int column,
				std::string value, std::string objType);
			static void deferredValidation();

			// load the constants section
			static void loadConstants(TiXmlHandle &hRoot);
			// load the assets section
			static void loadAssets (TiXmlHandle &hRoot, std::vector<std::string> &matLib);
			// load the pipelines section
			static void loadPipelines (TiXmlHandle &hRoots);
			// load interface elements
			static void loadInterface(TiXmlHandle &hRoots);

			// check for tags that are not allowed
			static void checkForNonValidChildTags(std::string parent, std::vector<std::string> &excluded, 
											TiXmlElement *pElem);
			static void checkForNonValidAttributes(std::string parent, std::vector<std::string> &excluded, 
											TiXmlElement *pElem);

			// read an item from a library
			static int readItemFromLib(TiXmlElement *p, std::string tag, std::string *lib, std::string *item);
			static int readItemFromLib(TiXmlElement *p, std::string tag, std::string *fullName);

			// read child tags
			static void readChildTags(std::string parent, 
											AttributeValues *anObj, 
											nau::AttribSet &attribs, 
											std::vector<std::string> &excluded, 
											TiXmlElement *pElem, 
											bool showOnlyExcluded=false);

			static void readAttributes(std::string parent, 
											AttributeValues *anObj, 
											nau::AttribSet &attribs, 
											std::vector<std::string> &excluded, 
											TiXmlElement *pElem);
			// returns a text message to include in the throw statement. 
			// if message  == "" then attribute is ok
			static void validateObjectAttribute(std::string type, std::string context, std::string component, std::string *message);
			static void validateObjectTypeAndComponent(std::string type, std::string component, std::string *message);

			// converts to lower caps
			static std::string toLower(std::string strToConvert);
			// read file and return full path relative to project
			static std::string readFile(TiXmlElement *p, std::string tag, std::string item);
			// get valid attribute values 
			static std::string &getValidValuesString(std::unique_ptr<Attribute> &a);
			// read a child tag 
			static Data *readChildTag(std::string parent, TiXmlElement *pElem, Enums::DataType type, 
				AttribSet &attribs);
			static std::string & readChildTagString(std::string parent, TiXmlElement *pElem, AttribSet &attribs);
			// read an attribute 
			//static void *readAttribute(const char *name, TiXmlElement *p, Enums::DataType type, AttribSet &attribs);
			static Data *readAttribute(std::string tag, std::unique_ptr<Attribute> &attr, TiXmlElement *p);
			static std::string &readAttributeString(std::string tag, std::unique_ptr<Attribute> &attr, TiXmlElement *p);
			// check if a tring is in a vector
			static bool isExcluded(std::string what, std::vector<std::string> &excluded);
			// get all keys in a vector
			static void getKeystoVector(std::map<std::string, std::unique_ptr<Attribute> > &, 
				std::vector<std::string> *result);
			// build excluded vector
			static void buildExcludedVector(std::map<std::string, std::unique_ptr<Attribute> > &, 
				std::set<std::string> &included, std::vector<std::string> *excluded);
			// checks if a constant has been defined
			static bool isConstantDefined(std::string s);

			// read single attributes. values can be numbers of constant labels
			static bool readFloatAttribute(TiXmlElement *p, std::string label, float *value);
			static bool readIntAttribute(TiXmlElement *p, std::string label, int *value);
			static bool readUIntAttribute(TiXmlElement *p, std::string label, unsigned int *value);

			// Asset Loading
			static void loadUserAttrs(TiXmlHandle handle);
			static void loadScenes(TiXmlHandle handle);
			static void loadGeometry(TiXmlElement *elem); 
			static void loadViewports(TiXmlHandle handle);
			static void loadCameras(TiXmlHandle handle);
			static void loadLights(TiXmlHandle handle);
			static void loadEvents(TiXmlHandle handle);
			static void loadAtomicSemantics(TiXmlHandle handle);

			//Pass Elements
			static void loadPassCamera(TiXmlHandle hPass, Pass *aPass); 	
			static void loadPassMode(TiXmlHandle hPass, Pass *aPass);
			static void loadPassScripts(TiXmlHandle hPass, Pass *aPass);
			static void loadPassPreProcess(TiXmlHandle hPass, Pass *aPass);
			static void loadPassPostProcess(TiXmlHandle hPass, Pass *aPass);
			static void loadPassLights(TiXmlHandle hPass, Pass *aPass);
			static void loadPassScenes(TiXmlHandle hPass, Pass *aPass);
//			static void loadPassClearDepthAndColor(TiXmlHandle hPass, Pass *aPass);
			static void loadPassViewport(TiXmlHandle hPass, Pass *aPass);
			static void loadPassParams(TiXmlHandle hPass, Pass *aPass);
			static void loadPassRenderTargets(TiXmlHandle hPass, Pass *aPass, 
				std::map<std::string, Pass*> passMapper);
			static void loadPassTexture(TiXmlHandle hPass, Pass *aPass);
			static void loadPassMaterial(TiXmlHandle hPass, Pass *aPass);
			static void loadPassMaterialMaps(TiXmlHandle hPass, Pass *aPass);
			static void loadPassInjectionMaps(TiXmlHandle hPass, Pass *aPass);

#ifdef NAU_OPTIX
			static void loadPassOptixSettings(TiXmlHandle hPass, Pass *aPass);
			static void loadPassOptixPrimeSettings(TiXmlHandle hPass, Pass *aPass);
#endif
			static void loadPassComputeSettings(TiXmlHandle hPass, Pass *aPass);

			//Debug Loading
			//static void loadDebug (TiXmlHandle &hRoot);
			//static void loadDebugConfigData (TiXmlHandle &handle, const char *configMapName);
			//static void loadDebugFunctionlog (TiXmlHandle &hRoot);
			//static void loadDebugFunctionlogXmlFormat (TiXmlHandle &hRoot);
			//static void loadDebugLogperframe (TiXmlHandle &hRoot);
			//static void loadDebugErrorchecking (TiXmlHandle &hRoot);
			//static void loadDebugImagelog (TiXmlHandle &hRoot);
			//static void loadDebugImagelogimageicon (TiXmlHandle &hRoot);
			//static void loadDebugShaderlog (TiXmlHandle &hRoot);
			//static void loadDebugFramelog (TiXmlHandle &hRoot);
			//static void loadDebugFramelogFrameicon (TiXmlHandle &hRoot);
			//static void loadDebugFramelogFramemovie (TiXmlHandle &hRoot);
			//static void loadDebugTimerlog (TiXmlHandle &hRoot);
			//static void loadDebugArrayData (TiXmlHandle &hRoot, const char *functionName, void *functionSetPointer);
			//static void loadDebugPlugins (TiXmlHandle &hRoot);

			// load Material Lib elements
			static void loadMatLibRenderTargets(TiXmlHandle hPass, MaterialLib *aLib, std::string path);
			static void loadMatLibTextures(TiXmlHandle hPass, MaterialLib *aLib, std::string path);
			static void loadMatLibStates(TiXmlHandle hPass, MaterialLib *aLib);
			static void loadMatLibShaders(TiXmlHandle hPass, MaterialLib *aLib, std::string path);
			static void loadMatLibBuffers(TiXmlHandle hPass, MaterialLib *aLib, std::string path);

			// load material elements
			static void loadMaterialColor(TiXmlHandle handle, MaterialLib *aLib, std::shared_ptr<Material> &aMat);
			static void loadMaterialTextures(TiXmlHandle handle, MaterialLib *aLib, std::shared_ptr<Material> &aMat);
			static void loadMaterialImageTextures(TiXmlHandle handle, MaterialLib *aLib, std::shared_ptr<Material> &aMat);
			static void loadMaterialBuffers(TiXmlHandle handle, MaterialLib *aLib, std::shared_ptr<Material> &aMat);
			static void loadMaterialShader(TiXmlHandle handle, MaterialLib *aLib, std::shared_ptr<Material> &aMat);
			static void loadMaterialState(TiXmlHandle handle, MaterialLib *aLib, std::shared_ptr<Material> &aMat);

			// load state
			static void loadState(TiXmlElement *pElemAux, MaterialLib *aLib, Material *aMat, IState *s);

			// aux pre alocated variables
			static char s_pFullName[256];
			static string s_Dummy;
			static vec4 s_Dummy_vec4;
			static vec2 s_Dummy_vec2;
			static bvec4 s_Dummy_bvec4;
			static float s_Dummy_float;
			static int s_Dummy_int;
			static unsigned int s_Dummy_uint;
			static bool s_Dummy_bool;
			static uivec3 s_Dummy_uivec3;
			static uivec2 s_Dummy_uivec2;
			static std::string s_Dummy_string;

			static std::map<std::string, float> s_Constants;
		};
	};
};

#endif //PROJECTLOADER_H



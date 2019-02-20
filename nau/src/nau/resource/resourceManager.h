#ifndef RESOURCE_MANAGER_H
#define RESOURCE_MANAGER_H

#include "nau/material/material.h"
#include "nau/material/iArrayOfTextures.h"
#include "nau/render/iRenderable.h"
#include "nau/render/iRenderTarget.h"
#include "nau/scene/sceneObject.h"

#include <vector>
#include <string>
#include <map>

#ifdef _WINDLL
#ifdef nau_EXPORTS
#define nau_API __declspec(dllexport)   
#else  
#define nau_API __declspec(dllimport)   
#endif 
#else
#define nau_API
#endif

using namespace nau::material; 

namespace nau 
{
	namespace resource
	{
		class ResourceManager
		{
		private:
			std::string m_Path;
			
			std::map<std::string, nau::render::IRenderTarget*> m_RenderTargets;
			std::map<std::string, std::shared_ptr<nau::render::IRenderable>> m_Meshes;
			std::map<std::string, nau::material::IProgram*> m_Programs;
			std::map<std::string, nau::material::IState*> m_States;
			std::vector<nau::material::ITexture*> m_Textures;
			std::map<std::string, nau::material::IBuffer*> m_Buffers;
			std::map<std::string, nau::material::IArrayOfTextures*> m_ArraysOfTextures;

			static int renderableCount;

			std::shared_ptr<nau::render::IRenderable> m_EmptyMesh;

		public:
			ResourceManager (std::string path);
			~ResourceManager (void);

			void clear();

			/***Textures***/

			nau_API ITexture* createTexture (std::string label, std::string internalFormat, int width,
				int height, int depth = 1, int layers = 1, int levels = 1, int samples = 1);
			nau_API ITexture* createTexture(std::string label, int internalFormat, int width,
				int height, int depth = 1, int layers = 1, int levels = 1, int samples = 1);
			nau_API ITexture* createCubeMapTexture(std::string label, std::string internalFormat, int width);

			/// create texture with default attributes, texture requires building prior to usage
			nau_API ITexture * createTexture(std::string label);

			/// create texture from file
			nau_API ITexture* addTexture (std::string fn, std::string label = "", bool mipmap = true);
			/// create cube map from file
			nau_API ITexture* addCubeMapTexture (std::vector<std::string> &fn, std::string &label, bool mipmap = true);
			
			nau_API void removeTexture (const std::string &name);

			nau_API bool hasTexture(const std::string &name);
			nau_API ITexture* getTexture (const std::string &name);
			nau_API ITexture* getTexture(unsigned int i);
			nau_API ITexture* getTextureByID(unsigned int id);
			nau_API int getNumTextures();

			/***Rendertargets***/
			nau_API nau::render::IRenderTarget* createRenderTarget (std::string name);
			nau_API void removeRenderTarget (std::string name);
			nau_API bool hasRenderTarget(const std::string &name);
			nau_API nau::render::IRenderTarget *getRenderTarget(const std::string &name);
			nau_API int getNumRenderTargets();
			nau_API void getRenderTargetNames(std::vector<std::string> *v);


			/***Renderables***/
			nau_API std::string makeMeshName(const std::string &name, const std::string &filename);
			nau_API std::shared_ptr<IRenderable> &createRenderable(const std::string &type, const std::string &name);
			nau_API void removeRenderable(const std::string &name);
			nau_API bool hasRenderable (const std::string &name);
			nau_API std::shared_ptr<IRenderable> &getRenderable (const std::string &name);

			/***States***/
			nau_API IState * createState(const std::string &stateName);
			nau_API bool hasState (const std::string &stateName);
			nau_API nau::material::IState* getState (const std::string &stateName);

			/***Shaders***/
			nau_API bool hasProgram (std::string programName);
			nau_API nau::material::IProgram* getProgram (std::string programName);
			nau_API unsigned int getNumPrograms();
			nau_API std::vector<std::string> *getProgramNames();

			/***Buffers***/
			nau_API nau::material::IBuffer* getBuffer(std::string name);
			nau_API nau::material::IBuffer* createBuffer(std::string name);
			nau_API bool hasBuffer(std::string name);
			nau_API nau::material::IBuffer* getBufferByID(int id);
			nau_API void getBufferNames(std::vector<std::string> *names);
			nau_API void clearBuffers();

			/***Arrays of Textures***/
			nau_API nau::material::IArrayOfTextures* getArrayOfTextures(std::string name);
			nau_API nau::material::IArrayOfTextures* createArrayOfTextures(std::string name);
			nau_API bool hasArrayOfTextures(std::string name);
			nau_API void getArrayOfTexturesNames(std::vector<std::string> *names);
		};
	};
};
#endif //RESOURCEMANAGER_H

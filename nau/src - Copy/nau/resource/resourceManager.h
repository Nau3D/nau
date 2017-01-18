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

			ITexture* createTexture (std::string label, std::string internalFormat, int width, 
				int height, int depth = 1, int layers = 1, int levels = 1, int samples = 1);
			ITexture* createTexture(std::string label, int internalFormat, int width,
				int height, int depth = 1, int layers = 1, int levels = 1, int samples = 1);

			/// create texture with default attributes, texture requires building prior to usage
			ITexture * createTexture(std::string label);

			/// create texture from file
			ITexture* addTexture (std::string fn, std::string label = "", bool mipmap = true);
			/// create cube map from file
			ITexture* addTexture (std::vector<std::string> &fn, std::string &label, bool mipmap = true);
			
			void removeTexture (const std::string &name);

			bool hasTexture(const std::string &name);
			ITexture* getTexture (const std::string &name);
			ITexture* getTexture(unsigned int i);
			ITexture* getTextureByID(unsigned int id);
			int getNumTextures();

			/***Rendertargets***/
			nau::render::IRenderTarget* createRenderTarget (std::string name);
			void removeRenderTarget (std::string name);
			bool hasRenderTarget(const std::string &name);
			nau::render::IRenderTarget *getRenderTarget(const std::string &name);
			int getNumRenderTargets();
			void getRenderTargetNames(std::vector<std::string> *v);


			/***Renderables***/
			std::string makeMeshName(const std::string &name, const std::string &filename);
			std::shared_ptr<IRenderable> &createRenderable(const std::string &type, const std::string &name);
			void removeRenderable(const std::string &name);
			bool hasRenderable (const std::string &name);
			std::shared_ptr<IRenderable> &getRenderable (const std::string &name);

			/***States***/
			IState * createState(const std::string &stateName);
			bool hasState (const std::string &stateName);
			nau::material::IState* getState (const std::string &stateName);

			/***Shaders***/
			bool hasProgram (std::string programName);
			nau::material::IProgram* getProgram (std::string programName);
			unsigned int getNumPrograms();
			std::vector<std::string> *getProgramNames();

			/***Buffers***/
			nau::material::IBuffer* getBuffer(std::string name);
			nau::material::IBuffer* createBuffer(std::string name);
			bool hasBuffer(std::string name);
			nau::material::IBuffer* getBufferByID(int id);
			void getBufferNames(std::vector<std::string> *names);
			void clearBuffers();

			/***Arrays of Textures***/
			nau::material::IArrayOfTextures* getArrayOfTextures(std::string name);
			nau::material::IArrayOfTextures* createArrayOfTextures(std::string name);
			bool hasArrayOfTextures(std::string name);
			void getArrayOfTexturesNames(std::vector<std::string> *names);
		};
	};
};
#endif //RESOURCEMANAGER_H

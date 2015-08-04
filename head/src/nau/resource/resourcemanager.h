#ifndef RESOURCEMANAGER_H
#define RESOURCEMANAGER_H

#if NAU_OPENGL_VERSION >= 430
#include "nau/render/iBuffer.h"
#endif
#include "nau/material/material.h"
#include "nau/render/irenderable.h"
#include "nau/render/rendertarget.h"
#include "nau/render/texture.h"
#include "nau/resource/texturemanager.h"
#include "nau/scene/sceneobject.h"


#include <vector>
#include <string>
#include <map>

namespace nau 
{
	namespace resource
	{
		class ResourceManager
		{
		private:
			TextureManager* m_pTextureManager;
			std::string m_Path;
			
			std::map<std::string, nau::render::RenderTarget*> m_RenderTargets; /***MARK***/ //Replace with a manager?
			std::map<std::string, nau::render::IRenderable*> m_Meshes;
			std::map<std::string, nau::render::IProgram*> m_Programs;
			std::map<std::string, nau::render::IState*> m_States;

			std::map<std::string, nau::render::IBuffer*> m_Buffers;
			static int renderableCount;

		public:
			ResourceManager (std::string path);
			~ResourceManager (void);

			void clear();

			/***Textures***/
			bool hasTexture(std::string name);
			int getNumTextures();
			nau::render::Texture* getTexture(unsigned int i);
			nau::render::Texture* getTextureByID(unsigned int id);
			nau::render::Texture* getTexture (std::string name);
			nau::render::Texture* addTexture (std::string fn, std::string label = "", bool mipmap = 1);
			nau::render::Texture* addTexture (std::vector<std::string> fn, std::string label, bool mipmap = 1);
			void removeTexture (std::string name);
			
			//nau::render::Texture* createTexture (std::string label, 
			//	std::string internalFormat, 
			//	std::string aFormat, 
			//	std::string aType, int width, int height,
			//	unsigned char* data = NULL);

			nau::render::Texture* createTexture (std::string label, 
				std::string internalFormat, 
				int width, int height, int depth = 1, int layers = 1, int levels = 1, int samples = 1);

			//nau::render::Texture* createTextureMS (std::string label, 
			//	std::string internalFormat, 
			//	int width, int height,
			//	int samples);

			/// create texture with default attributes, texture requires building prior to usage
			nau::render::Texture * createTexture(std::string label);

			/***TexImage***/
			nau::material::TexImage* createTexImage(nau::render::Texture *t);
			nau::material::TexImage* getTexImage(std::string aTextureName);


			/***Rendertargets***/
			nau::render::RenderTarget* createRenderTarget (std::string name);
			void removeRenderTarget (std::string name);
			bool hasRenderTarget(const std::string &name);
			nau::render::RenderTarget *getRenderTarget(const std::string &name);
			int getNumRenderTargets();
			std::vector<std::string>* ResourceManager::getRenderTargetNames();

			/***Renderables***/
			nau::render::IRenderable* createRenderable(std::string type, std::string name="", std::string filename = "");
			bool hasRenderable (std::string meshName, std::string filename);
			nau::render::IRenderable* getRenderable (std::string meshName, std::string filename);
			nau::render::IRenderable* addRenderable (nau::render::IRenderable* aMesh, std::string filename);
			void removeRenderable(std::string name);

			/***States***/
			bool hasState (std::string stateName);
			nau::render::IState* getState (std::string stateName);
			void addState (nau::render::IState* aState);

			/***Shaders***/
			bool hasProgram (std::string programName);
			nau::render::IProgram* getProgram (std::string programName);
			unsigned int getNumPrograms();
			std::vector<std::string> *getProgramNames();


			/***Buffers***/
			nau::render::IBuffer* getBuffer(std::string name);
			nau::render::IBuffer* createBuffer(std::string name);
			bool hasBuffer(std::string name);
			nau::render::IBuffer* getBufferByID(int id);
			void getBufferNames(std::vector<std::string> &names);
			void clearBuffers();
		};
	};
};
#endif //RESOURCEMANAGER_H

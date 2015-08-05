#ifndef RESOURCEMANAGER_H
#define RESOURCEMANAGER_H

#if NAU_OPENGL_VERSION >= 430
#include "nau/material/iBuffer.h"
#endif
#include "nau/material/material.h"
#include "nau/material/texture.h"
#include "nau/render/iRenderable.h"
#include "nau/render/renderTarget.h"
#include "nau/resource/texturemanager.h"
#include "nau/scene/sceneobject.h"


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
			TextureManager* m_pTextureManager;
			std::string m_Path;
			
			std::map<std::string, nau::render::RenderTarget*> m_RenderTargets; /***MARK***/ //Replace with a manager?
			std::map<std::string, nau::render::IRenderable*> m_Meshes;
			std::map<std::string, nau::material::IProgram*> m_Programs;
			std::map<std::string, nau::material::IState*> m_States;

			std::map<std::string, nau::material::IBuffer*> m_Buffers;
			static int renderableCount;

		public:
			ResourceManager (std::string path);
			~ResourceManager (void);

			void clear();

			/***Textures***/
			bool hasTexture(std::string name);
			int getNumTextures();
			Texture* getTexture(unsigned int i);
			Texture* getTextureByID(unsigned int id);
			Texture* getTexture (std::string name);
			Texture* addTexture (std::string fn, std::string label = "", bool mipmap = 1);
			Texture* addTexture (std::vector<std::string> fn, std::string label, bool mipmap = 1);
			void removeTexture (std::string name);
			
			//nau::render::Texture* createTexture (std::string label, 
			//	std::string internalFormat, 
			//	std::string aFormat, 
			//	std::string aType, int width, int height,
			//	unsigned char* data = NULL);

			Texture* createTexture (std::string label, 
				std::string internalFormat, 
				int width, int height, int depth = 1, int layers = 1, int levels = 1, int samples = 1);

			//nau::render::Texture* createTextureMS (std::string label, 
			//	std::string internalFormat, 
			//	int width, int height,
			//	int samples);

			/// create texture with default attributes, texture requires building prior to usage
			Texture * createTexture(std::string label);

			/***TexImage***/
			nau::material::TexImage* createTexImage(Texture *t);
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
			nau::material::IState* getState (std::string stateName);
			void addState (nau::material::IState* aState);

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
			void getBufferNames(std::vector<std::string> &names);
			void clearBuffers();
		};
	};
};
#endif //RESOURCEMANAGER_H

#ifndef RESOURCEMANAGER_H
#define RESOURCEMANAGER_H

#include "nau/material/material.h"
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
			//TextureManager* m_pTextureManager;
			std::string m_Path;
			
			std::map<std::string, nau::render::IRenderTarget*> m_RenderTargets;
			std::map<std::string, nau::render::IRenderable*> m_Meshes;
			std::map<std::string, nau::material::IProgram*> m_Programs;
			std::map<std::string, nau::material::IState*> m_States;
			//std::map<std::string, nau::material::ITexImage*> m_TexImages;    
			std::vector<nau::material::ITexture*> m_Textures;
			std::map<std::string, nau::material::IBuffer*> m_Buffers;
			static int renderableCount;

		public:
			ResourceManager (std::string path);
			~ResourceManager (void);

			void clear();

			/***Textures***/
			bool hasTexture(std::string name);
			int getNumTextures();
			ITexture* getTexture(unsigned int i);
			ITexture* getTextureByID(unsigned int id);
			ITexture* getTexture (std::string name);
			ITexture* addTexture (std::string fn, std::string label = "", bool mipmap = 1);
			ITexture* addTexture (std::vector<std::string> fn, std::string label, bool mipmap = 1);
			void removeTexture (std::string name);
			
			//nau::render::ITexture* createTexture (std::string label, 
			//	std::string internalFormat, 
			//	std::string aFormat, 
			//	std::string aType, int width, int height,
			//	unsigned char* data = NULL);

			ITexture* createTexture (std::string label, 
				std::string internalFormat, 
				int width, int height, int depth = 1, int layers = 1, int levels = 1, int samples = 1);

			//nau::render::ITexture* createTextureMS (std::string label, 
			//	std::string internalFormat, 
			//	int width, int height,
			//	int samples);

			/// create texture with default attributes, texture requires building prior to usage
			ITexture * createTexture(std::string label);

			/***ITexImage***/
			//nau::material::ITexImage* createTexImage(ITexture *t);
			//nau::material::ITexImage* getTexImage(std::string aTextureName);


			/***Rendertargets***/
			nau::render::IRenderTarget* createRenderTarget (std::string name);
			void removeRenderTarget (std::string name);
			bool hasRenderTarget(const std::string &name);
			nau::render::IRenderTarget *getRenderTarget(const std::string &name);
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

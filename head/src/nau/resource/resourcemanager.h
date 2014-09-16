#ifndef RESOURCEMANAGER_H
#define RESOURCEMANAGER_H

#include <nau/render/irenderable.h>
#include <nau/resource/texturemanager.h>
#include <nau/scene/sceneobject.h>
#include <nau/render/rendertarget.h>
#include <nau/render/texture.h>
#include <nau/material/material.h>

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
			
			nau::render::Texture* createTexture (std::string label, 
				std::string internalFormat, 
				std::string aFormat, 
				std::string aType, int width, int height,
				unsigned char* data = NULL);

			nau::render::Texture* createTexture (std::string label, 
				std::string internalFormat, 
				int width, int height, int layers = 0);

			nau::render::Texture* createTextureMS (std::string label, 
				std::string internalFormat, 
				int width, int height,
				int samples);

			// TEXIMAGE
			nau::material::TexImage* createTexImage(nau::render::Texture *t);
			nau::material::TexImage* getTexImage(std::string aTextureName);


			/***Rendertargets***/
			nau::render::RenderTarget* createRenderTarget (std::string name, int width, int height);
			void removeRenderTarget (std::string name);
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

			/***SHADERS***/
			bool hasProgram (std::string programName);
			nau::render::IProgram* getProgram (std::string programName);
			unsigned int getNumPrograms();
			std::vector<std::string> *getProgramNames();

			/***Materials***/
//			nau::material::Material* addMaterial (nau::material::Material* aMaterial);
			//void deleteTexImage(std::string aTextureName);
			//nau::render::Texture* newEmptyTexture(std::string &name);
			//nau::render::Texture* createTexture (std::string label, 
			//	std::string internalFormat, 
			//	std::string aFormat, 
			//	std::string aType, int width, int height);
			//ISceneObject* addSceneObject (void);
		};
	};
};
#endif //RESOURCEMANAGER_H

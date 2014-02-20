#ifndef TEXTUREMANAGER_H
#define TEXTUREMANAGER_H

#include <string>
#include <vector>
#include <map>

//#include <nau/render/textureCubeMap.h>
#include <nau/render/texture.h>
#include <nau/material/teximage.h>


namespace nau 
{
	namespace resource
	{

		class TextureManager {
		
		friend class ResourceManager;

		private:
		        std::string m_Path;	
		        std::vector<nau::render::Texture*> m_Lib;
				std::map<std::string, nau::material::TexImage*> m_TexImageLib;

		public:

			void clear();
			void setPath (std::string path);

			nau::render::Texture* addTexture (std::string filename, std::string label, bool mipmap=true);
			nau::render::Texture* addTexture (std::vector<std::string> filenames, std::string label, bool mipmap=true);

			nau::render::Texture* createTexture (
						std::string label, 
						std::string internalFormat, 
						std::string aFormat, 
						std::string aType, int width, int height,
						unsigned char* data = NULL);

			nau::render::Texture* createTexture (
						std::string label, 
						std::string internalFormat, 
						int width, int height);

			nau::render::Texture* createTextureMS (
						std::string label, 
						std::string internalFormat, 
						int width, int height,
						int samples);


			void removeTexture (std::string name);

			bool hasTexture(std::string &name);
			nau::render::Texture* getTexture (int id);
			nau::render::Texture* getTexture (std::string name);
			std::vector<std::string>* getTextureLabels();

			int getNumTextures ();
			
			nau::render::Texture* getTextureOrdered (unsigned int position);
			int getTexturePosition (nau::render::Texture *t);


			// TEXIMAGE
			nau::material::TexImage* createTexImage(nau::render::Texture *t);
			nau::material::TexImage* getTexImage(std::string aTextureName);

			//void deleteTexImage(std::string aTextureName);
			//nau::render::Texture* addTexture (int id, char *label);
			//nau::render::Texture* createTexture ();			
			//nau::render::Texture* newEmptyTexture(std::string name);

		  protected:
			  TextureManager (std::string path);
			  //TextureManager(const CTextureManager&);
			  //TextureManager& operator= (const CTextureManager&);
		};
	};
};

#endif //TEXTUREMANAGER_H

#ifndef TEXTUREMANAGER_H
#define TEXTUREMANAGER_H

#include "nau/material/texture.h"
#include "nau/material/texImage.h"

#include <string>
#include <vector>
#include <map>

namespace nau 
{
	namespace resource
	{

		class TextureManager {
		
		friend class ResourceManager;

		private:
		        std::string m_Path;	
		        std::vector<nau::material::Texture*> m_Lib;
				std::map<std::string, nau::material::TexImage*> m_TexImageLib;

		public:

			void clear();
			void setPath (std::string path);

			// for "regular" textures
			nau::material::Texture* addTexture (std::string filename, std::string label, bool mipmap=true);
			// for cubemap textures
			nau::material::Texture* addTexture (std::vector<std::string> filenames, std::string label, bool mipmap=true);

			//nau::render::Texture* createTexture (
			//			std::string label, 
			//			std::string internalFormat, 
			//			std::string aFormat, 
			//			std::string aType, int width, int height,
			//			unsigned char* data = NULL);

			nau::material::Texture* createTexture (
						std::string label, 
						std::string internalFormat, 
						int width, int height, int depth = 1, int layers = 1, int levels = 1, int samples = 1);

			//nau::render::Texture* createTextureMS (
			//			std::string label, 
			//			std::string internalFormat, 
			//			int width, int height,
			//			int samples);

			/// create texture with default attributes, texture requires building prior to usage
			nau::material::Texture * createTexture(std::string label);

			void removeTexture (std::string name);

			bool hasTexture(std::string &name);
			nau::material::Texture* getTexture (int id);
			nau::material::Texture* getTexture (std::string name);
			std::vector<std::string>* getTextureLabels();

			int getNumTextures ();
			
			nau::material::Texture* getTextureOrdered (unsigned int position);
			int getTexturePosition (nau::material::Texture *t);


			// TEXIMAGE
			nau::material::TexImage* createTexImage(nau::material::Texture *t);
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

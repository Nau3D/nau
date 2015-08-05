#ifndef FONT_H
#define FONT_H

#include "nau/render/iRenderable.h"
#include "nau/material/texture.h"
#include "nau/material/texImage.h"

#include <map>
#include <string>

using namespace nau::render;

namespace nau 
{
	namespace resource
	{

		class Font {

		public:
			Font();
			~Font();


			void createSentenceRenderable(IRenderable &renderable, std::string sentence);
			void setMaterialName(std::string aMatName);
			const std::string &getMaterialName();

			void setName(std::string fontName);
			const std::string &getFontName();

			void setFixedSize(bool f);
			bool getFixedSize();

			// these methods should called only by FontXMLLoader
			void setFontHeight(unsigned int height);
			void addChar(char code, int width, float x1, float x2, float y1, float y2, int A, int C);

		protected:


			std::string mFontName;
			bool mFixedSize;


			class Char {
				public:
					// TexCoords
					float x1,x2,y1,y2;
					/// Char width
					int width;
					int A,C;
			};

			/** Font char data, results from parsing 
			 * the XML file for the font */
			std::map<char, Char> mChars;
			/// font char height
			unsigned int mHeight;
			/// total chars parsed
			unsigned int mNumChars;

			std::string mMaterialName;


		};
	}
};

#endif
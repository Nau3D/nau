#ifndef TEXTURE_H
#define TEXTURE_H

#include "nau/config.h"
#include "nau/attribute.h"
#include "nau/attributeValues.h"
#include "nau/material/textureSampler.h"


#ifdef __SLANGER__
#include <wx/bitmap.h>
#include <wx/image.h>
#include <IL/ilu.h>
#endif

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <string>


using namespace nau;

namespace nau {

	namespace material {
		class TextureSampler;
	}
}

namespace nau
{
	namespace render
	{
		class Texture: public AttributeValues
		{
		public:

			ENUM_PROP(DIMENSION, 0);
			ENUM_PROP(FORMAT, 1);
			ENUM_PROP(TYPE, 2);
			ENUM_PROP(INTERNAL_FORMAT, 3);
			ENUM_PROP(COUNT_ENUMPROPERTY, 4);
			
			INT_PROP(ID, 0);
			INT_PROP(WIDTH, 1);
			INT_PROP(HEIGHT, 2);
			INT_PROP(DEPTH, 3);
			INT_PROP(LEVELS, 4);
			INT_PROP(SAMPLES, 5);
			INT_PROP(LAYERS, 6);
			INT_PROP(COMPONENT_COUNT, 7);
			INT_PROP(ELEMENT_SIZE, 8);

			BOOL_PROP(MIPMAP, 0);

			static AttribSet Attribs;

			//int addAtrib(std::string name, Enums::DataType dt, void *value);

			// Note: no validation is performed!
			//void setProp(int prop, Enums::DataType type, void *value);

			static Texture* Create (std::string file, std::string label, bool mipmap=true);
			//static Texture* Create (std::string label);

			//static Texture* Create(std::string label, std::string internalFormat,
			//	std::string aFormat, std::string aType, int width, int height, 
			//	unsigned char* data );

			static Texture* Create(std::string label, std::string internalFormat,
				int width, int height, int depth = 1, int layers = 1, int levels = 1, int samples = 1);

			//static Texture* CreateMS(std::string label, std::string internalFormat,
			//	int width, int height, 
			//	int samples );

			static Texture* Create(std::string label);
	

#ifdef __SLANGER__
			virtual wxBitmap *getBitmap(void);
#endif
			virtual std::string& getLabel (void);
			virtual void setLabel (std::string label);
			//! prepare a texture for rendering
			virtual void prepare(unsigned int unit, nau::material::TextureSampler *ts) = 0;
			//! restore default texture in texture unit
			virtual void restore(unsigned int unit) = 0;
			//! builds a texture with the attribute parameters previously set
			virtual void build() = 0;
			virtual ~Texture(void);

		protected:
			// For textures with data, ex. loaded images
			//Texture(std::string label, std::string aDimension, std::string internalFormat, 
			//	std::string aFormat, std::string aType, int width, int height);
			/// For 2D textures without data, ex texture storage
			//Texture(std::string label, std::string aDimension, std::string internalFormat, 
			//	int width, int height);

			Texture(std::string label);

			/// For inheritance reasons only
			Texture() {bitmap=NULL;};

			static bool Init();
			static bool Inited;

			std::string m_Label;
			unsigned char *m_Bitmap = NULL;
#ifdef __SLANGER__
			wxBitmap *bitmap;
#endif
		};
	};
};

#endif

#ifndef TEXTURE_H
#define TEXTURE_H

#include <nau/material/textureSampler.h>
#include <nau/attribute.h>
#include <nau/attributeValues.h>
#include <nau/config.h>

#ifdef __SLANGER__
#include <wx/bitmap.h>
#include <wx/image.h>
#include <IL/ilu.h>
#endif

#include <string>
#include <math.h>


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

			//typedef enum { DIMENSION, FORMAT, TYPE, INTERNAL_FORMAT,
			//	COUNT_ENUMPROPERTY} EnumProperty;
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
			//typedef enum { ID, WIDTH, HEIGHT, DEPTH, LEVELS, SAMPLES, LAYERS, COMPONENT_COUNT,ELEMENT_SIZE,
			//	COUNT_INTPROPERTY} IntProperty;

			//typedef enum { COUNT_UINTPROPERTY} UIntProperty;

			BOOL_PROP(MIPMAP, 0);
			FLOAT_PROP(DUMMY, 0);
			//typedef enum { MIPMAP, COUNT_BOOLPROPERTY } BoolProperty;

			//typedef enum {COUNT_FLOAT4PROPERTY} Float4Property;
			//typedef enum {COUNT_FLOATPROPERTY} FloatProperty;


			static AttribSet Attribs;

			//std::map<int,int> m_IntProps;
			//std::map<int,int> m_EnumProps;
			//std::map<int,unsigned int> m_UIntProps;
			//std::map<int,bool> m_BoolProps;
			//std::map<int, vec4> m_Float4Props;
			//std::map<int, float> m_FloatProps;

			int addAtrib(std::string name, Enums::DataType dt, void *value);

			// Note: no validation is performed!
			void setProp(int prop, Enums::DataType type, void *value);
			//void *getProp(int prop, Enums::DataType type);

			//int getPropi(IntProperty prop);
			//int getPrope(EnumProperty prop);
			//unsigned int getPropui(UIntProperty prop);
			//bool getPropb(BoolProperty prop);

			//void initArrays();

			static Texture* Create (std::string file, std::string label, bool mipmap=true);
			//static Texture* Create (std::string label);

			static Texture* Create(std::string label, std::string internalFormat,
				std::string aFormat, std::string aType, int width, int height, 
				unsigned char* data );

			static Texture* Create(std::string label, std::string internalFormat,
				int width, int height, int layers = 0);

			static Texture* CreateMS(std::string label, std::string internalFormat,
				int width, int height, 
				int samples );
	
//#if NAU_OPENGL_VERSION < 420 || NAU_OPTIX
//			static int GetCompatibleFormat(int anInternalFormat);
//			static int GetCompatibleType(int aFormat);
//#endif

#ifdef __SLANGER__
			virtual wxBitmap *getBitmap(void);
#endif
			virtual std::string& getLabel (void);
			virtual void setLabel (std::string label);

			virtual void prepare(unsigned int unit, nau::material::TextureSampler *ts) = 0;
			virtual void restore(unsigned int unit) = 0;
		
			virtual ~Texture(void);

		protected:
			// For textures with data, ex. loaded images
			Texture(std::string label, std::string aDimension, std::string internalFormat, 
				std::string aFormat, std::string aType, int width, int height);
			/// For 2D textures without data, ex texture storage
			Texture(std::string label, std::string aDimension, std::string internalFormat, 
				int width, int height);
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

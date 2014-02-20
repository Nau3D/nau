#include <nau/config.h>

 #if NAU_OPENGL_VERSION >= 420

#ifndef IMAGE_TEXTURE_H
#define IMAGE_TEXTURE_H


#include <string>
#include <math.h>

#include <nau/attribute.h>




using namespace nau;

namespace nau
{
	namespace render
	{
		class ImageTexture
		{
		public:

			typedef enum { ACCESS,
				COUNT_ENUMPROPERTY} EnumProperty;

			typedef enum { 
				COUNT_INTPROPERTY} IntProperty;

			typedef enum {LEVEL, TEX_ID, COUNT_UINTPROPERTY} UIntProperty;

			typedef enum {COUNT_FLOAT4PROPERTY} Float4Property;
			typedef enum {COUNT_FLOATPROPERTY} FloatProperty;


			static AttribSet Attribs;

			std::map<int,int> m_EnumProps;
			std::map<int,int> m_IntProps;
			std::map<int,unsigned int> m_UIntProps;
			std::map<int,bool> m_BoolProps;
			std::map<int, vec4> m_Float4Props;
			std::map<int, float> m_FloatProps;

			// Note: no validation is performed!
			void setProp(int prop, Enums::DataType type, void *value);

			int getPropi(IntProperty prop);
			unsigned int getPropui(UIntProperty prop);
			void *getProp(int prop, Enums::DataType type);

			void initArrays();

			static ImageTexture* Create (std::string label, unsigned int texID, unsigned int level, unsigned int access);
			static ImageTexture* Create (std::string label, unsigned int texID);

			virtual void prepare(int unit) = 0;
			virtual void restore() = 0;
		
			virtual ~ImageTexture(void){};

			virtual std::string& getLabel (void);
			virtual void setLabel (std::string label);

		protected:
			ImageTexture() {};

			static bool Init();
			static bool Inited;

			std::string m_Label;
			unsigned int m_InternalFormat;

		};
	};
};

#endif

#endif
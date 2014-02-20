#ifndef COLORMATERIAL_H
#define COLORMATERIAL_H

#include <fstream>
#include <iostream>

#include <nau/enums.h>

namespace nau
{
	namespace material
	{

		class ColorMaterial {

		private:
			float m_Ambient[4];
			float m_Specular[4];
			float m_Emission[4];
			float m_Diffuse[4];
			float m_Shininess;


		public:

			typedef enum {
				AMBIENT,
				SPECULAR,
				EMISSION,
				DIFFUSE,
				AMBIENT_AND_DIFFUSE,
				SHININESS,
				COUNT_COLORCOMPONENTS
			} ColorComponent;

			static const std::string ColorString[COUNT_COLORCOMPONENTS];
			static void getComponentTypeAndId(std::string s, nau::Enums::DataType *dt, ColorComponent *c);
			static bool validateComponent(std::string s);

			ColorMaterial();
			~ColorMaterial();

			void setColorComponent(ColorComponent c, float r, float g = 0, float b = 0, float a = 0);
			void setColorComponent(ColorComponent c, float *f);
			float *getColorCompoment(ColorComponent c);

			void setAmbient (const float *values);
			void setAmbient (float r, float g, float b, float a);
			const float* getAmbient (void) const;

			void setSpecular (const float *values);
			void setSpecular (float r, float g, float b, float a);
			const float* getSpecular (void) const;

			void setEmission (const float *values);
			void setEmission (float r, float g, float b, float a);
			const float* getEmission (void) const;

			void setDiffuse (const float *values);
			void setDiffuse (float r, float g, float b, float a);
			const float* getDiffuse (void) const;

			void setShininess (float shininess);
			const float getShininess () const;
			float *getShininessPtr();

			void prepare ();
			void restore();
			void clear();

			void clone(const ColorMaterial &mat);

			//void save(std::ofstream &f);
			//void load(std::ifstream &f);

		};
	};
};

#endif // COLORMATERIAL_H

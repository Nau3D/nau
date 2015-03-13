#ifndef MATERIAL_H
#define MATERIAL_H

#include "nau/clogger.h"
#include "nau/material/colormaterial.h"
#include "nau/material/materialTexture.h"
#include "nau/material/programvalue.h"
//#include "nau/material/texturemat.h"
#include "nau/material/imaterialbuffer.h"
#include "nau/render/imageTexture.h"
#include "nau/render/iprogram.h"
#include "nau/render/istate.h" 
#include "nau/render/texture.h"

#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace nau::render;

namespace nau
{
	namespace material
	{
		class MaterialLibManager;
		class MaterialLib;
		class Material {
			
			friend class MaterialLibManager;
			friend class MaterialLib;

		private:
		   
//			typedef enum  { FRONT_TO_BACK, BACK_TO_FRONT, NONE} orderType;

			nau::material::ColorMaterial m_Color;
//			nau::material::TextureMat *m_Texmat;
#if NAU_OPENGL_VERSION >=  420
			std::map<int, ImageTexture*> m_ImageTextures;
#endif

			// ID -> (binding point, *buffer)
			std::map<int, IMaterialBuffer *> m_Buffers;

			std::map<int, MaterialTexture *> m_Textures;

//			std::string m_Shader;
			nau::render::IProgram *m_Shader;
			nau::render::IState *m_State;

			// These are values specified in the material library
			std::map<std::string, nau::material::ProgramValue> m_ProgramValues;
			std::map<std::string, nau::material::ProgramValue> m_UniformValues;
			
			bool m_Enabled;
			bool m_useShader;
			std::string m_Name;	
			Material();

		public:
			~Material();

			Material *clone();

			void setName (std::string name);
			std::string& getName ();

			void prepare ();
			void prepareNoShaders();
			void setUniformValues();
			void restore();
			void restoreNoShaders();

			// Reset material to defaults
			void clear();
			void enable (void);
			void disable (void);
			bool isEnabled (void);

#if NAU_OPENGL_VERSION >=  420
			void attachImageTexture(std::string label, unsigned int unit, unsigned int texID);
			ImageTexture *getImageTexture(unsigned int unit);
#endif

			void attachBuffer(IMaterialBuffer *b);
			bool hasBuffer(int id);
			IMaterialBuffer *getBuffer(int id);
			//int getBufferBindingPoint(int id);

//			nau::material::TextureMat* getTextures (void);
			MaterialTexture *getMaterialTexture(int unit);
			bool createTexture (int unit, std::string fn);
			void attachTexture (int unit, std::string label);
			void attachTexture(int unit, Texture *t);
			Texture *getTexture(int unit);
			TextureSampler* getTextureSampler(unsigned int unit);
			void unsetTexture(int unit);
			void getTextureNames(std::vector<std::string> *vs);
			void getTextureUnits(std::vector<int> *vi);

			void attachProgram (std::string shaderName);
			void Material::cloneProgramFromMaterial(Material *mat);
			nau::render::IProgram *getProgram();
			std::string getProgramName();
			bool isInSpecML(std::string programValueName);
			void addProgramValue (std::string name, nau::material::ProgramValue progVal);
			void enableShader(bool value);
			bool isShaderEnabled();
			void clearProgramValues(); 
			void checkProgramValuesAndUniforms();

			std::map<std::string, nau::material::ProgramValue>& getProgramValues();
			std::map<std::string, nau::material::ProgramValue>& getUniformValues();
			ProgramValue *getProgramValue(std::string name);
			void setValueOfUniform(std::string name, void *values);
			void getValidProgramValueNames(std::vector<std::string> *vs);
			void getUniformNames(std::vector<std::string> *vs);

			nau::material::ColorMaterial& getColor (void);
			nau::render::IState* getState (void);

			void setState(IState *s);
		};
	};
};
#endif // MATERIAL_H


#ifndef MATERIAL_H
#define MATERIAL_H

#include "nau/clogger.h"
#include "nau/material/colorMaterial.h"
#include "nau/material/iImageTexture.h"
#include "nau/material/iMaterialBuffer.h"
#include "nau/material/iProgram.h"
#include "nau/material/iState.h" 
#include "nau/material/materialTexture.h"
#include "nau/material/programBlockValue.h"
#include "nau/material/programValue.h"
#include "nau/material/iTexture.h"

#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

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
		   
			nau::material::ColorMaterial m_Color;
			std::map<int, IImageTexture *> m_ImageTextures;

			// ID -> (binding point, *buffer)
			std::map<int, IMaterialBuffer *> m_Buffers;

			std::map<int, MaterialTexture *> m_Textures;

			IProgram *m_Shader;
			IState *m_State;

			// These are values specified in the material library
			std::map<std::string, nau::material::ProgramValue> m_ProgramValues;
			// These are the active program uniforms
			std::map<std::string, nau::material::ProgramValue> m_UniformValues;
			
			std::map<std::pair<std::string, std::string>, nau::material::ProgramBlockValue> m_ProgramBlockValues;
			bool m_Enabled;
			bool m_useShader;
			std::string m_Name;	
			Material();
			std::shared_ptr<Material> clone();

		public:
			~Material();


			void setName (std::string name);
			std::string& getName ();

			void prepare ();
			void prepareNoShaders();
			void restore();
			void restoreNoShaders();

			void setUniformValues();
			void setUniformBlockValues();

			// Reset material to defaults
			void enable (void);
			void disable (void);
			bool isEnabled (void);

			void attachImageTexture(std::string label, unsigned int unit, unsigned int texID);
			IImageTexture *getImageTexture(unsigned int unit);
			void getImageTextureUnits(std::vector<unsigned int> *v);

			void attachBuffer(IMaterialBuffer *b);
			bool hasBuffer(int id);
			IMaterialBuffer *getBuffer(int id);
			void getBufferBindings(std::vector<unsigned int> *vi);

			MaterialTexture *getMaterialTexture(int unit);
			bool createTexture (int unit, std::string fn);
			void attachTexture (int unit, std::string label);
			void attachTexture(int unit, ITexture *t);
			ITexture *getTexture(int unit);
			ITextureSampler* getTextureSampler(unsigned int unit);
			void unsetTexture(int unit);
			void getTextureNames(std::vector<std::string> *vs);
			void getTextureUnits(std::vector<unsigned int> *vi);

			void attachProgram (std::string shaderName);
			void Material::cloneProgramFromMaterial(std::shared_ptr<Material> &mat);
			IProgram *getProgram();
			std::string getProgramName();
			void addProgramValue (std::string name, nau::material::ProgramValue progVal);
			void addProgramBlockValue (std::string block, std::string name, nau::material::ProgramBlockValue progVal);
			void enableShader(bool value);
			bool isShaderEnabled();
			void clearProgramValues(); 
			void checkProgramValuesAndUniforms();

			std::map<std::string, nau::material::ProgramValue>& getProgramValues();
			std::map<std::string, nau::material::ProgramValue>& getUniformValues();
			std::map<std::pair<std::string, std::string>, nau::material::ProgramBlockValue>& getProgramBlockValues();
			ProgramValue *getProgramValue(std::string name);
			void setValueOfUniform(std::string name, void *values);
			void getValidProgramValueNames(std::vector<std::string> *vs);
			void getUniformNames(std::vector<std::string> *vs);

			nau::material::ColorMaterial& getColor (void);
			IState* getState (void);

			void setState(IState *s);
		};
	};
};
#endif // MATERIAL_H


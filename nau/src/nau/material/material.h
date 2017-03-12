#ifndef MATERIAL_H
#define MATERIAL_H

#include "nau/clogger.h"
#include "nau/material/colorMaterial.h"
#include "nau/material/iImageTexture.h"
#include "nau/material/iMaterialBuffer.h"
#include "nau/material/iProgram.h"
#include "nau/material/iState.h" 
#include "nau/material/materialArrayOfImageTextures.h"
#include "nau/material/materialTexture.h"
#include "nau/material/materialArrayOfTextures.h"
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


#ifdef _WINDLL
#ifdef nau_EXPORTS
#define nau_API __declspec(dllexport)   
#else  
#define nau_API __declspec(dllimport)   
#endif 
#else
#define nau_API
#endif


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

			MaterialArrayOfTextures m_ArrayOfTextures;
			MaterialArrayOfImageTextures *m_ArrayOfImageTextures;

			IProgram *m_Shader;
			IState *m_State;

			// These are values specified in the material library
			std::map<std::string, nau::material::ProgramValue> m_ProgramValues;
			// These are the active program uniforms
			std::map<std::string, nau::material::ProgramValue> m_UniformValues;
			
			std::map<std::pair<std::string, std::string>, nau::material::ProgramBlockValue> m_ProgramBlockValues;
			bool m_Enabled;
			std::string m_Name;	
			Material();
			std::shared_ptr<Material> clone();

		public:
			nau_API ~Material();


			nau_API void setName (std::string name);
			nau_API std::string& getName ();

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

			void setArrayOfTextures(IArrayOfTextures *at, int unit);
			MaterialArrayOfTextures *getMaterialArrayOfTextures();

			void setArrayOfImageTextures(MaterialArrayOfImageTextures *m);
			MaterialArrayOfImageTextures * getArrayOfImageTextures();

			nau_API void attachImageTexture(std::string label, unsigned int unit, unsigned int texID);
			nau_API IImageTexture *getImageTexture(unsigned int unit);
			nau_API void getImageTextureUnits(std::vector<unsigned int> *v);

			nau_API void attachBuffer(IMaterialBuffer *b);
			nau_API bool hasBuffer(int id);
			nau_API IMaterialBuffer *getMaterialBuffer(int id);
			nau_API IBuffer *getBuffer(int id);
			nau_API void getBufferBindings(std::vector<unsigned int> *vi);

			nau_API MaterialTexture *getMaterialTexture(int unit);
			nau_API bool createTexture (int unit, std::string fn);
			nau_API void attachTexture (int unit, std::string label);
			nau_API void attachTexture(int unit, ITexture *t);
			nau_API ITexture *getTexture(int unit);
			nau_API ITextureSampler* getTextureSampler(unsigned int unit);
			nau_API void unsetTexture(int unit);
			nau_API void getTextureNames(std::vector<std::string> *vs);
			nau_API void getTextureUnits(std::vector<unsigned int> *vi);

			nau_API void attachProgram (std::string shaderName);
			void cloneProgramFromMaterial(std::shared_ptr<Material> &mat);
			nau_API IProgram *getProgram();
			nau_API std::string getProgramName();
			nau_API void addProgramValue (std::string name, nau::material::ProgramValue progVal);
			nau_API void addProgramBlockValue (std::string block, std::string name, nau::material::ProgramBlockValue progVal);
			nau_API bool isShaderLinked();
			nau_API void clearProgramValues();
			nau_API void checkProgramValuesAndUniforms(std::string &result);

			nau_API std::map<std::string, nau::material::ProgramValue>& getProgramValues();
			nau_API std::map<std::string, nau::material::ProgramValue>& getUniformValues();
			nau_API std::map<std::pair<std::string, std::string>, nau::material::ProgramBlockValue>& getProgramBlockValues();
			nau_API ProgramValue *getProgramValue(std::string name);
			nau_API void setValueOfUniform(std::string name, void *values);
			nau_API void getValidProgramValueNames(std::vector<std::string> *vs);
			nau_API void getUniformNames(std::vector<std::string> *vs);

			nau_API nau::material::ColorMaterial& getColor (void);
			nau_API IState* getState (void);

			nau_API void setState(IState *s);
		};
	};
};
#endif // MATERIAL_H


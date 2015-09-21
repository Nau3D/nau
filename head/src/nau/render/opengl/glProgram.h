#ifndef GLPROGRAM_H
#define GLPROGRAM_H


#include "nau/material/iProgram.h"

#include "nau/render/opengl/glUniform.h"

#include <GL/glew.h>

#include <vector>
#include <string>

using namespace nau::material;

namespace nau
{
	namespace render
	{

		class GLProgram : public IProgram 
		{
		public:
			static int ShaderGLId[SHADER_COUNT];

		private:
			std::vector<std::string> m_File; // filenames
			std::vector<std::string> m_Source; // source code
			std::vector<int> m_ID;
			std::vector<bool> m_Compiled;
			GLuint  m_P; // program id
			int m_NumUniforms;
			int m_MaxLength;

			std::string m_Name;
			//list of uniforms
			std::vector<GLUniform> m_Uniforms; 
			//map from block name to location
			std::map<std::string, unsigned int> m_Blocks; 
			
			bool m_PLinked;
			bool m_ShowGlobalUniforms;
		
		public:
		
			GLProgram();
			~GLProgram();

			virtual bool loadShader(IProgram::ShaderType type, const std::string &filename);
			bool reload (void);

			void setName(const std::string &name);
			const std::string &getName();

			virtual const std::string &getShaderFile(ShaderType type);
			virtual bool setShaderFile(ShaderType type, const std::string &name);


			virtual bool getPropertyb(int query);
			virtual int getPropertyi(int query);
			
			bool prepare (void);
			bool restore (void);

			bool setValueOfUniform (const std::string &name, void *values);
			bool setValueOfUniform(int loc, void *values);
			void prepareBlocks();

			int getAttributeLocation (const std::string &name);
			int getUniformLocation(std::string uniformName);

			std::string getShaderInfoLog(ShaderType type);
			char *getProgramInfoLog(); 
			int programValidate();
	
			bool compileShader (IProgram::ShaderType);
			bool linkProgram (void);
			void useProgram (void);

			unsigned int getProgramID();

			bool isCompiled(ShaderType type);
			bool areCompiled();
			bool isLinked();

			int getNumberOfUniforms (void);
			int getNumberOfUserUniforms (void);

			const IUniform &getIUniform(int i);
			const GLUniform  &getUniform (int i);
			const GLUniform& getUniform (const std::string &name);
			void updateUniforms ();
			int findUniform (const std::string &name);
			int findUniformByLocation(int loc);

		private:

			//void init();
			bool reloadShaderFile(IProgram::ShaderType aType);

			void setUniforms();
			void setBlocks();
			void setValueOfUniform (int i);
			void showGlobalUniforms (void);

			void getAttribsLoc();
		};
	};
};

#endif
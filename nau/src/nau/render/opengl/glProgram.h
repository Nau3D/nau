#ifndef GLPROGRAM_H
#define GLPROGRAM_H


#include "nau/material/iProgram.h"

#include "nau/render/opengl/glUniform.h"

#include <glbinding/gl/gl.h>
using namespace gl;

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
			static GLenum ShaderGLId[SHADER_COUNT];

		private:
			std::vector<std::string> m_File; // filenames
			std::vector<std::string> m_Source; // source code
			// vector of (shader ID , attached)
			std::vector<int> m_ID;
			std::vector<bool> m_Attached;

			std::vector<bool> m_Compiled;
			unsigned int  m_P; // program id
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
			const std::string &getProgramInfoLog();
			int programValidate();
	
			bool compileShader (IProgram::ShaderType);
			bool linkProgram (void);
			void useProgram (void);

			unsigned int getProgramID();

			bool isCompiled(ShaderType type);
			bool areCompiled();
			bool isLinked();

			void getAttributeNames(std::vector<std::string> *s);

			int getNumberOfUniforms (void);
			int getNumberOfUserUniforms (void);

			void getUniformBlockNames(std::vector<std::string> *s);
			const IUniform &getIUniform(int i);
			nau_API const GLUniform  &getUniform (int i);
			nau_API const GLUniform& getUniform (const std::string &name);
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

			std::string m_ReturnString;
			//void getAttribsLoc();
		};
	};
};

#endif

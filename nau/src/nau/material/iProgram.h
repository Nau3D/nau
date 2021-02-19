#ifndef IPROGRAM_H
#define IPROGRAM_H

#include "nau/config.h"
#include "nau/enums.h"
#include "nau/material/iUniform.h"

#include <string>
#include <vector>

namespace nau
{
	namespace material
	{
		class IProgram
		{
		protected:
			bool m_HasTessShader;
			IProgram() :m_HasTessShader(false) {}

		public:
			static IProgram* create ();

//#if NAU_OPENGL_VERSION >=430
			const static int SHADER_COUNT = 8;

			enum ShaderType {
				VERTEX_SHADER,
				GEOMETRY_SHADER,
				TESS_CONTROL_SHADER,
				TESS_EVALUATION_SHADER,
				FRAGMENT_SHADER,
				COMPUTE_SHADER,
				TASK_SHADER,
				MESH_SHADER
			 };

			static std::vector<std::string> ShaderNames;
			static nau_API std::vector<std::string> &GetShaderNames();

			virtual bool isShaderSupported(IProgram::ShaderType);
			bool hasTessellationShader();
			virtual bool loadShader(IProgram::ShaderType type, const std::vector<std::string> &files) = 0;
			virtual bool reload (void) = 0;
			
			virtual  bool prepare (void) = 0;
			virtual  bool restore (void) = 0;

			virtual std::string getShaderInfoLog(ShaderType type) = 0;
			virtual const std::string &getProgramInfoLog() = 0; 
			virtual int programValidate() = 0;

			virtual void getAttributeNames(std::vector<std::string> *s) = 0;

			virtual bool setValueOfUniform (const std::string &name, void *values) = 0; 
			virtual void prepareBlocks() = 0;

			virtual int getNumberOfUniforms (void) = 0;
			virtual int getNumberOfUserUniforms (void) = 0;
			virtual void getUniformBlockNames(std::vector<std::string> *s) = 0;

			virtual int getAttributeLocation (const std::string &name) = 0;
			virtual int getUniformLocation(std::string uniformName) = 0;

			virtual const std::vector<std::string> &getShaderFiles(ShaderType type) = 0;
			virtual bool setShaderFiles(ShaderType type, const std::vector<std::string> &files) = 0;

			virtual ~IProgram(void) {};

			virtual void setName(const std::string &name) = 0;
			virtual const std::string &getName() = 0;

			virtual bool isCompiled(ShaderType type) = 0;
			virtual bool areCompiled() = 0;
			virtual bool isLinked() = 0;

			virtual bool reloadShaderFile(ShaderType type) = 0;

			virtual bool compileShader (IProgram::ShaderType) = 0;
			virtual bool linkProgram (void) = 0;
			virtual void useProgram (void) = 0;

			virtual unsigned int getProgramID() = 0;


			virtual bool getPropertyb(int query) = 0;
			virtual int getPropertyi(int query) = 0;

			virtual const IUniform &getIUniform(int i) = 0;

		};
	};
};

#endif

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
		public:
			static IProgram* create ();

//#if NAU_OPENGL_VERSION >=430
			const static int SHADER_COUNT = 6;

			enum ShaderType {
				VERTEX_SHADER,
				GEOMETRY_SHADER,
				TESS_CONTROL_SHADER,
				TESS_EVALUATION_SHADER,
				FRAGMENT_SHADER,
				COMPUTE_SHADER
			 };
//#elif NAU_OPENGL_VERSION >= 400
//			const static int SHADER_COUNT = 5;
//
//			enum ShaderType {
//				VERTEX_SHADER,
//				GEOMETRY_SHADER,
//				TESS_CONTROL_SHADER,
//				TESS_EVALUATION_SHADER,
//				FRAGMENT_SHADER,
//			 };
//#elif NAU_OPENGL_VERSION >= 320
//			const static int SHADER_COUNT = 3;
//
//			enum ShaderType {
//				VERTEX_SHADER,
//				GEOMETRY_SHADER,
//				FRAGMENT_SHADER,
//			 };
//#else
//			const static int SHADER_COUNT = 2;
//
//			enum ShaderType {
//				VERTEX_SHADER,
//				FRAGMENT_SHADER,
//			 };
//#endif
			static std::string ShaderNames[IProgram::SHADER_COUNT];

			virtual bool isShaderSupported(IProgram::ShaderType);
			virtual bool loadShader(IProgram::ShaderType type, const std::string &filename) = 0;
			virtual bool reload (void) = 0;
			
			virtual  bool prepare (void) = 0;
			virtual  bool restore (void) = 0;

			virtual std::string getShaderInfoLog(ShaderType type) = 0;
			virtual char *getProgramInfoLog() = 0; 
			virtual int programValidate() = 0;

			virtual bool setValueOfUniform (const std::string &name, void *values) = 0; 
			virtual void prepareBlocks() = 0;
			//virtual bool setValueOfUniform(int loc, void *values) = 0;
			//virtual bool setValueOfUniform (const std::string &name, int *values) = 0;

			virtual int getNumberOfUniforms (void) = 0;
			virtual int getNumberOfUserUniforms (void) = 0;
			virtual void getUniformBlockNames(std::vector<std::string> *s) = 0;

			virtual int getAttributeLocation (const std::string &name) = 0;
			virtual int getUniformLocation(std::string uniformName) = 0;

			virtual const std::string &getShaderFile(ShaderType type) = 0;
			virtual bool setShaderFile(ShaderType type, const std::string &name) = 0;

			virtual ~IProgram(void) {};

			virtual void setName(const std::string &name) = 0;
			virtual const std::string &getName() = 0;

			virtual bool isCompiled(ShaderType type) = 0;
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

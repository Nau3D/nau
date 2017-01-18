#ifndef GL_UNIFORM_H
#define GL_UNIFORM_H

#include <glbinding/gl/gl.h>
using namespace gl;

#include "nau/enums.h"
#include "nau/material/iUniform.h"
#include "nau/math/data.h"



#include <string>
#include <map>
#include <memory>

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
	namespace render
	{
		class GLUniform: public nau::material::IUniform {

		private:
			int m_Program;

			unsigned int m_GLType;
			int m_Cardinality;
			int m_Size;
			int m_ArraySize;
			
			int m_Loc;
			//nau::math::Data *m_Values;
			std::shared_ptr<char> m_Values;	

			static bool Init();
			static bool Inited;

		public:
		
			/// converts GLTypes to string
			static std::map<GLenum, std::string> spGLSLType;
			/// converts GLTypes to Enums::DataType, simpler basic types
			static std::map<GLenum, Enums::DataType>   spSimpleType;
			//static std::map<int, int> 	spGLSLTypeSize;

			enum { // values for semantics
				NOT_USED = 0,
				NONE,
				CLAMP,
				COLOR,
				NORMALIZED
			};

			nau_API GLUniform();
			nau_API ~GLUniform();

			void reset (void);
			
			void setGLType(int type, int arraySize);

			int getArraySize();
			int getGLType ();
			std::string getStringGLType();
			

			int getCardinality();
			
			void setArrayValue(int index, void *v);

			//void setValues(nau::math::Data *v);
			//nau::math::Data *getValues();
			void setValues (void *v);
			void *getValues(void);

			int getLoc (void);
			void setLoc (int loc);

			void setValueInProgram();
		};
	};
};

#endif

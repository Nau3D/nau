#ifndef GL_UNIFORM_H
#define GL_UNIFORM_H

#include <string>
#include <map>

#include <GL/glew.h>

#include <nau/enums.h>
#include <nau/render/iprogramvalue.h>


namespace nau
{
	namespace render
	{
		class GLUniform: public nau::render::IUniform {

		private:
			int m_Program;

			GLenum m_GLType;
			int m_Cardinality;
			int m_Size;
			int m_ArraySize;
			
			int m_Loc;
			void *m_Values;	

			static bool Init();
			static bool Inited;

		public:
		
			/// converts GLTypes to string
			static std::map<int, std::string> spGLSLType;
			/// converts GLTypes to Enums::DataType, simpler basic types
			static std::map<int, Enums::DataType>   spSimpleType;
			//static std::map<int, int> 	spGLSLTypeSize;

			enum { // values for semantics
				NOT_USED = 0,
				NONE,
				CLAMP,
				COLOR,
				NORMALIZED
			};

			GLUniform();
			~GLUniform();

			void reset (void);
			

			void setGLType(int type, int arraySize);

			int getArraySize();
			int getGLType ();
			std::string getStringGLType();
			

			int getCardinality();
			
			void setArrayValue(int index, void *v);
			void setValues (void *v);
			void *getValues(void);

			int getLoc (void);
			void setLoc (int loc);

			void setValueInProgram();
		};
	};
};

#endif

#ifndef CUNIFORM_H
#define CUNIFORM_H

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>



namespace nau
{
	namespace render
	{

		class GlUniform {

		private:

			int m_Type;
			int m_Cardinality;
			
			std::string m_Name;
			int m_Semantics;
			float m_UpperLimit, m_LowerLimit;
			int m_Loc;
			float m_Values[16];	

		public:
		
			
			enum { // values for semantics
				NOT_USED = 0,
				NONE,
				CLAMP,
				COLOR,
				NORMALIZED
			};

			GlUniform();
			~GlUniform();

			void reset (void);
			
			void setName (std::string &name);
			std::string &getName (void);

			void setType (int type);
			int getType ();
			std::string GlUniform::getProgramValueType();
			
			int getCardinality();
			
			void setValues (float *v);
			void setValues (int *v);
			float *getValues(void);

			int getLoc (void);
			void setLoc (int loc);

		};
	};
};

#endif

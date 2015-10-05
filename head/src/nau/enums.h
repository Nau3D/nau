#ifndef ENUMS_H
#define ENUMS_H

#include <string>

namespace nau {
	
	class Enums
	{

	public:

		enum DataType {
						INT, IVEC2, IVEC3, IVEC4,
						UINT, UIVEC2, UIVEC3, UIVEC4,
						BOOL, BVEC2, BVEC3, BVEC4,
						FLOAT, VEC2, VEC3, VEC4,
						DOUBLE, DVEC2, DVEC3, DVEC4,
						MAT2, MAT3, MAT4, 
						MAT2x3, MAT2x4, 
						MAT3x2, MAT3x4,
						MAT4x2, MAT4x3,
						DMAT2, DMAT3, DMAT4,
						DMAT2x3, DMAT2x4,
						DMAT3x2, DMAT3x4,
						DMAT4x2, DMAT4x3,

						SAMPLER,
						ENUM,

						BYTE, UBYTE, SHORT, USHORT, STRING,
						COUNT_DATATYPE
		};

		static int getCardinality(DataType p);
		static Enums::DataType getType(std::string s);
		static bool isValidType(std::string s);
		static int getSize(DataType p);
		static void* getDefaultValue(DataType p);
		/// returns true if p1 is compatible with p2 
		static bool isCompatible(DataType p1, DataType p2);
		static bool isBasicType(DataType t);

		static const std::string DataTypeToString[COUNT_DATATYPE];
		static std::string &valueToString(DataType p, void *v);
		static DataType getBasicType(DataType dt);

	private:
		static std::string m_Result;

	};
};

#endif // TYPES_H
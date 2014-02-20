#ifndef ENUMS_H
#define ENUMS_H

#include <string>

namespace nau {
	
	class Enums
	{

	public:

		typedef enum DataType {
						INT,
						UINT,
						SAMPLER,
						BOOL,
						ENUM,
						IVEC2,
						IVEC3,
						IVEC4,
						BVEC2,
						BVEC3,
						BVEC4,
						FLOAT,
						VEC2,
						VEC3,
						VEC4,
						MAT2,
						MAT3,
						MAT4,
						COUNT_DATATYPE
		};

		static int getCardinality(DataType p);
		static Enums::DataType getType(std::string s);
		static bool isValidType(std::string s);
		static int getSize(DataType p);
		/// returns true if p1 is compatible with p2 (note that p2 may be incompatible  with p1)z
		static bool isCompatible(DataType p1, DataType p2);

		static const std::string DataTypeToString[COUNT_DATATYPE];

	};
};

#endif // TYPES_H
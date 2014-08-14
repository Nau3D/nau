#include <nau/enums.h>
#include <nau/errors.h>


using namespace nau;

const std::string Enums::DataTypeToString[] = {
						"INT", "IVEC2", "IVEC3", "IVEC4", 
						"UINT", "UIVEC2", "UIVEC3", "UIVEC4",
						"BOOL", "BVEC2", "BVEC3", "BVEC4", 
						"FLOAT", "VEC2", "VEC3", "VEC4",
						"DOUBLE", "DVEC2", "DVEC3", "DVEC4",

						"MAT2", "MAT3", "MAT4",
						"MAT2x3", "MAT2x4", "MAT3x2", "MAT3x4", "MAT4x2", "MAT4x3",
						"DMAT2", "DMAT3", "DMAT4",
						"DMAT2x3", "DMAT2x4", "DMAT3x2", "DMAT3x4", "DMAT4x2", "DMAT4x3",
						
						"SAMPLER", "ENUM"};




bool 
Enums::isValidType(std::string s) 
{
	for (int i = 0; i < COUNT_DATATYPE; i++) {

		if (s == DataTypeToString[i]) 
			return true;	
	}
	return false;

}


Enums::DataType 
Enums::getType(std::string s) 
{
	for (int i = 0; i < COUNT_DATATYPE; i++) {

		if (s == DataTypeToString[i]) {
		

			return (DataType)i;
		}
	}
	NAU_THROW("Invalid Data Type: %s", s);
}


int
Enums::getSize(DataType p) 
{
	switch (p) {
	
		case INT:
		case IVEC2:
		case IVEC3: 
		case IVEC4:
		case BOOL:
		case BVEC2:
		case BVEC3:
		case BVEC4:
		case SAMPLER:
		case ENUM:
			return(sizeof(int) * getCardinality(p));
		case UINT:
		case UIVEC2:
		case UIVEC3:
		case UIVEC4:
			return(sizeof(unsigned int) * getCardinality(p));
		case FLOAT:
		case VEC2:
		case VEC3:
		case VEC4:
		case MAT2:
		case MAT3:
		case MAT4:
		case MAT2x3:
		case MAT2x4:
		case MAT3x2:
		case MAT3x4:
		case MAT4x2:
		case MAT4x3:
			return (sizeof(float) * getCardinality(p));
		case DOUBLE:
		case DVEC2:
		case DVEC3:
		case DVEC4:
		case DMAT2:
		case DMAT3:
		case DMAT4:
		case DMAT2x3:
		case DMAT2x4:
		case DMAT3x2:
		case DMAT3x4:
		case DMAT4x2:
		case DMAT4x3:
			return (sizeof(double) * getCardinality(p));
		default: 
			return 0;
	}
}


int Enums::getCardinality(DataType p)  
{
	int card = 1;

	switch (p) {
		case INT:
		case UINT:
		case SAMPLER:
		case BOOL:
		case FLOAT:
		case DOUBLE:
		case ENUM:
			card = 1;
			break;
		case IVEC2:
		case BVEC2:
		case VEC2:
		case DVEC2:
			card = 2;
			break;
		case IVEC3:
		case BVEC3:
		case VEC3:
		case DVEC3:
			card = 3;
			break;
		case IVEC4:
		case BVEC4:
		case VEC4:
		case DVEC4: 
		case MAT2:
		case DMAT2:
			card = 4;
			break;
		case MAT3:
		case DMAT3:
			card = 9;
			break;
		case MAT4:
		case DMAT4:
			card = 16;
			break;
		case MAT2x3:
		case MAT3x2:
		case DMAT2x3:
		case DMAT3x2:
			card = 6;
			break;
		case MAT2x4:
		case MAT4x2:
		case DMAT2x4:
		case DMAT4x2:
			card = 8;
			break;
		case MAT3x4:
		case MAT4x3:
		case DMAT3x4:
		case DMAT4x3:
			card = 12;
			break;
	}
	return card;
};


bool
Enums::isCompatible(DataType p1, DataType p2) 
{
	if (p1 == p2)
		return true;

	if ((p1 == ENUM && p2 == INT) || (p1 == INT && p2 == ENUM))
		return true;

	if ((p1 == ENUM && p2 == UINT) || (p1 == UINT && p2 == ENUM))
		return true;
	if (p1 == INT && p2 == UINT)
		return true;

	return false;
}
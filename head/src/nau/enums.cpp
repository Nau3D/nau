#include <nau/enums.h>
#include <nau/errors.h>


using namespace nau;

const std::string Enums::DataTypeToString[] = {"INT", "UINT", "SAMPLER", "BOOL", "ENUM",
						"IVEC2", "IVEC3", "IVEC4", "BVEC2",	"BVEC3", "BVEC4",
						"FLOAT", "VEC2", "VEC3", "VEC4", "MAT2", "MAT3", "MAT4"};


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
		case UINT:
		case SAMPLER:
		case BOOL:
		case ENUM:
		case IVEC2:
		case BVEC2:
		case BVEC3:
		case IVEC3: 
		case BVEC4:
		case IVEC4:
			return(sizeof(int) * getCardinality(p));
		case FLOAT:
		case VEC2:
		case VEC3:
		case VEC4:
		case MAT2:
		case MAT3:
		case MAT4:
			return (sizeof(float) * getCardinality(p));
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
		case ENUM:
			card = 1;
			break;
		case IVEC2:
		case BVEC2:
		case VEC2:
			card = 2;
			break;
		case IVEC3:
		case BVEC3:
		case VEC3:
			card = 3;
			break;
		case IVEC4:
		case BVEC4:
		case VEC4:
		case MAT2:
			card = 4;
			break;
		case MAT3:
			card = 9;
			break;
		case MAT4:
			card = 16;
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
#include "nau/enums.h"

#include "nau/errors.h"
#include "nau/math/number.h"
#include "nau/math/numberArray.h"
#include "nau/math/vec2.h"
#include "nau/math/vec3.h"
#include "nau/math/vec4.h"
#include "nau/math/matrix.h"

#include <assert.h>


using namespace nau;
using namespace nau::math;

std::string Enums::m_Result;

const std::vector<std::string> Enums::DataTypeToString = {
						"INT", "IVEC2", "IVEC3", "IVEC4", 
						"UINT", "UIVEC2", "UIVEC3", "UIVEC4",
						"BOOL", "BVEC2", "BVEC3", "BVEC4", 
						"FLOAT", "VEC2", "VEC3", "VEC4",
						"DOUBLE", "DVEC2", "DVEC3", "DVEC4",

						"MAT2", "MAT3", "MAT4",
						"MAT2x3", "MAT2x4", "MAT3x2", "MAT3x4", "MAT4x2", "MAT4x3",
						"DMAT2", "DMAT3", "DMAT4",
						"DMAT2x3", "DMAT2x4", "DMAT3x2", "DMAT3x4", "DMAT4x2", "DMAT4x3",
						
						"SAMPLER", "ENUM",

						"BYTE", "UBYTE", "SHORT", "USHORT", "STRING", "INTARRAY"};

const std::vector<std::string> &
Enums::GetDataTypeToString() {

	return DataTypeToString;
}


bool 
Enums::isValidType(std::string s) {

	for (int i = 0; i < COUNT_DATATYPE; i++) {

		if (s == DataTypeToString[i]) 
			return true;	
	}
	return false;

}


Enums::DataType 
Enums::getType(std::string s) {

	for (int i = 0; i < COUNT_DATATYPE; i++) {

		if (s == DataTypeToString[i]) {
		

			return (DataType)i;
		}
	}
	NAU_THROW("Invalid Data Type: %s", s.c_str());
}


//void *
//Enums::getDefaultValue(DataType p) {
//
//	int s = getSize(p);
//	void *m = malloc(s);
//	memset(m, 0, s);
//	return m;
//}


Data *
Enums::getDefaultValue(DataType p) {

	switch (p) {

	case INTARRAY:
		return new NauIntArray();
	case BYTE:
		return new NauByte();
	case UBYTE:
		return new NauUByte();
	case SHORT:
		return new NauShort();
	case USHORT:
		return new NauUShort();
	case INT:
		return new NauInt();
	case IVEC2:
		return new ivec2();
	case IVEC3:
		return new ivec3();
	case IVEC4:
		return new ivec4();
	case BOOL:
		return new NauInt();
	case BVEC2:
		return new bvec2();
	case BVEC3:
		return new bvec3();
	case BVEC4:
		return new bvec4();
	case SAMPLER:
		return new NauInt();
	case ENUM:
		return new NauInt();
	case UINT:
		return new NauUInt();
	case UIVEC2:
		return new uivec2();
	case UIVEC3:
		return new uivec3();
	case UIVEC4:
		return new uivec4();
	case FLOAT:
		return new NauFloat();
	case VEC2:
		return new vec2();
	case VEC3:
		return new vec3();
	case VEC4:
		return new vec4();
	case MAT2:
		return new mat2();
	case MAT3:
		return new mat3();
	case MAT4:
		return new mat4();
	case MAT2x3:
		return new mat2x3();
	case MAT2x4:
		return new mat2x4();
	case MAT3x2:
		return new mat3x2();
	case MAT3x4:
		return new mat3x4();
	case MAT4x2:
		return new mat4x2();
	case MAT4x3:
		return new mat4x3();
	case DOUBLE:
		return new NauDouble();
	case DVEC2:
		return new dvec2();
	case DVEC3:
		return new dvec3();
	case DVEC4:
		return new dvec4();
	case DMAT2:
		return new dmat2();
	case DMAT3:
		return new dmat3();
	case DMAT4:
		return new dmat4();
	case DMAT2x3:
		return new dmat2x3();
	case DMAT2x4:
		return new dmat2x4();
	case DMAT3x2:
		return new dmat3x2();
	case DMAT3x4:
		return new dmat3x4();
	case DMAT4x2:
		return new dmat4x2();
	case DMAT4x3:
		return new dmat4x3();
	default:
		assert(false && "Missing data type in enums - getDefaultValue");
		return 0;
	}
}

int
Enums::getSize(DataType p) {

	switch (p) {

		case BYTE:
		case UBYTE:
			return sizeof(char);
		case SHORT:
		case USHORT:
			return sizeof(short);
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
			assert(false && "Missing data type in enums");
			return 0;
	}
}


int Enums::getCardinality(DataType p) {

	int card = 1;

	switch (p) {
		case BYTE:
		case UBYTE:
		case SHORT:
		case USHORT:
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
		case UIVEC2:
		case DVEC2:
			card = 2;
			break;
		case IVEC3:
		case BVEC3:
		case VEC3:
		case UIVEC3:
		case DVEC3:
			card = 3;
			break;
		case IVEC4:
		case BVEC4:
		case VEC4:
		case UIVEC4:
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
		case INTARRAY:
			card = 1;
			break;
		default:
			assert(false && "Missing data type in enums");
			return 1;
	}
	return card;
};


bool
Enums::isCompatible(DataType nau, DataType shader) {

	if (nau == shader)
		return true;

	if ((nau == INT || nau == INTARRAY) && shader == SAMPLER)
		return true;

	if (nau == VEC4 && (shader == VEC4 || shader == VEC3 || shader == VEC2))
		return true;
	if (nau == VEC3 && (shader == VEC3 || shader == VEC2))
		return true;
	
	if (nau == IVEC4 && (shader == IVEC4 || shader == IVEC3 || shader == IVEC2))
		return true;
	if (nau == IVEC3 && (shader == IVEC3 || shader == IVEC2))
		return true;

	if (nau == BVEC4 && (shader == BVEC4 || shader == BVEC3 || shader == BVEC2))
		return true;
	if (nau == BVEC3 && (shader == BVEC3 || shader == BVEC2))
		return true;

	if (nau == UIVEC4 && (shader == UIVEC4 || shader == UIVEC3 || shader == UIVEC2))
		return true;
	if (nau == UIVEC3 && (shader == UIVEC3 || shader == UIVEC2))
		return true;

	if (nau == DVEC4 && (shader == DVEC4 || shader == DVEC3 || shader == DVEC2))
		return true;
	if (nau == DVEC3 && (shader == DVEC3 || shader == DVEC2))
		return true;

	return false;
}


bool
Enums::isBasicType(DataType t) {

	if (t == INT || t == UINT || t == BOOL || t == FLOAT || t == DOUBLE
		|| t == SAMPLER || t == ENUM || t == BYTE || t == UBYTE || t == SHORT || t == USHORT)
		return true;
	else
		return false;

}


const std::string 
Enums::valueToString(DataType p, void *v) {

	void *v1 = ((Data *)v)->getPtr();
	return pointerToString(p, v1);
}


std::string 
Enums::pointerToString(DataType p, void *v, int arraySize) {

	char *ptc = (char *)v;
	int size = getSize(p);

	std::string res;
	res = pointerToString(p, ptc);

	for (int i = 1; i < arraySize; ++i) {
		ptc += size;
		res += ", "; res += pointerToString(p, ptc);
	}
	m_Result = res;
	return m_Result;
}


std::string 
Enums::pointerToString(DataType p, void *v) {

	int intconverter;
	unsigned int uintconverter;
	bool boolconverter;

	switch (p) {
	case BYTE:
		intconverter = *((char *)v);
		m_Result = std::to_string(intconverter);
		return m_Result;
	case SHORT:
		intconverter = *((short *)v);
		m_Result = std::to_string(intconverter);
		return m_Result;
	case BOOL:
		boolconverter = *((bool *)v);
		m_Result = boolconverter == false ? "false" : "true";
		return m_Result;
	case SAMPLER:
	case ENUM:
	case INT:
		intconverter = *((int *)v);
		m_Result = std::to_string(intconverter);
		return m_Result;
	case UBYTE:
		uintconverter = *((unsigned char *)v);
		m_Result = std::to_string(uintconverter);
		return m_Result;
	case USHORT:
		uintconverter = *((unsigned short *)v);
		m_Result = std::to_string(uintconverter);
		return m_Result;
	case UINT:
		uintconverter = *((unsigned int *)v);
		m_Result = std::to_string(uintconverter);
		return m_Result;
	case FLOAT:
		m_Result = std::to_string(*((float *)v));
		return m_Result;

	case DOUBLE:
		char s[22];
		snprintf(s, 21, "%.17f", *((double *)v));
		m_Result = std::string(s);
		return m_Result;

	case IVEC2:
		m_Result = ivec2((int *)v).toString();
		return m_Result;
	case IVEC3:
		m_Result = ivec3((int *)v).toString();
		return m_Result;
	case IVEC4:
		m_Result = ivec4((int *)v).toString();
		return m_Result;
	case BVEC2:
		m_Result = bvec2((bool *)v).toString();
		return m_Result;
	case BVEC3:
		m_Result = bvec3((bool *)v).toString();
		return m_Result;
	case BVEC4:
		m_Result = bvec4((bool *)v).toString();
		return m_Result;
	case UIVEC2:
		m_Result = uivec2((unsigned int *)v).toString();
		return m_Result;
	case UIVEC3:
		m_Result = uivec3((unsigned int *)v).toString();
		return m_Result;
	case UIVEC4:
		m_Result = uivec4((unsigned int *)v).toString();
		return m_Result;
	case VEC2:
		m_Result = vec2((float *)v).toString();
		return m_Result;
	case VEC3:
		m_Result = vec3((float *)v).toString();
		return m_Result;
	case VEC4:
		m_Result = vec4((float *)v).toString();
		return m_Result;
	case DVEC2:
		m_Result = dvec2((double *)v).toString();
		return m_Result;
	case DVEC3:
		m_Result = dvec3((double *)v).toString();
		return m_Result;
	case DVEC4:
		m_Result = dvec4((double *)v).toString();
		return m_Result;

	case MAT2:
		m_Result = mat2((float *)v).toString();
		return m_Result;
	case MAT3:
		m_Result = mat3((float *)v).toString();
		return m_Result;
	case MAT4:
		m_Result = mat4((float *)v).toString();
		return m_Result;
	case DMAT2:
		m_Result = dmat2((double *)v).toString();
		return m_Result;
	case DMAT3:
		m_Result = dmat3((double *)v).toString();
		return m_Result;
	case DMAT4:
		m_Result = dmat4((double *)v).toString();
		return m_Result;

	case MAT2x3:
		m_Result = mat2x3((float *)v).toString();
		return m_Result;
	case MAT2x4:
		m_Result = mat2x4((float *)v).toString();
		return m_Result;
	case MAT3x2:
		m_Result = mat3x2((float *)v).toString();
		return m_Result;
	case MAT3x4:
		m_Result = mat3x4((float *)v).toString();
		return m_Result;
	case MAT4x2:
		m_Result = mat4x2((float *)v).toString();
		return m_Result;
	case MAT4x3:
		m_Result = mat4x3((float *)v).toString();
		return m_Result;
	case DMAT2x3:
		m_Result = dmat2x3((double *)v).toString();
		return m_Result;
	case DMAT2x4:
		m_Result = dmat2x4((double *)v).toString();
		return m_Result;
	case DMAT3x2:
		m_Result = dmat3x2((double *)v).toString();
		return m_Result;
	case DMAT3x4:
		m_Result = dmat3x4((double *)v).toString();
		return m_Result;
	case DMAT4x2:
		m_Result = dmat4x2((double *)v).toString();
		return m_Result;
	case DMAT4x3:
		m_Result = dmat4x3((double *)v).toString();
		return m_Result;
	default:
		assert(false && "Missing data type in enums");
		return m_Result;
	}
}


// Outputs similar to what is expected in std140 from OpenGL
std::string 
Enums::pointerToStringAligned(DataType p, void *v) {

	m_Result = "[ ";
	switch (p) {
	case MAT3: {
		float *m = ((float *)v);
		for (unsigned int i = 0; i < 3; ++i) {
			for (unsigned int j = 0; j < 3; ++j) {
				m_Result += std::to_string(m[i * 4 + j]) + ", ";
			}
			m_Result += std::to_string(m[i * 4 + 3]) + " \n";
		}
		m_Result += " ]";
		return m_Result;
	}
	case DMAT3:
		m_Result = ((dmat3 *)v)->toString();
		return m_Result;

	case MAT2x3:
		m_Result = ((mat2x3 *)v)->toString();
		return m_Result;

	case DMAT2x3:
		m_Result = ((dmat2x3 *)v)->toString();
		return m_Result;
	case MAT4x3:
		m_Result = ((mat4x3 *)v)->toString();
		return m_Result;

	case DMAT4x3:
		m_Result = ((dmat4x3 *)v)->toString();
		return m_Result;
	default:
		return pointerToString(p, v);
	}
}


Enums::DataType
Enums::getBasicType(Enums::DataType p) {

	switch (p) {
	case BYTE:
	case SHORT:
	case SAMPLER:
	case UBYTE:
	case USHORT:
		return p;

	case ENUM:
	case INT:
	case IVEC2:
	case IVEC3:
	case IVEC4:
		return INT;

	case BOOL:
	case BVEC2:
	case BVEC3:
	case BVEC4:
		return BOOL;

	case UINT:
	case UIVEC2:
	case UIVEC3:
	case UIVEC4:
		return UINT;

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
		return FLOAT;

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
		return DOUBLE;
	default:
		assert(false && "Missing data type in enums");
		return FLOAT;
	}


}


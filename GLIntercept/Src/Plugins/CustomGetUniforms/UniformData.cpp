#include "UniformData.h"
#include "../../MainLib/InterceptPluginInterface.h"
#include <sstream>
#include <string>
#include <iostream>
#define CINT 0;
#define CUINT 1;
#define CFLOAT 2;
#define CBOOL 3;
#define OTHER -1;

unsigned int UniformData::getLength(int number)
{
    unsigned int length = 0;
	int pow = 1;

    if (number < 0) //When it is beneat zero is has a - before the number, so that makes the length larger
        length++;

    number = abs(number); //this is because you need to work with the positive version of it.

    for (unsigned char c = 1; c < 10; c++) //It only goes to 10 since 2^32 is in text 10 symbols long
    {
		pow *= 10;
		if (number < pow) //when you reach a point where it is smaller than pow(10, c) you know the length.
			length += c;
            //Like, when it is 987 it is smaller than 1000 but not smaller than 100. So you can see that the zeros of the result of pow(10, c) represents the length of the number. c is the number of zeros so c is added to the length.
    }

    return length;
}

UniformData::UniformData(int p, int l){
	program = p;
	location = l;

	outputString = 0;
	isValueSet = false;
}

UniformData::UniformData(int p, int l, string n, void* d, unsigned int t){	
	program = p;
	location = l;
	name = n;
	data = d;
	datatype = t;

	outputString = 0;
	isValueSet = false;
}

void UniformData::Update(void* d){
	if (isValueSet){
		free(data);
	}
	data = d;
	isValueSet = true;
}

void UniformData::SetType(unsigned int t){
	datatype_enum = t;

	//DataSize;
	switch(t){
		case GL_BOOL_VEC4:
		case GL_INT_VEC4:
		case GL_FLOAT_VEC4:
			datasize[0]=1;
			datasize[1]=4;
			break;
		case GL_FLOAT_MAT3:
			datasize[0]=3;
			datasize[1]=3;
			break;
		case GL_FLOAT_MAT4:
			datasize[0]=4;
			datasize[1]=4;
			break;
		case GL_SAMPLER:
		case GL_SAMPLER_2D:
			datasize[0]=1;
			datasize[1]=1;
			break;
		default:
			datasize[0]=1;
			datasize[1]=1;
	}

	//types
	switch(t){
		case GL_BOOL_VEC4:
			datatype = CBOOL;
			break;
		case GL_INT_VEC4:
			datatype = CINT;
			break;
		case GL_FLOAT:
		case GL_FLOAT_VEC4:
		case GL_FLOAT_MAT3:
		case GL_FLOAT_MAT4:
			datatype = CFLOAT;
			break;
		default:
			datatype=OTHER
	}
}

void UniformData::SetName(string new_name){
	name = string(new_name);
}

void UniformData::SetName(char *new_name){
	name = string(strdup(new_name));
}

char *datatypeName(unsigned int type){
	switch(type){
		case 0:
			return "integer";
		case 2:
			return "float";
		case 3:
			return "boolean";
		default:
			return "other";
	}
}

char *UniformData::InfoString(){
	ostringstream os;

	if (outputString){
		free(outputString);
	}

	os << name << "(" << program << ", " << location << ") type: " << std::hex << datatype_enum << " : " << datatypeName(datatype);
	

	outputString = strdup(os.str().c_str());
	return outputString;
}

char *UniformData::ValueString(){
	ostringstream os;

	if (outputString){
		free(outputString);
	}
	if (isValueSet){

		
		if ((datasize[0] == datasize[1]) && (datasize[0] == 1)){ //Single Value
			switch(datatype){
				case 0:
					os << name << "(" << program << ", " << location << ") = " << ((int*) data)[0];
					break;
				case 1:
					os << name << "(" << program << ", " << location << ") = " << ((unsigned int*) data)[0];
					break;
				case 2:
					os << name << "(" << program << ", " << location << ") = " << ((float*) data)[0];
					break;
				case 3:
					os << name << "(" << program << ", " << location << ") = " << ((bool*) data)[0];
					break;
				default:
					os << name << "(" << program << ", " << location << ") = " << std::hex << ((unsigned int*) data)[0];
					break;
			}
		}
		else if (datasize[0] == 1){ //Array
			switch(datatype){
				case 0:
					os << name << "(" << program << ", " << location << ") = ";
					printRow(&os, 0);
					break;
				case 1:
					os << name << "(" << program << ", " << location << ") = ";
					printRow(&os, 0);
					break;
				case 2:
					os << name << "(" << program << ", " << location << ") = ";
					printRow(&os, 0);
					break;
				case 3:
					os << name << "(" << program << ", " << location << ") = ";
					printRow(&os, 0);
					break;
				default:
					os << name << "(" << program << ", " << location << ") = ";
					printRow(&os, 0);
					break;
			}
		}
		else  if (datasize[0] != 0){ //Matrix
			//int blanklength = 7 + name.length() + getLength(program) + getLength(location);
			os << name << "(" << program << ", " << location << ") = (about to be implemented)";
		}
		else{
			os << name << "(" << program << ", " << location << ") = (not implemented yet / unknown)";
		}
	}
	else{
		os << name << "(" << program << ", " << location << ") = (null)";
	}

	outputString = strdup(os.str().c_str());
	return outputString;
}


void UniformData::printRow(ostringstream *os, unsigned int row){
	int rowoffset = row * datasize[1];
	
	switch(datatype){
		case 0:
			*os << "[" << ((int*) data)[0 + rowoffset];
			for (unsigned int i = 1; i<datasize[1]; i++)
				 *os << ", " << ((int*) data)[i + rowoffset];
			*os << "]";
			break;
		case 1:
			*os << "[" << ((unsigned int*) data)[0 + rowoffset];
			for (unsigned int i = 1; i<datasize[1]; i++)
				 *os << ", " << ((unsigned int*) data)[i + rowoffset];
			*os << "]";
			break;
		case 2:
			*os << "[" << ((float*) data)[0 + rowoffset];
			for (unsigned int i = 1; i<datasize[1]; i++)
				 *os << ", " << ((float*) data)[i + rowoffset];
			*os << "]";
			break;
		case 3:
			*os << "[" << ((bool*) data)[0 + rowoffset];
			for (unsigned int i = 1; i<datasize[1]; i++)
				 *os << ", " << ((bool*) data)[i + rowoffset];
			*os << "]";
			break;
		default:
			*os << "[" << std::hex << ((unsigned int*) data)[0 + rowoffset];
			for (unsigned int i = 1; i<datasize[1]; i++)
				 *os << ", " << ((float*) data)[i + rowoffset];
			*os << "]";
			break;
	}
}
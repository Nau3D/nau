#include "UniformData.h"
#include "../../MainLib/InterceptPluginInterface.h"
#include <sstream>
#include <string>
#include <iostream>

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

	datatype = OTHER;

	outputString = 0;
	isValueSet = false;
}

//UniformData::UniformData(int p, int l, string n, void* d, unsigned int t){	
//	program = p;
//	location = l;
//	name = n;
//	data = d;
//	datatype = t;
//
//	outputString = 0;
//	isValueSet = false;
//}

void UniformData::Update(void* d){
	if (isValueSet){
		free(data);
	}
	data = d;
	isValueSet = true;
}

void UniformData::Update(const char *charptr, FunctionArgs &accessArgs){
	unsigned int arraysize = 0;
	void *pointer;
	GLint sizei;
	
	if (isValueSet){
		free(data);
		data=0;
		isValueSet=false;
	}

	switch(charptr[0]){
		case '1':
			arraysize = 1;
			break;
		case '2':
			arraysize = 2;
			break;
		case '3':
			arraysize = 3;
			break;
		case '4':
			arraysize = 4;
			break;
	}

	//only if it's not matrix which start as glProgramUniformMatrices
	if (arraysize){

		// types: "f","i","ui","fv","iv","uiv"
		switch(charptr[1]){
			case 'f':
				data = malloc(sizeof(float) * arraysize);
				if (charptr[2]){ //fv
		            accessArgs.Get(sizei);
		            accessArgs.Get(pointer);
					memcpy(data,pointer,sizeof(float) * arraysize);
					SetTypeManual(CFLOAT, arraysize, 1);
				}
				else{ //f
					GLfloat value;
					for(unsigned int i=0; i<arraysize; i++){
						float *tmp = (float *) data;
						accessArgs.Get(value);
						tmp[i] = value;
					}
					SetTypeManual(CFLOAT, 1, 1);
				}
				break;
			case 'i':
				data = malloc(sizeof(int) * arraysize);
				if (charptr[2]){ //iv
		            accessArgs.Get(sizei);
		            accessArgs.Get(pointer);
					memcpy(data,pointer,sizeof(int) * arraysize);
					SetTypeManual(CINT, arraysize, 1);
				}
				else{ //i
					GLint value;
					for(unsigned int i=0; i<arraysize; i++){
						int *tmp = (int *) data;
						accessArgs.Get(value);
						tmp[i] = value;
					}
					SetTypeManual(CINT, 1, 1);
				}
				break;
			case 'u':
				data = malloc(sizeof(unsigned int) * arraysize);
				if (charptr[3]){ //uiv
		            accessArgs.Get(sizei);
		            accessArgs.Get(pointer);
					memcpy(data,pointer,sizeof(unsigned int) * arraysize);
					SetTypeManual(CUINT, arraysize, 1);
				}
				else{ //ui
					GLuint value;
					for(unsigned int i=0; i<arraysize; i++){
						unsigned int *tmp = (unsigned int *) data;
						accessArgs.Get(value);
						tmp[i] = value;
					}
					SetTypeManual(CUINT, 1, 1);
				}
				break;
		}

	}
	else{
		charptr += 8;
		unsigned int rows, columns;
		switch(charptr[0]){
			case '2':
				rows = 2;
				break;
			case '3':
				rows = 3;
				break;
			case '4':
				rows = 4;
				break;
		}
		if (charptr[1] == 'x'){
			switch(charptr[2]){
				case '2':
					columns = 2;
					break;
				case '3':
					columns = 3;
					break;
				case '4':
					columns = 4;
					break;
			}
		}
		arraysize = rows * columns;
		data = malloc(sizeof(float) * arraysize);
		accessArgs.Get(sizei);
		accessArgs.Get(pointer);
		memcpy(data,pointer,sizeof(float) * arraysize);
		SetTypeManual(CFLOAT, columns, rows);
	}
	isValueSet = true;
	
}

void UniformData::SetTypeManual(unsigned int t, unsigned int c, unsigned int r){
	datatype = t;
	datasize[0]=r;
	datasize[1]=c;
}

void UniformData::SetType(unsigned int t){
	datatype_enum = t;
	return;
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
			datatype = OTHER;
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

bool UniformData::IsValueSet(){
	return isValueSet;
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

char *UniformData::UniformString(){
	ostringstream os;

	if (outputString){
		free(outputString);
	}

	os << name << "(" << program << ", " << location << ") type: " << std::hex << datatype_enum << " : " << datatypeName(datatype);
	
	if (isValueSet){

		
		if ((datasize[0] == datasize[1]) && (datasize[0] == 1)){ //Single Value
			switch(datatype){
				case CINT:
					os << " = " << ((int*) data)[0];
					break;
				case CUINT:
					os << " = " << ((unsigned int*) data)[0];
					break;
				case CFLOAT:
					os << " = " << ((float*) data)[0];
					break;
				case CBOOL:
					os << " = " << ((bool*) data)[0];
					break;
				default:
					os << " = " << std::hex << ((unsigned int*) data)[0];
					break;
			}
		}
		else if (datasize[0] == 1){ //Array
			os << " = ";
			printRow(&os, 0);
		}
		else if (datasize[0] != 0){ //Matrix
			os << " = ";
			int blanklength = os.str().length();
			char *blank = (char*) malloc (sizeof(char) * (blanklength + 1));
			for (int i = 0; i < blanklength; i++){
				blank[i] = ' ';
			}
			blank[blanklength] = 0;
			printRow(&os, 0);
			for (int i = 1; i < datasize[0]; i++)
			{
				os << '\n' << blank;
				printRow(&os, i);
			}
		}
		else{
			os << " = (not implemented yet / unknown)";
		}
	}
	else{
		os << " = (null / not set)";
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
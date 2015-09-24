/** ----------------------------------------------------------
 * \class VSGLInfoLib
 *
 * Lighthouse3D
 *
 * VSGLInfoLib - Very Simple GL Information Library
 *
 *	
 * \version 0.1.0
 *  - Initial Release
 *
 * This class provides information about GL stuff
 *
 * Full documentation at 
 * http://www.lighthouse3d.com/very-simple-libs
 *
 ---------------------------------------------------------------*/

#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <ostream>
#include <fstream>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <utility>

//#include "NauGlBufferInfo.h"




/// checks if an extension is supported
bool isExtensionSupported(std::string extName);

std::vector<unsigned int> &getProgramNames();

//Additional Custom Functions
std::string getDatatypeString(int datatype);

//untested because nau does not have blocks
void getBlockNames(unsigned int program, std::vector<std::string> &namelist);

void getUniformNames(unsigned int program, std::vector<std::string> &namelist);

void getBlockData(unsigned int program, std::string blockName, int &datasize, int &blockbindingpoint, int &bufferbindingpoint, std::vector<std::string> &uniformnamelist);
void getUniformData(unsigned int program, std::string uniformName, std::string &uniformType, int &uniformSize, int &uniformArrayStride, std::vector<std::string> &values);
void getUniformData(unsigned int program, std::string blockName, std::string uniformName, std::string &uniformType, int &uniformSize, int &uniformArrayStride, std::vector<std::string> &values);

void getUniformValuef(float *f, int rows, int columns, std::vector<std::string> &values);
void getUniformValuei(int *f, int rows, int columns, std::vector<std::string> &values);
void getUniformValueui(unsigned int *f, int rows, int columns, std::vector<std::string> &values);
void getUniformValued(double *f, int rows, int columns, std::vector<std::string> &values);

void getProgramInfoData(unsigned int program, std::vector<std::pair<std::string, char>> &shadersInfo, std::vector<std::string> &stdInfo,  std::vector<std::string> &geomInfo,  std::vector<std::string> &tessInfo);
void getAttributesData(unsigned int program, std::vector<std::pair<std::string, std::pair<int,std::string>>> &attributeList);

//void getCurrentVAOInfoData(std::vector<std::pair<std::pair<int, int>, std::vector<int>>> &vaoInfoData);

//std::vector<int> getCurrentBufferNames();
//bool getBufferInfoFromMap(int buffer, NauGlBufferInfo &bufferInfo);
//std::map<int, NauGlBufferInfo> *getBufferInfoMap();

//int openBufferMapPointers(int buffer, int offsetNumber, int sizePerOffset, int size, std::vector<int> sizes, std::vector<void*> &pointers);
//void closeBufferMapPointers(int prevBuffer);
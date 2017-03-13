#include "nau/loader/vtkLoader.h"


#include "nau/slogger.h"
#include "nau/system/file.h"

#include <algorithm>
#include <ctime>
#include <iostream>
#include <fstream>
#include <string>

using namespace nau::loader;
using namespace nau::system;


VTKTextureLoader::VTKTextureLoader(const std::string &filename): ITextureLoader(filename),
	m_Width(1), m_Height(1), m_Depth(1), m_NumPoints(0), m_Data(NULL) {

}


VTKTextureLoader::~VTKTextureLoader(void) {

	freeImage();
}


int 
VTKTextureLoader::loadImage (bool notUsed) {

	std::string file = m_Filename;
	File::FixSlashes(file);

	std::ifstream vfs(file, std::ios::binary);
	if (!vfs.is_open())
		return 0;

	std::string line;
	std::getline(vfs, line);

	// check header
	const char *s = line.c_str() + 2;
	if (line[0] != '#' || (strncmp(s, "VTK", 3) && strncmp(s, "vtk", 3)))
		return 0;

	// second line is just the header
	std::getline(vfs, line);

	//third line: ASCII or BINARY
	std::getline(vfs, line);
	if (!strcmp(line.c_str(), "ASCII"))
		m_Mode = ASCII;
	else if(!strcmp(line.c_str(), "BINARY"))
		m_Mode = BINARY;
	else
		return 0;

	// fourth line: type must be structured points
	std::getline(vfs, line);
	if (!strstr(line.c_str(), "STRUCTURED_POINTS"))
		return 0;

	// fifth line: dimensions
	std::getline(vfs, line);
	if (3 != sscanf(line.c_str(), "DIMENSIONS %d %d %d", &m_Width, &m_Height, &m_Depth))
		return 0;

	// sixth line: ORIGIN
	std::getline(vfs, line);
	//seventh line: SPACING
	std::getline(vfs, line);

	//eigth line: number of points
	std::getline(vfs, line);
	if (1 != sscanf(line.c_str(), "POINT_DATA %d", &m_NumPoints))
		return 0;

	//nineth line: data type
	std::getline(vfs, line);
	char dataType[32], dummy[32];
	if (2 != sscanf(line.c_str(), "SCALARS %s %s", dummy, dataType))
		return 0;

	if (!strcmp(dataType, "unsigned_char"))
		m_DataType = Enums::BYTE;
	else if (!strcmp(dataType, "unsigned_short"))
		m_DataType = Enums::USHORT;
	else if (!strcmp(dataType, "unsigned int"))
		m_DataType = Enums::UINT;
	else if (!strcmp(dataType, "short"))
		m_DataType = Enums::SHORT;
	else if (!strcmp(dataType, "float"))
		m_DataType = Enums::FLOAT;
	else
		return 0;

	// line 10: LOOKUP_TABLE
	std::getline(vfs, line);

	// read points
	if (m_Mode == ASCII) {
		// TO DO
	}
	else { // read binary
		int count = m_Width * m_Height * m_Depth;
		size_t siz = count * Enums::getSize(m_DataType);
		m_Data = (unsigned char *)malloc(siz);
		vfs.read(reinterpret_cast<char *>(m_Data), siz);
	}	

	return 1;
}


unsigned char* 
VTKTextureLoader::getData (void) {

	return m_Data;
}


int 
VTKTextureLoader::getWidth (void) {

	return m_Width;
}


int 
VTKTextureLoader::getHeight (void) {

	return m_Height;
}


int
VTKTextureLoader::getDepth(void) {

	return m_Depth;
}


std::string
VTKTextureLoader::getFormat (void) {

	return "R8";
}

			
std::string
VTKTextureLoader::getType (void) {

	return Enums::DataTypeToString[m_DataType];
}


void
VTKTextureLoader::freeImage(void) {

	if (m_Data != NULL) {
		free (m_Data);
		m_Data = NULL;
	}
}


void 
VTKTextureLoader::convertToFloatLuminance() {

}


void 
VTKTextureLoader::convertToRGBA() {

}


void 
VTKTextureLoader::save(ITexImage *ti, std::string filename) {

}


void 
VTKTextureLoader::save(int width, int height, unsigned char *data, std::string filename) {

}

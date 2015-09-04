#include "nau/loader/devilTextureLoader.h"


#include "nau/slogger.h"
#include "nau/system/file.h"


#include <ctime>

using namespace nau::loader;
using namespace nau::system;

bool DevILTextureLoader::inited = false;

DevILTextureLoader::DevILTextureLoader(void)
{
	if (!DevILTextureLoader::inited) {
		ilInit();
		DevILTextureLoader::inited = true;
	}
	/*** Check configuration for type of render ***/
	ilGenImages (1, &m_IlId);
}

DevILTextureLoader::~DevILTextureLoader(void)
{
	ilDeleteImage(m_IlId);
}

int 
DevILTextureLoader::loadImage (std::string file)
{
	ilBindImage(m_IlId);
	ilEnable(IL_ORIGIN_SET);
	ilOriginFunc(IL_ORIGIN_LOWER_LEFT); 
	int success = ilLoadImage((ILstring)(file.c_str()));
	if (success)
		ilConvertImage(IL_RGBA,IL_UNSIGNED_BYTE);

	return(success); 
	
}

unsigned char* 
DevILTextureLoader::getData (void)
{
	if (m_IlId > 0) {
		ilBindImage(m_IlId);
		return(ilGetData());
	}
	else {
		return (0);
	}
}

int 
DevILTextureLoader::getWidth (void)
{
	if (m_IlId > 0) {
		ilBindImage(m_IlId);
		return(ilGetInteger(IL_IMAGE_WIDTH));
	}
	else {
		return (0);
	}
}

int 
DevILTextureLoader::getHeight (void)
{
	if (m_IlId > 0) {
		ilBindImage(m_IlId);
		return(ilGetInteger(IL_IMAGE_HEIGHT));
	}
	else {
		return (0);
	}
}

std::string 
DevILTextureLoader::getFormat (void)
{
	/***MARK***/
	return "RGBA";
}
			
std::string
DevILTextureLoader::getType (void)
{
	/***MARK***/
	return "UNSIGNED_BYTE";
}

void 
DevILTextureLoader::freeImage (void)
{
	/***MARK***/
}


void
DevILTextureLoader::save(ITexImage *ti, std::string filename) {

	void *data = ti->getData();
	char res;
	int w = ti->getWidth();
	int h = ti->getHeight();
	int n = ti->getNumComponents();
	std::string type = ti->getType();

	ILuint ilFormat, ilType;
	if (n == 1 || n == 3 || n == 4) {

		switch (n) {
		case 1: ilFormat = IL_LUMINANCE; break;
		case 3: ilFormat = IL_RGB; break;
		case 4: ilFormat = IL_RGBA; break;
		}

		ilType = convertType(type);

		if (!DevILTextureLoader::inited) {
			ilInit();
			DevILTextureLoader::inited = true;
		}

		ILuint image;
		ilOriginFunc(IL_ORIGIN_LOWER_LEFT);
		ilGenImages(1, &image);
		ilBindImage(image);
		ilTexImage(w, h, 1, n, ilFormat, ilType, data);
		ilEnable(IL_FILE_OVERWRITE);

		std::string ext = File::GetExtension(filename);
		if (ext == "hdr") {
			ilConvertImage(ilFormat, IL_FLOAT);
			res = ilSave(IL_HDR, (ILstring)filename.c_str());
		}
		else {
			if (ilFormat == IL_LUMINANCE && (ilType == IL_FLOAT || ilType == IL_INT))
				ilConvertImage(ilFormat, IL_UNSIGNED_SHORT);
			else
				ilConvertImage(ilFormat, IL_UNSIGNED_BYTE);
			res = ilSave(IL_PNG, (ILstring)filename.c_str());
		}


		if (res == 0)
			SLOG("Can't save image %s - format not supported", filename.c_str());
		ilDeleteImage(image);
	}
}





ILuint 
DevILTextureLoader::convertType(std::string type) {

	ILuint ilType = IL_TYPE_UNKNOWN;

	if (type == "FLOAT")
		ilType = IL_FLOAT;
	else if (type == "UNSIGNED_BYTE" || type == "UNSIGNED_INT_8_8_8_8_REV")
		ilType = IL_UNSIGNED_BYTE;
	else if (type == "UNSIGNED_SHORT")
		ilType = IL_UNSIGNED_SHORT;
	else if (type == "UNSIGNED_INT")
		ilType = IL_UNSIGNED_INT;
	else if (type == "SHORT")
		ilType = IL_SHORT;
	else if (type == "BYTE")
		ilType = IL_BYTE;
	else if (type == "INT")
		ilType = IL_INT;

	return ilType;
}


void
DevILTextureLoader::save(int width, int height, char *data, std::string filename) {


	ILuint image;
	ilInit();
	ilOriginFunc(IL_ORIGIN_LOWER_LEFT); 
	ilGenImages(1,&image);
	ilBindImage(image);
	ilTexImage(width, height, 1, 4, IL_RGBA, IL_UNSIGNED_BYTE, data);
	ilEnable(IL_FILE_OVERWRITE);
	ilSave(IL_PNG, (ILstring)filename.c_str());
	ilDeleteImage(image);
}
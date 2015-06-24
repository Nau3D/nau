#include "fileutil.h"

using namespace nau::system;


bool 
FileUtil::exists(const std::string &fn) {

	bool b;

	FILE *fp = fopen(fn.c_str(),"r");
	if (fp) {
		fclose(fp);
		b = true;
	}
	else {
		b = false;
	}
	return (b);
	
}

std::string
FileUtil::GetExtension(const std::string &fn) {

	size_t found = fn.find_last_of(".");
	std::string res;

	if (found == fn.npos) 
		res = "";
	else
		res = fn.substr(found+1);

	return res;
}

std::string 
FileUtil::GetName(const std::string &fn) {

	size_t found = fn.find_last_of("/\\");

	if (found == fn.npos) 
		return(fn.substr(0,fn.size())); // clone string
	else
		return(fn.substr(found+1));
}

std::string 
FileUtil::GetPath(const std::string &fn) {

	size_t found = fn.find_last_of("/\\");

	if (found == fn.npos) 
		return(fn.substr(0,fn.size()-1)); // clone string
	else
		return(fn.substr(0,found));
}


std::string 
FileUtil::GetFullPath(const std::string &currentDir, const std::string &relFN) {

	std::string res,resAux;

	if (IsRelative(relFN)) {
		resAux = currentDir + SLASH + relFN;
		res = CleanFullPath(resAux);
	}
	else
		res = CleanFullPath(relFN);

	return res;
}


std::string 
FileUtil::CleanFullPath(const std::string &fn) {

	std::string res,resAux;
	size_t pos,pos1;

	resAux = fn;
//	if (IsRelative(resAux))
//		return(resAux);
	
	pos = resAux.find("\\.\\",0);
	while (pos != resAux.npos) {
		std::string aux1 = resAux.substr(0,pos);
		std::string aux2 = resAux.substr(pos+2,resAux.size()-1);
		resAux = resAux.substr(0,pos) + resAux.substr(pos+2,resAux.size()-1);

		pos = resAux.find("\\.\\",pos+1);
	}

	pos = resAux.find("\\..\\",0);
	while (pos != resAux.npos) {
		std::string aux1 = resAux.substr(0,pos);
		pos1 = aux1.find_last_of(SLASH);
		std::string aux2 = resAux.substr(pos+3,resAux.size()-1);
		resAux = aux1.substr(0,pos1) + resAux.substr(pos+3,resAux.size()-1);

		pos = resAux.find("\\..\\",0);
	}

	pos = resAux.find("\\../",0);
	while (pos != resAux.npos) {
		std::string aux1 = resAux.substr(0,pos);
		pos1 = aux1.find_last_of(SLASH);
		std::string aux2 = resAux.substr(pos+3,resAux.size()-1);
		resAux = aux1.substr(0,pos1) + resAux.substr(pos+3,resAux.size()-1);

		pos = resAux.find("\\..\\",0);
	}

	pos = resAux.find("/../",0);
	while (pos != resAux.npos) {
		std::string aux1 = resAux.substr(0,pos);
		pos1 = aux1.find_last_of("/\\");
		std::string aux2 = resAux.substr(pos+3,resAux.size()-1);
		resAux = aux1.substr(0,pos1) + resAux.substr(pos+3,resAux.size()-1);

		pos = resAux.find("/../",0);
	}

	return(resAux);
}


std::string 
FileUtil::validate(std::string s1) {

	std::string res = s1;
	std::string s = s1;
	for (unsigned int i = 0; i < s1.length(); ++i) {
		if (s[i] == ':' || s[i] == '/' || s[i] == '\\' || s[i] == '*' || s[i] == '?' || s[i] == '<' ||
			s[i] == '>' || s[i] == '|')
			res[i] = '_';
	}
	return res;
}


bool
FileUtil::IsRelative(const std::string &fn) {

	if (fn[0] == '/' || fn[1] == ':')
		return(false);
	else
		return(true);
}

std::string  
FileUtil::GetRelativePathTo(const std::string &currentDir, const std::string &absFN) {

	// declarations - put here so this should work in a C compiler
	int afMarker = 0, rfMarker = 0;
	int cdLen = 0, afLen = 0;
	int i = 0;
	int levels = 0;
	static char relativeFilename[MAX_FILENAME_LEN+1];
	char *currentDirectory = (char *)currentDir.c_str();
	char *absoluteFilename = (char *)absFN.c_str();

	cdLen = strlen(currentDirectory);
	afLen = strlen(absoluteFilename);

	// make sure the names are not too long or too short
	if(cdLen > MAX_FILENAME_LEN || cdLen < ABSOLUTE_NAME_START+1 ||
		afLen > MAX_FILENAME_LEN || afLen < ABSOLUTE_NAME_START+1)
	{
		return NULL;
	}

	// Handle DOS names that are on different drives:
	if(currentDirectory[0] != absoluteFilename[0])
	{
		// not on the same drive, so only absolute filename will do
		strcpy(relativeFilename, absoluteFilename);
		return relativeFilename;
	}

	// they are on the same drive, find out how much of the current directory
	// is in the absolute filename
	i = ABSOLUTE_NAME_START;
	while(i < afLen && i < cdLen && currentDirectory[i] == absoluteFilename[i])
	{
		i++;
	}

	if(i == cdLen && (absoluteFilename[i] == SLASH || absoluteFilename[i-1] == SLASH))
	{
		// the whole current directory name is in the file name,
		// so we just trim off the current directory name to get the
		// current file name.
		if(absoluteFilename[i] == SLASH)
		{
			// a directory name might have a trailing slash but a relative
			// file name should not have a leading one...
			i++;
		}

		strcpy(relativeFilename, &absoluteFilename[i]);
		return relativeFilename;
	}


	// The file is not in a child directory of the current directory, so we
	// need to step back the appropriate number of parent directories by
	// using "..\"s.  First find out how many levels deeper we are than the
	// common directory
	afMarker = i;
	levels = 1;

	// count the number of directory levels we have to go up to get to the
	// common directory
	while(i < cdLen)
	{
		i++;
		if(currentDirectory[i] == SLASH)
		{
			// make sure it's not a trailing slash
			i++;
			if(currentDirectory[i] != '\0')
			{
				levels++;
			}
		}
	}

	// move the absolute filename marker back to the start of the directory name
	// that it has stopped in.
	while(afMarker > 0 && absoluteFilename[afMarker-1] != SLASH)
	{
		afMarker--;
	}

	// check that the result will not be too long
	if(levels * 3 + afLen - afMarker > MAX_FILENAME_LEN)
	{
		return NULL;
	}

	// add the appropriate number of "..\"s.
	rfMarker = 0;
	for(i = 0; i < levels; i++)
	{
		relativeFilename[rfMarker++] = '.';
		relativeFilename[rfMarker++] = '.';
		relativeFilename[rfMarker++] = SLASH;
	}

	// copy the rest of the filename into the result string
	strcpy(&relativeFilename[rfMarker], &absoluteFilename[afMarker]);

//	std::string cleanFilename = CleanFullPath(relativeFilename);
	return relativeFilename;
}


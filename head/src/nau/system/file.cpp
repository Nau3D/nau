#include "nau/system/file.h"

#include "nau/config.h"

#include <sstream>
#include <direct.h>

#ifdef WIN32
#	define MKDIR _mkdir
#else
#	define MKDIR mkdir
#endif


using namespace nau::system;


#ifdef NAU_PLATFORM_WIN32

#include <direct.h>
#include <dirent.h>
const std::string File::PATH_SEPARATOR("\\");
#define ABSOLUTE_NAME_START 3
#else

#include <unistd.h>

const std::string File::PATH_SEPARATOR("/")
#define ABSOLUTE_NAME_START 1
#endif

#define PROTOCOL "file://"


File::File(std::string filepath, bool native) :
	m_vPath(),
	m_FileName(filepath),
	m_FileExtension(),
	m_Drive(""),
	m_IsRelative (true),
	m_IsLeaf (false),
	m_HasExtension (false),
	m_IsNative (native) {

	if (false == m_IsNative) {
		construct (filepath);
	}
}

File::~File(void) {

}


std::string 
File::TextRead (const std::string &fn) {

	FILE *fp;
	char *content = NULL;
	std::string s;

	size_t count=0;

	fp = fopen(fn.c_str(),"rt");

	if (fp != NULL) {
		  
		fseek(fp, 0, SEEK_END);
		count = ftell(fp);
		rewind(fp);

		if (count > 0) {
			content = (char *)malloc(sizeof(char) * (count+1));
			count = fread(content,sizeof(char),count,fp);
			content[count] = '\0';
		}
		fclose(fp);
	}
	if (content == NULL)
		s = "";
	else 
		s = content;
	return s;
}


int 
File::TextWrite(const std::string &fn, const std::string &s) {

	FILE *fp;
	int status = 0;

	fp = fopen(fn.c_str(),"w");

	if (fp != NULL) {
				
		if (fwrite(s.c_str(),sizeof(char),s.size(),fp) == s.size())
			status = 1;
		fclose(fp);
	}
	return(status);
}


std::string
File::GetCurrentFolder() {

	std::string s = "";
	char folder[255];
#ifdef WIN32
	if (_getcwd(folder, 255))
#else
	if (getcwd(folder, 255))
#endif
		s = std::string(folder);

	return s;
}


std::string
File::GetAppFolder() {

	std::string folder;
	char app[255];

#if NAU_PLATFORM_WIN32
	GetModuleFileNameA(NULL, app, 255);
#endif
	folder = GetPath(app);
	return folder;
}


bool 
File::CreateDir(std::string path) {

	int res = MKDIR(path.c_str());

	return (res == 0);
}


std::string
File::BuildFullFileName(std::string path, std::string filename) {

	return path + PATH_SEPARATOR + filename;
}


void 
File::RecurseDirectory(std::string path, std::vector<std::string> *res) {

	DIR *dir;
	struct dirent *ent;
	
	if((dir = opendir (path.c_str())) != NULL) {

		while ((ent = readdir(dir)) != NULL) {
		
			if (std::string(ent->d_name) == "." || std::string(ent->d_name) == "..") {
		
			}
			else if (ent->d_type == DT_DIR) {
				std::string nextDir = std::string(ent -> d_name);
				//nextDir += "\\";
				RecurseDirectory(path + "\\" + nextDir, res);
			}
			else {
				res->push_back(File::BuildFullFileName(path, ent -> d_name));
			}
		}
	}
	
	closedir (dir);
}


bool 
File::Exists(const std::string &fn) {

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
File::GetExtension(const std::string &fn) {

	size_t found = fn.find_last_of(".");
	std::string res;

	if (found == fn.npos) 
		res = "";
	else
		res = fn.substr(found+1);

	return res;
}

std::string 
File::GetName(const std::string &fn) {

	size_t found = fn.find_last_of("/\\");

	if (found == fn.npos) 
		return(fn.substr(0,fn.size())); // clone string
	else
		return(fn.substr(found+1));
}

std::string 
File::GetPath(const std::string &fn) {

	size_t found = fn.find_last_of("/\\");

	if (found == fn.npos) 
		return(fn.substr(0,fn.size()-1)); // clone string
	else
		return(fn.substr(0,found));
}


std::string 
File::GetFullPath(const std::string &currentDir, const std::string &relFN) {

	std::string res,resAux;

	if (IsRelative(relFN)) {
		resAux = currentDir + PATH_SEPARATOR + relFN;
		res = CleanFullPath(resAux);
	}
	else
		res = CleanFullPath(relFN);

	return res;
}


std::string 
File::CleanFullPath(const std::string &fn) {

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
		pos1 = aux1.find_last_of(PATH_SEPARATOR);
		std::string aux2 = resAux.substr(pos+3,resAux.size()-1);
		resAux = aux1.substr(0,pos1) + resAux.substr(pos+3,resAux.size()-1);

		pos = resAux.find("\\..\\",0);
	}

	pos = resAux.find("\\../",0);
	while (pos != resAux.npos) {
		std::string aux1 = resAux.substr(0,pos);
		pos1 = aux1.find_last_of(PATH_SEPARATOR);
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
File::Validate(std::string s1) {

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
File::IsRelative(const std::string &fn) {

	if (fn[0] == '/' || fn[1] == ':')
		return(false);
	else
		return(true);
}

std::string  
File::GetRelativePathTo(const std::string &currentDir, const std::string &absFN) {

	// declarations - put here so this should work in a C compiler
	int afMarker = 0, rfMarker = 0;
	size_t cdLen = 0, afLen = 0;
	unsigned int i = 0;
	unsigned int levels = 0;
	static char relativeFilename[PATH_MAX+1];
	char *currentDirectory = (char *)currentDir.c_str();
	char *absoluteFilename = (char *)absFN.c_str();

	cdLen = strlen(currentDirectory);
	afLen = strlen(absoluteFilename);

	// make sure the names are not too long or too short
	if(cdLen > PATH_MAX || cdLen < ABSOLUTE_NAME_START+1 ||
		afLen > PATH_MAX || afLen < ABSOLUTE_NAME_START+1)
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

	if(i == cdLen && (absoluteFilename[i] == PATH_SEPARATOR[0] || absoluteFilename[i-1] == PATH_SEPARATOR[0]))
	{
		// the whole current directory name is in the file name,
		// so we just trim off the current directory name to get the
		// current file name.
		if(absoluteFilename[i] == PATH_SEPARATOR[0])
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
		if(currentDirectory[i] == PATH_SEPARATOR[0])
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
	while(afMarker > 0 && absoluteFilename[afMarker-1] != PATH_SEPARATOR[0])
	{
		afMarker--;
	}

	// check that the result will not be too long
	if(levels * 3 + afLen - afMarker > PATH_MAX)
	{
		return NULL;
	}

	// add the appropriate number of "..\"s.
	rfMarker = 0;
	for(i = 0; i < levels; i++)
	{
		relativeFilename[rfMarker++] = '.';
		relativeFilename[rfMarker++] = '.';
		relativeFilename[rfMarker++] = PATH_SEPARATOR[0];
	}

	// copy the rest of the filename into the result string
	strcpy(&relativeFilename[rfMarker], &absoluteFilename[afMarker]);

//	std::string cleanFilename = CleanFullPath(relativeFilename);
	return relativeFilename;
}



std::string
File::getURI (void)
{
	std::string aux (join (PROTOCOL, "/", true));
	
#if defined(NAU_PLATFORM_WIN32)
	// Do a final pass to convert all windows-specific chars to their URL 
	// legit counterparts

	std::string::size_type loc = aux.find ('\\', 0);
	while (loc != std::string::npos) {
	  aux[loc] = '/';
	  loc = aux.find ('\\', loc+1);
	}
#endif
	
	return (aux);
}

std::string 
File::getFilename (void)
{
	return m_FileName;
}

void
File::setFilename (std::string aFilename)
{
	m_FileName = aFilename;
}

std::string 
File::getExtension (void)
{
	return m_FileExtension;
}

std::string 
File::getFullPath (void)
{
	if (false == m_IsNative) {
		return (join ("", PATH_SEPARATOR, true));
	} else {
		return m_FileName; /***MARK***/
	}
}

std::string
File::getPath (void)
{
	if (false == m_IsNative) {
		return (join ("", PATH_SEPARATOR, false));
	} 
	return "";
}
std::string
File::getDrive (void)
{
	if (false == m_IsNative) {
		return m_Drive;
	}
	return "";
}

std::string
File::getRelativeTo (std::string path)
{
        File FilePath (path);
	return (getRelativeTo (FilePath));
}

std::string
File::getRelativeTo (File &file)
{
	File relative ("");

	if (m_Drive != file.getDrive()) {
		return file.getFullPath();
	}

	std::vector<std::string>::iterator pathIter1;
	std::vector<std::string>::iterator pathIter2;

	pathIter1 = m_vPath.begin();
	pathIter2 = file.getPathComponents().begin();
	
	if ( ((pathIter1 != m_vPath.end()) && (*pathIter1) == (*pathIter2)) && (pathIter2 != file.getPathComponents().end())) {

		pathIter1++;
		pathIter2++;
	}

	if (pathIter1 != m_vPath.end()) {
		while (pathIter1 != m_vPath.end()) {
			if (*pathIter1 != m_FileName) {
				relative.appendToPath ("..");
			}
			pathIter1++;
		}
	}
	if (pathIter2 != file.getPathComponents().end()) {
		while (pathIter2 != file.getPathComponents().end()) {
			relative.appendToPath (*pathIter2);
			pathIter2++;
		}
	}

	return relative.getFullPath();
}

std::vector<std::string>&
File::getPathComponents (void)
{
	return m_vPath;
}

void
File::appendToPath (std::string path)
{
	m_vPath.push_back (path);
}

std::string
File::getAbsolutePath ()
{
  // If path is not relative then we have nothing to do
  if (! m_IsRelative) {
    return (getFullPath());
  }

  std::string head (getCurrentWorkingDir());
  
  head.append (PATH_SEPARATOR);
  head.append (getFullPath());

  return (std::string (head));
  
}

void
File::construct (std::string filepath)
{
	if (filepath.size() < 1) {
		return;
	}

	checkURI (filepath);
	
#ifdef NAU_PLATFORM_WIN32
	size_t DrivePos = filepath.find (":");

	if (1 == DrivePos) {
		m_Drive = filepath.substr (0, 1);
		m_IsRelative = false;
		filepath = filepath.substr (2);	
	}

	
#else
	if ('/' == filepath.at (0)) {
		m_IsRelative = false;
		filepath = filepath.substr (1);
	}
#endif


	size_t OldPos = 0;
	size_t Pos = filepath.find_first_of (PATH_SEPARATOR);
	while (Pos != std::string::npos) {
		m_vPath.push_back (filepath.substr (OldPos, (Pos - OldPos)));
		OldPos = Pos + 1;
		Pos = filepath.find_first_of (PATH_SEPARATOR, Pos + 1);
	}

	std::string aux = filepath.substr (OldPos);

	if (aux.size() > 0) {
		m_IsLeaf = true;
		Pos = filepath.find_last_of (".");
		if (Pos != std::string::npos) {
			m_HasExtension = true;
			m_FileName = filepath.substr (OldPos, (Pos - OldPos));
			m_FileExtension = filepath.substr (Pos + 1);
		} else {
			m_FileName = filepath.substr (OldPos);
		}
	}
}

void
File::checkURI (std::string &filepath)
{
	if (filepath.size() < 1) {
		return;
	}

	size_t Pos = filepath.find (PROTOCOL);

	if (Pos != std::string::npos) {
		filepath = filepath.substr (8);
	}

	decode (filepath);

#ifdef NAU_PLATFORM_WIN32
	if ('/' == filepath.at (0)) {
		filepath = filepath.substr (1);
	}

	Pos = filepath.find_first_of ("/");
	while (Pos != std::string::npos) {
		filepath.replace (Pos, 1, "\\");
		Pos = filepath.find_first_of ("/", Pos + 1);
	}
#endif
}

std::string 
File::join (std::string head, std::string separator, bool file)
{
	std::stringstream FileURI (std::stringstream::out);

	FileURI << head;

	// Ugly code ahead. ""!=head means that the string is an URL because 
	// this method got abused to build non URL strings somewhere along the
	// way. This whole class needs refactoring asap.

	if ("" != head) { /***MARK***/ //Very, VERY, ugly!
			FileURI << separator;
			if (m_IsRelative) {
			  FileURI << getCurrentWorkingDir() << separator;
			}
#if defined(NAU_PLATFORM_WIN32)
			else {
			  FileURI << m_Drive << ":";
			}
#endif
	}
	else if (!m_IsRelative) {
#if defined (NAU_PLATFORM_WIN32)
	  FileURI << m_Drive << ":";
#else
	  FileURI << '/';
#endif
	}

	std::vector<std::string>::iterator stringIter;

	stringIter = m_vPath.begin();

	for ( ; stringIter != m_vPath.end(); stringIter++) {
		FileURI << (*stringIter) << separator;
	}

	if (true == file) {
		if (true == m_IsLeaf) {
			FileURI << m_FileName;
		}
		if (true == m_HasExtension) {
			FileURI << "." << m_FileExtension;
		}
	}

	return FileURI.str();
}

void
File::decode (std::string &filepath)
{

	/*
	Character					Code			 Code		
								  Points			Points
									(Hex) 		(Dec)
	Space							  20			 32		
   'Less Than' symbol ("<")  3C			 60
   'Greater Than' symbol (">")  3E	       62 
   'Pound' character ("#")  23	       35 	
   Percent character ("%")  25	       37 	
   */

	size_t Pos = 0;

	Pos = filepath.find ("%20");
	while (Pos != std::string::npos) {
		filepath.replace (Pos, 3, " ");
		Pos = filepath.find ("%20", Pos + 1);
	}

	Pos = filepath.find ("%3C");
	while (Pos != std::string::npos) {
		filepath.replace (Pos, 3, "<");
		Pos = filepath.find ("%3C", Pos + 1);
	}

	Pos = filepath.find ("%3E");
	while (Pos != std::string::npos) {
		filepath.replace (Pos, 3, ">");
		Pos = filepath.find ("%3E", Pos + 1);
	}

	Pos = filepath.find ("%23");
	while (Pos != std::string::npos) {
		filepath.replace (Pos, 3, "#");
		Pos = filepath.find ("%23", Pos + 1);
	}

	Pos = filepath.find ("%25", 0);
	while (Pos != std::string::npos) {
		filepath.replace (Pos, 3, "%");
		Pos = filepath.find ("%25", Pos + 1);
	}
}

File::FileType
File::getType (void) {

	if ("dae" == m_FileExtension) {
		return File::COLLADA;
	}
	else if ("patch" == m_FileExtension) {
		return File::PATCH;
	}
	else if ("3ds" == m_FileExtension) {
		return File::THREEDS;
	}
	else if ("nbo" == m_FileExtension) {
		return File::NAUBINARYOBJECT;
	}
	else if ("obj" == m_FileExtension) {
		return File::WAVEFRONTOBJ;
	}
	else if ("xml" == m_FileExtension) {
		return File::OGREXMLMESH;
	}
	else if ("blend" == m_FileExtension) {
		return File::BLENDER;
	}
	else if ("ply" == m_FileExtension) {
		return File::PLY;
	}
	else if ("lwo" == m_FileExtension) {
		return File::LIGHTWAVE;
	}
	else if ("stl" == m_FileExtension) {
		return File::STL;
	}
	else if ("cob" == m_FileExtension) {
		return File::TRUESPACE;
	}
	else if ("scn" == m_FileExtension) {
		return File::TRUESPACE;
	}
	else
	return File::UNKNOWN;
}

// Get our current working directory
std::string
File::getCurrentWorkingDir()
{
  
  bool PathSizeTooSmall = false;
  unsigned int PathSize = 255;
  char *cwd = new char[PathSize];
  
  do {
   
#if NAU_PLATFORM_WIN32 
       char *result = _getcwd (cwd, PathSize);
#else
       char *result = getcwd (cwd, PathSize);
#endif

    // Need to enlarge the path 
    if (NULL == result) {
      PathSizeTooSmall = true;
      
      delete [] cwd;
      PathSize *= 2;
      cwd = new char[PathSize];
    } 
    else {
      PathSizeTooSmall = false;
    }
  
  } while (PathSizeTooSmall);

  std::string head (cwd);
 
  delete cwd;

  return (std::string (head));
}

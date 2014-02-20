#include <nau/system/file.h>

#include <sstream>

using namespace nau::system;

#include <nau/config.h>

#ifdef NAU_PLATFORM_WIN32

#include <direct.h>

const std::string File::PATH_SEPARATOR ("\\");

#else

#include <unistd.h>

const std::string File::PATH_SEPARATOR ("/");

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
	m_IsNative (native)
{
	if (false == m_IsNative) {
		construct (filepath);
	}
}

File::~File(void)
{
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
	else if ("cbo" == m_FileExtension) {
		return File::NAUBINARYOBJECT;
	}
	else if ("obj" == m_FileExtension) {
		return File::WAVEFRONTOBJ;
	}
	else if ("xml" == m_FileExtension) {
		return File::OGREXMLMESH;
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

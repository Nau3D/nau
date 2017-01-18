#ifndef __FILEUTIL__
#define __FILEUTIL__

#include "nau/config.h"

#include <string>
#include <vector>

#define MAX_FILENAME_LEN 512

// The number of characters at the start of an absolute filename.  e.g. in DOS,
// absolute filenames start with "X:\" so this value should be 3, in UNIX they start
// with "\" so this value should be 1.
#define ABSOLUTE_NAME_START 3

// set this to '\\' for DOS or '/' for UNIX

#ifdef NAU_PLATFORM_WIN32
#define SLASH '\\'		
#else
#define SLASH '/'				
#endif'

namespace nau {

	namespace system {


		class FileUtil {

		private:
			FileUtil();
			~FileUtil();

		public:
			static std::string GetName(const std::string &fn);
			static std::string GetExtension(const std::string &fn);
			static std::string GetPath(const std::string &fn);
			static std::string GetRelativePathTo(const std::string &currentDir, const std::string &absFileName);
			static std::string GetFullPath(const std::string &currentDir, const std::string &relFileName);
			static bool IsRelative(const std::string &fn);
			static std::string CleanFullPath(const std::string &fn);
			static bool Exists(const std::string &fn);
			static std::string Validate(std::string s);
			static std::string BuildFullFileName(std::string path, std::string filename);
			static void RecurseDirectory(std::string path, std::vector<std::string> *res);
		};
	};
};

#endif

// textfile.h: interface for reading and writing text files
// www.lighthouse3d.com
//
// You may use these functions freely.
// they are provided as is, and no warranties, either implicit,
// or explicit are given
//////////////////////////////////////////////////////////////////////

#ifndef TEXTFILE_H
#define TEXTFILE_H

#include <stdio.h>
#include <stdlib.h>
#include <string>

namespace nau
{
	namespace system
	{

		std::string textFileRead(const std::string &fn);
		int textFileWrite(const std::string &fn, const std::string &s);
	};
};

#endif //TEXTFILE_H

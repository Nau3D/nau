// textfile.cpp
//
// simple reading and writing for text files
//
// www.lighthouse3d.com
//
// You may use these functions freely.
// they are provided as is, and no warranties, either implicit,
// or explicit are given
//////////////////////////////////////////////////////////////////////

#include "nau/system/textfile.h"

namespace nau
{
	namespace system
	{
		std::string 
		textFileRead (const std::string &fn) 
		{
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
		textFileWrite(const std::string &fn, const std::string &s) {

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
	};
};



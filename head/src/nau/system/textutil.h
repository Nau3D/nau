#ifndef TEXTUTIL_H
#define TEXTUTIL_H


#include <climits>
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <iterator>
#include <ostream>
#include <string>
#include <vector>

namespace nau {

  namespace system {

    namespace TextUtil {
      
      // Trims whitespace from the beginning and end of a string
      std::string TrimWhitespace (const std::string &InputString);

      // Create a random filename useful for temp files. Size is the size of the filename without the extension 
      std::string CreateRandomFilename (unsigned int Size = 8, const std::string Extension = "txt");

      // Parse an integer value from a string
      int ParseInt (const std::string &InputString);

      // Parse a float value from a string
      float ParseFloat (const std::string &InputString);

      // Create a string from an integer value
      std::string ToString (int Value);

      // Create a string from a float value
      std::string ToString (float Value);

      // Parse error constants
      const int ParseIntError = INT_MAX;
      const float ParseFloatError = (float)HUGE_VAL;

	  float * ParseFloats (const std::string &InputString, int count);
	  int *ParseInts (const std::string &InputString, int count);

	  //void Join(const std::vector<std::string>& vec, const char* delim, std::string *result);
	  void Join(const std::vector<std::string>& vec, std::string delim, std::string *result);


	  
	}; // namespace TextUtil

  }; // namespace system

}; // namespace nau

#endif // TEXTUTIL_H

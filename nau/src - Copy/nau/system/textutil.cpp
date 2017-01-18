#include "nau/system/textutil.h"

#include "nau/math/randomGen.h"

#include <iostream>
#include <sstream>

using namespace nau::system;


std::string
TextUtil::TrimWhitespace (const std::string &input_string)
{
  unsigned int lindex = (unsigned int)input_string.find_first_not_of (" \t\n");
  unsigned int rindex = (unsigned int)input_string.find_last_not_of (" \t\n");

  if (std::string::npos == lindex) {
    lindex = 0;
  }

  if (std::string::npos != rindex) {
    rindex++;   
  }
  
  return (input_string.substr (lindex, rindex-lindex));
}


std::string
TextUtil::CreateRandomFilename (unsigned int Size, const std::string Extension) {

  if (0 == Size) {
    Size = 4;
  }
  
  std::string filename (Size, 'x');

  nau::math::RandomGenerator gen;

  for (unsigned int i = 0; i < filename.size(); i++) {
    filename[i] = gen.randomInt ('a', 'z');
  }

  return filename + "." + Extension;
}


int 
TextUtil::ParseInt (const std::string &InputString) {

  std::istringstream stream (InputString);
  int intval;

  stream >> intval;

  if (stream.fail()) {
    return ParseIntError;
  }

  return intval;
}


float 
TextUtil::ParseFloat (const std::string &InputString) {

  std::istringstream stream (InputString);
  float floatval;

  stream >> floatval;

  if (stream.fail()) {
    return ParseFloatError;
  }

  return floatval;
}


std::string 
TextUtil::ToString (int Value) {

  std::ostringstream stream;
  
  stream << Value;

  return (stream.str());
}


std::string 
nau::system::TextUtil::ToString (float Value) {

  std::ostringstream stream;

  stream << Value;
  
  return (stream.str());
}


int *
TextUtil::ParseInts (const std::string &InputString, int count) {

	int i, n;
	char s[32];
	char *ptr = (char *)InputString.c_str();
	int *result = (int *)malloc(sizeof(int)*count);

	for (i = 0; i < count; i++) {
		if (sscanf(ptr, "%31[^ ]%n", s, &n) == 1) {
			result[i] = TextUtil::ParseInt(s);
			ptr +=n;
			if (*ptr != ' ' && *ptr != '\0')
				break;
			ptr++;
		}
		else
			break;
	}
	for (int j = i; j < count; j++) {
		result[j] = 0;
	}

	return result;
}


float *
TextUtil::ParseFloats (const std::string &InputString, int count) {

	int i, n;
	char s[32];
	char *ptr = (char *)InputString.c_str();
	float *result = (float *)malloc(sizeof(float)*count);

	for (i = 0; i < count; i++) {
		if (sscanf(ptr, "%31[^ ]%n", s, &n) == 1) {
			result[i] = TextUtil::ParseFloat(s);
			ptr +=n;
			if (*ptr != ' ' && *ptr != '\0')
				break;
			ptr++;
		}
		else
			break;
	}
	for (int j = i; j < count; j++) {
		result[j] = 0;
	}

	return result;
}


void
TextUtil::Join(const std::vector<std::string>& vec, std::string delim, std::string *result) {

	std::stringstream res;
	std::string s;
	for (auto s1 : vec) {

		s += s1 + delim;
	}
	*result = s.substr(0, s.length() - delim.length());
}

#include <nau/system/textutil.h>
#include <nau/math/randomgen.h>

#include <iostream>
#include <sstream>

// Trims whitespace from the beginning and end of a string
std::string
nau::system::textutil::TrimWhitespace (const std::string &input_string)
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
nau::system::textutil::CreateRandomFilename (unsigned int Size, const std::string Extension)
{
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

// Parse and integer value from a string
int 
nau::system::textutil::ParseInt (const std::string &InputString)
{
  std::istringstream stream (InputString);
  int intval;

  stream >> intval;

  if (stream.fail()) {
    return ParseIntError;
  }

  return intval;
}

// Parse a float value from a string
float 
nau::system::textutil::ParseFloat (const std::string &InputString)
{
  std::istringstream stream (InputString);
  float floatval;

  stream >> floatval;

  if (stream.fail()) {
    return ParseFloatError;
  }

  return floatval;
}

// Create a string from an integer value
std::string 
nau::system::textutil::ToString (int Value)
{
  std::ostringstream stream;
  
  stream << Value;

  return (stream.str());
}

// Create a string from a float value
std::string 
nau::system::textutil::ToString (float Value)
{
  std::ostringstream stream;
  
  stream << Value;
  
  return (stream.str());
}


int *
nau::system::textutil::ParseInts (const std::string &InputString, int count)
{
	int i, n;
	char s[32];
	char *ptr = (char *)InputString.c_str();
	int *result = (int *)malloc(sizeof(int)*count);

	for (i = 0; i < count; i++) {
		if (sscanf(ptr, "%31[^ ]%n", s, &n) == 1) {
			result[i] = textutil::ParseInt(s);
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
nau::system::textutil::ParseFloats (const std::string &InputString, int count)
{
	int i, n;
	char s[32];
	char *ptr = (char *)InputString.c_str();
	float *result = (float *)malloc(sizeof(float)*count);

	for (i = 0; i < count; i++) {
		if (sscanf(ptr, "%31[^ ]%n", s, &n) == 1) {
			result[i] = textutil::ParseFloat(s);
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
nau::system::textutil::join(const std::vector<std::string>& vec, const char* delim, std::string *result)
{
	std::stringstream res;
	std::string s;
	copy(vec.begin(), vec.end(), std::ostream_iterator<std::string>(res, delim));
	*result = res.str();
}
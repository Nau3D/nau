/*=============================================================================
  GLIntercept - OpenGL intercept/debugging tool
  Copyright (C) 2004  Damian Trebilco

  Licensed under the MIT license - See Docs\license.txt for details.
=============================================================================*/
#ifndef __FC_CONFIG_DATA_H_
#define __FC_CONFIG_DATA_H_

#include <string>
#include <vector>
#include <ConfigParser.h>

using namespace std;

//@
//  Summary:
//    This structure holds all the configuration data used by the Custom Get Uniforms
//    and associated classes.
//  
class GUConfigData
{
public:

  //@
  //  Summary:
  //    Constructor. Inits all config data to default values.
  //  
  GUConfigData();

  //@
  //  Summary:
  //    To read the config data values.
  //  
  //  Parameters:
  //    parser - The parser to read the data from.
  //
  void ReadConfigData(const ConfigParser *parser);

  vector<uint>    captureKeys;                     // The key combination to capture frame uniform data
  bool printUnfound;
  


protected:

  //@
  //  Summary:
  //    To get the key codes for the specified token value from the parser.
  //    If the token does not exist, the key code array is not changed.
  //
  //  Parameters:
  //    tokenName - The name of the token to extract from the parser.
  //
  //    parser   - The parser to read the data from.
  //
  //    keyCodes - The array to fill out with keycodes read fro the parser.
  //
  void GetKeyCodes(const string &tokenName, const ConfigParser *parser, vector<uint> &keyCodes) const;


};



#endif // __FC_CONFIG_DATA_H_

/*=============================================================================
  GLIntercept - OpenGL intercept/debugging tool
  Copyright (C) 2004  Damian Trebilco

  Licensed under the MIT license - See Docs\license.txt for details.
=============================================================================*/
#include <OSDefines.h>
#include "GUConfigData.h"

#include <ConfigParser.h>
#include <FileUtils.h>
#include <InputUtils.h>
#include <CommonErrorLog.h>

USING_ERRORLOG


///////////////////////////////////////////////////////////////////////////////
//
GUConfigData::GUConfigData():
printUnfound(true)
{

}

///////////////////////////////////////////////////////////////////////////////
//
void GUConfigData::ReadConfigData(const ConfigParser *parser)
{
  //Get the key codes
  GetKeyCodes("CaptureKeys",parser,captureKeys);
  const ConfigToken *printUnfoundToken = parser->GetToken("PrintUnfoundValues");
  if(printUnfoundToken)
  {
    printUnfoundToken->Get(printUnfound);
  }
}

///////////////////////////////////////////////////////////////////////////////
//
void GUConfigData::GetKeyCodes(const string &tokenName, const ConfigParser *parser, vector<uint> &keyCodes) const
{
  const ConfigToken *testToken;

  //Get the token for the string
  testToken = parser->GetToken(tokenName);
  if(testToken)
  {
    //Clear any existing key codes
    keyCodes.clear();

    //Loop for the number of values in the token
    for(uint i=0;i<testToken->GetNumValues();i++)
    {
      string value;
      testToken->Get(value,i);

      //Get the key code of the string
      uint newValue = InputUtils::GetKeyCode(value);
      if(newValue != 0)
      {
        //Add the value to the array
        keyCodes.push_back(newValue);
      }
    }       
  }

}

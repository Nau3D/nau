/*=============================================================================
  GLIntercept - OpenGL intercept/debugging tool
  Copyright (C) 2004  Damian Trebilco

  Licensed under the MIT license - See Docs\license.txt for details.
=============================================================================*/
#include "GUInput.h"

#include <InputUtils.h>
#include <CommonErrorLog.h>


USING_ERRORLOG


///////////////////////////////////////////////////////////////////////////////
//
GUInput::GUInput(const InputUtils *newInputUtils):
hookLevel(0),
inputUtils(newInputUtils)
{
}

///////////////////////////////////////////////////////////////////////////////
//
GUInput::~GUInput()
{
  //Un-register the hooks if hooks are registered
  UnHookInput();
}

///////////////////////////////////////////////////////////////////////////////
//
bool GUInput::IsKeyDown(uint keyCode)
{
  //If no hook/weak hook, just return the standard input value
  if(hookLevel <= 1)
  {
    return inputUtils->IsKeyDown(keyCode);
  }

  return false;
}

///////////////////////////////////////////////////////////////////////////////
//
bool GUInput::IsAllKeyDown(const vector<uint> &keyCodes)
{
  //If no hook/weak hook, just return the standard input value
  if(hookLevel <= 1)
  {
    return inputUtils->IsAllKeyDown(keyCodes);
  }

  return false;
}

#ifdef GLI_BUILD_WINDOWS

//Mouse/Keyboard hook pointers
HHOOK keyHook   = NULL;
HHOOK mouseHook = NULL;

//Prototypes for keyboard/mouse handlers
LRESULT CALLBACK KeyboardProc(int nCode, WPARAM wParam, LPARAM lParam);
LRESULT CALLBACK MouseProc   (int nCode, WPARAM wParam, LPARAM lParam); 

///////////////////////////////////////////////////////////////////////////////
//
bool GUInput::HookInput(uint newLevel)
{
  //If there exists an existing hook level, return false
  if(hookLevel > 0)
  {
    LOGERR(("GUInput::HookInput - Input is already hooked"));
    return false;
  }

  //If the new level is zero, just return
  if(newLevel == 0)
  {
    return true;
  }

  //Test if any the the hooks are still active
  if(keyHook != NULL || mouseHook != NULL)
  {
    LOGERR(("GUInput::HookInput - Mouse/keyboard hooks are still active"));
    return false;
  }

  //Test for standard application level hooking
  if(newLevel == 1)
  {
    //Set the keyboard hook
    keyHook = SetWindowsHookEx(WH_KEYBOARD,KeyboardProc,NULL,GetCurrentThreadId());
    if(keyHook == NULL)
    {
      LOGERR(("GUInput::HookInput - Key hooking failed."));
      return false;
    }

    /* Don't set the mouse hook for now
    //Set the mouse hook
    mouseHook = SetWindowsHookEx(WH_MOUSE,MouseProc,NULL,GetCurrentThreadId());
    if(mouseHook == NULL)
    {
      LOGERR(("GUInput::HookInput - Mouse hooking failed."));
      UnhookWindowsHookEx(keyHook);
      return false;
    }
    */

    //Assign the hook level
    hookLevel = newLevel;
    return true;
  }


  //Perhaps add DirectInput support here

  LOGERR(("GUInput::HookInput - Invalid level number %u",newLevel));
  return false;
}

///////////////////////////////////////////////////////////////////////////////
//
void GUInput::UnHookInput()
{
  //If we are hooking the input
  if(hookLevel > 0)
  {
    UnhookWindowsHookEx(keyHook);
    //UnhookWindowsHookEx(mouseHook);

    //Reset hook data
    hookLevel = 0;
    keyHook   = NULL;
    mouseHook = NULL;
  }
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//
//
//   Windows hooking functions
//
//
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
//
LRESULT CALLBACK KeyboardProc(int nCode, WPARAM wParam, LPARAM lParam) 
{ 
  //Do not process the message 
  if (nCode < 0)
  {
    return CallNextHookEx(keyHook, nCode, wParam, lParam);
  }

  return 1;
} 

///////////////////////////////////////////////////////////////////////////////
//
LRESULT CALLBACK MouseProc(int nCode, WPARAM wParam, LPARAM lParam) 
{ 
  //Do not process the message 
  if (nCode < 0)
  {
    return CallNextHookEx(mouseHook, nCode, wParam, lParam);
  }

  return 1;
} 

#endif // GLI_BUILD_WINDOWS


#ifdef GLI_BUILD_LINUX

///////////////////////////////////////////////////////////////////////////////
//
bool GUInput::HookInput(uint newLevel)
{
  return false;
}

///////////////////////////////////////////////////////////////////////////////
//
void GUInput::UnHookInput()
{
}

#endif // GLI_BUILD_LINUX

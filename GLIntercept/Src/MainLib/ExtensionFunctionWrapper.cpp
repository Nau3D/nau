/*=============================================================================
  GLIntercept - OpenGL intercept/debugging tool
  Copyright (C) 2011  Damian Trebilco

  Licensed under the MIT license - See Docs\license.txt for details.
=============================================================================*/
#include "ExtensionFunctionWrapper.h"
#include "GLDriver.h"
#include "FunctionParamStore.h"

#include <stdarg.h>

//The driver to log calls by
extern GLDriver glDriver;

USING_ERRORLOG

// Pre-definition of manual override functions
GLfloat GLAPIENTRY glGetPathLengthNV(GLuint path,GLsizei startSegment, GLsizei numSegments);
GLuint64 GLAPIENTRY glGetTextureHandleNV(GLuint texture);
GLuint64 GLAPIENTRY glGetTextureSamplerHandleNV(GLuint texture, GLuint sampler);
GLuint64 GLAPIENTRY glGetImageHandleNV(GLuint texture, GLint level, GLboolean layered, GLint layer, GLenum format);

// Enum of manual extension wrapped functions
enum ExtensionWrapperFunctions
{
  // Manual wrap for glGetPathLengthNV as it returns a float
  EWF_GL_GET_PATH_LENGTH_NV = 0, 

  // Functions that have uint64 return values
  EWF_GL_GET_TEXTURE_HANDLE_NV,
  EWF_GL_GET_TEXTURE_SAMPLER_HANDLE_NV,
  EWF_GL_GET_IMAGE_HANDLE_NV,

  EWF_MAX
};


// Struct of manual extension points
struct GLExtensions
{
  GLfloat (GLAPIENTRY *glGetPathLengthNV) (GLuint path,GLsizei startSegment, GLsizei numSegments);

  GLuint64 (GLAPIENTRY *glGetTextureHandleNV) (GLuint texture);
  GLuint64 (GLAPIENTRY *glGetTextureSamplerHandleNV) (GLuint texture, GLuint sampler);
  GLuint64 (GLAPIENTRY *glGetImageHandleNV) (GLuint texture, GLint level, GLboolean layered, GLint layer, GLenum format);
};
GLExtensions GLEXT = 
{
  NULL,

  NULL,
  NULL,
  NULL,
};


// Helper class to manage manual overrides
struct ManualExtension
{
  ManualExtension(const char * a_name, void * a_wrapPtr, void ** a_extFunction)
  : m_index(0)
  , m_name(a_name)
  , m_wrapPtr(a_wrapPtr)
  , m_extFunction(a_extFunction) 
  {
  }

  uint         m_index;  //!< The registered function index
  const char * m_name;   //!< The name of the function

  void *  m_wrapPtr;     //!< The manual function extension wrapper
  void ** m_extFunction; //!< Pointer to where the extension called function is stored

};

static ManualExtension s_manualExtensions[EWF_MAX] =
{
  ManualExtension("glGetPathLengthNV", (void*)glGetPathLengthNV, (void**)&GLEXT.glGetPathLengthNV),

  ManualExtension("glGetTextureHandleNV",        (void*)glGetTextureHandleNV,        (void**)&GLEXT.glGetTextureHandleNV),
  ManualExtension("glGetTextureSamplerHandleNV", (void*)glGetTextureSamplerHandleNV, (void**)&GLEXT.glGetTextureSamplerHandleNV),
  ManualExtension("glGetImageHandleNV",          (void*)glGetImageHandleNV,          (void**)&GLEXT.glGetImageHandleNV),
};


///////////////////////////////////////////////////////////////////////////////
//
GLfloat GLAPIENTRY glGetPathLengthNV(GLuint path,GLsizei startSegment, GLsizei numSegments)
{
  uint funcIndex = s_manualExtensions[EWF_GL_GET_PATH_LENGTH_NV].m_index;

#ifdef OS_ARCH_x86
  glDriver.LogFunctionPre (funcIndex,FunctionArgs((char*)&path));
#elif defined(OS_ARCH_x64)
  FunctionParamStore localParamStore = FunctionParamStore(path, startSegment, numSegments);
  glDriver.LogFunctionPre (funcIndex, FunctionArgs((char *)&localParamStore.m_paramStore[0]));
#else
  #error Unknown platform
#endif

  // Call the real method
  GLfloat retValue = GLEXT.glGetPathLengthNV(path, startSegment, numSegments);

  glDriver.LogFunctionPost(funcIndex,FunctionRetValue((void*)0,retValue));

  return retValue;
}


///////////////////////////////////////////////////////////////////////////////
//
GLuint64 GLAPIENTRY glGetTextureHandleNV(GLuint texture)
{
  uint funcIndex = s_manualExtensions[EWF_GL_GET_TEXTURE_HANDLE_NV].m_index;

#ifdef OS_ARCH_x86
  glDriver.LogFunctionPre (funcIndex,FunctionArgs((char*)&texture));
#elif defined(OS_ARCH_x64)
  FunctionParamStore localParamStore = FunctionParamStore(texture);
  glDriver.LogFunctionPre (funcIndex, FunctionArgs((char *)&localParamStore.m_paramStore[0]));
#else
  #error Unknown platform
#endif

  // Call the real method
  GLuint64 retValue = GLEXT.glGetTextureHandleNV(texture);

  glDriver.LogFunctionPost(funcIndex,FunctionRetValue(retValue));

  return retValue;
}

///////////////////////////////////////////////////////////////////////////////
//
GLuint64 GLAPIENTRY glGetTextureSamplerHandleNV(GLuint texture, GLuint sampler)
{
  uint funcIndex = s_manualExtensions[EWF_GL_GET_TEXTURE_SAMPLER_HANDLE_NV].m_index;

#ifdef OS_ARCH_x86
  glDriver.LogFunctionPre (funcIndex,FunctionArgs((char*)&texture));
#elif defined(OS_ARCH_x64)
  FunctionParamStore localParamStore = FunctionParamStore(texture, sampler);
  glDriver.LogFunctionPre (funcIndex, FunctionArgs((char *)&localParamStore.m_paramStore[0]));
#else
  #error Unknown platform
#endif

  // Call the real method
  GLuint64 retValue = GLEXT.glGetTextureSamplerHandleNV(texture, sampler);

  glDriver.LogFunctionPost(funcIndex,FunctionRetValue(retValue));

  return retValue;
}

///////////////////////////////////////////////////////////////////////////////
//
GLuint64 GLAPIENTRY glGetImageHandleNV(GLuint texture, GLint level, GLboolean layered, GLint layer, GLenum format)
{
  uint funcIndex = s_manualExtensions[EWF_GL_GET_IMAGE_HANDLE_NV].m_index;

#ifdef OS_ARCH_x86
  glDriver.LogFunctionPre (funcIndex,FunctionArgs((char*)&texture));
#elif defined(OS_ARCH_x64)
  FunctionParamStore localParamStore = FunctionParamStore(texture, level, layered, layer, format);
  glDriver.LogFunctionPre (funcIndex, FunctionArgs((char *)&localParamStore.m_paramStore[0]));
#else
  #error Unknown platform
#endif

  // Call the real method
  GLuint64 retValue = GLEXT.glGetImageHandleNV(texture, level, layered, layer, format);

  glDriver.LogFunctionPost(funcIndex,FunctionRetValue(retValue));

  return retValue;
}

///////////////////////////////////////////////////////////////////////////////
//
ExtensionFunctionWrapper::ExtensionFunctionWrapper(FunctionTable * functionTable):
functionTable(functionTable) 
{
}

///////////////////////////////////////////////////////////////////////////////
//
ExtensionFunctionWrapper::~ExtensionFunctionWrapper()
{

}

///////////////////////////////////////////////////////////////////////////////
//
bool ExtensionFunctionWrapper::IsManualWrapFunction(const string & funcName)
{
  // Loop for all registered functions
  for(uint i = 0; i < EWF_MAX; i++)
  {
    // If matching, return true
    if(funcName == s_manualExtensions[i].m_name)
    {
      return true;
    }
  }

  return false;
}

///////////////////////////////////////////////////////////////////////////////
//
void * ExtensionFunctionWrapper::AddFunctionWrap(const string & funcName,void * functionPtr)
{
  // Find the index of the function
  for(uint i = 0; i < EWF_MAX; i++)
  {
    // If matching, return true
    if(funcName == s_manualExtensions[i].m_name)
    {
      // Get the return pointer
      void * retPtr = s_manualExtensions[i].m_wrapPtr;
      
      // Get the pointer to the place where the internal call is stored (so it can be overwritten if necessary)
      void ** wrapDataFunc = s_manualExtensions[i].m_extFunction;

      // Store the function pointer
      *wrapDataFunc = functionPtr;

      //Add the data to the table and get the index
      int funcIndex = functionTable->AddFunction(funcName, retPtr, functionPtr, wrapDataFunc, false);
      
      // Assign the new function index
      s_manualExtensions[i].m_index = funcIndex;

      return retPtr;
    }
  }

  return NULL;
}





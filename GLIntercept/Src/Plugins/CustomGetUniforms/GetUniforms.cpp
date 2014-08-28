#include "GetUniforms.h"

#include <ConfigParser.h>
#include <CommonErrorLog.h>

#include <cstdlib>
#include <sstream>
#include <iostream>

USING_ERRORLOG

//CUSTOM UNIFORM FETCHING Plugin Fetching glProgramUniform1f
//CUSTOM UNIFORM FETCHING Plugin Fetching glProgramUniform2f
//CUSTOM UNIFORM FETCHING Plugin Fetching glProgramUniform3f
//CUSTOM UNIFORM FETCHING Plugin Fetching glProgramUniform1i
//CUSTOM UNIFORM FETCHING Plugin Fetching glProgramUniform2i
//CUSTOM UNIFORM FETCHING Plugin Fetching glProgramUniform3i
//CUSTOM UNIFORM FETCHING Plugin Fetching glProgramUniform1ui
//CUSTOM UNIFORM FETCHING Plugin Fetching glProgramUniform2ui
//CUSTOM UNIFORM FETCHING Plugin Fetching glProgramUniform3ui
//CUSTOM UNIFORM FETCHING Plugin Fetching glProgramUniform1fv
//CUSTOM UNIFORM FETCHING Plugin Fetching glProgramUniform2fv
//CUSTOM UNIFORM FETCHING Plugin Fetching glProgramUniform3fv
//CUSTOM UNIFORM FETCHING Plugin Fetching glProgramUniform1iv
//CUSTOM UNIFORM FETCHING Plugin Fetching glProgramUniform2iv
//CUSTOM UNIFORM FETCHING Plugin Fetching glProgramUniform3iv
//CUSTOM UNIFORM FETCHING Plugin Fetching glProgramUniform1uiv
//CUSTOM UNIFORM FETCHING Plugin Fetching glProgramUniform2uiv
//CUSTOM UNIFORM FETCHING Plugin Fetching glProgramUniform3uiv
//CUSTOM UNIFORM FETCHING Plugin Fetching glProgramUniformMatrix2fv
//CUSTOM UNIFORM FETCHING Plugin Fetching glProgramUniformMatrix3fv
//CUSTOM UNIFORM FETCHING Plugin Fetching glProgramUniformMatrix4fv
//CUSTOM UNIFORM FETCHING Plugin Fetching glProgramUniformMatrix2x3fv
//CUSTOM UNIFORM FETCHING Plugin Fetching glProgramUniformMatrix3x2fv
//CUSTOM UNIFORM FETCHING Plugin Fetching glProgramUniformMatrix2x4fv
//CUSTOM UNIFORM FETCHING Plugin Fetching glProgramUniformMatrix4x2fv
//CUSTOM UNIFORM FETCHING Plugin Fetching glProgramUniformMatrix3x4fv
//CUSTOM UNIFORM FETCHING Plugin Fetching glProgramUniformMatrix4x3fv

extern string dllPath;

static int customgetuniforms_framecounter = 0;

GetUniforms::GetUniforms(InterceptPluginCallbacks *callBacks):
selected_uniform(0),
enableKeyState(false),
inputSys(callBacks->GetInputUtils()),
gliCallBacks(callBacks)
{
  //DEPRECATED
  LOGERR(("CUSTOM UNIFORM FETCHING Plugin has been successfully created"));
  gliCallBacks->RegisterGLFunction("glGetActiveUniform");

  int std_uniform_count = 4;
  std::string std_uniform_types[] = { 
	  "f","i","ui","fv","iv","uiv"
  };
  int std_uniform_types_size = 6;

  std::string std_uniform_matrices[] = { 
	  "Matrix2fv",
	  "Matrix3fv",
	  "Matrix4fv",
	  "Matrix2x3fv",
	  "Matrix3x2fv",
	  "Matrix2x4fv",
	  "Matrix4x2fv",
	  "Matrix3x4fv",
	  "Matrix4x3fv"
  };
  int std_uniform_matrices_size = 9;

  ostringstream os;
  string programUniform;
  for (int t=0; t< std_uniform_types_size; t++){
	  for (int i=1; i<=std_uniform_count; i++){
		  os.str("");
		  os.clear();
		  os << "glProgramUniform" << i << std_uniform_types[t];
		  programUniform = os.str();
		  gliCallBacks->RegisterGLFunction(programUniform.c_str());
          //LOGERR(("CUSTOM UNIFORM FETCHING Plugin Fetching %s",programUniform.c_str()));
	  }
  }

  for (int t=0; t< std_uniform_matrices_size; t++){
	  os.str("");
	  os.clear();
	  os << "glProgramUniform" << std_uniform_matrices[t];
	  programUniform = os.str();
	  gliCallBacks->RegisterGLFunction(programUniform.c_str());
      //LOGERR(("CUSTOM UNIFORM FETCHING Plugin Fetching %s",programUniform.c_str()));
  }
  //gliCallBacks->RegisterGLFunction("glProgramUniform*");

    //Parse the config file
  ConfigParser fileParser;
  if(fileParser.Parse(dllPath + "config.ini"))
  {
    configData.ReadConfigData(&fileParser);
    fileParser.LogUnusedTokens(); 
  }

  //Parse the config string
  ConfigParser stringParser;
  if(stringParser.ParseString(gliCallBacks->GetConfigString()))
  {
    configData.ReadConfigData(&stringParser);
    stringParser.LogUnusedTokens(); 
  }

}

///////////////////////////////////////////////////////////////////////////////
//
GetUniforms::~GetUniforms()
{

}

///////////////////////////////////////////////////////////////////////////////
//
void GetUniforms::GLFunctionPre (uint updateID, const char *funcName, uint funcIndex, const FunctionArgs & args )
{
	char buffer[1024] = {0};
	const char *charptr;

	void *pointer;

	GLuint program = -1;
	GLuint location = -1;
	GLsizei bufSize = -1;
	GLsizei *length;
	GLint *size;

    FunctionArgs accessArgs(args);
	if (strcmp("glGetActiveUniform",funcName) == 0){

		accessArgs.Get(program);
		accessArgs.Get(location);
		accessArgs.Get(bufSize);
		accessArgs.Get(pointer); length = (GLsizei *) pointer;
		accessArgs.Get(pointer); size = (GLint *) pointer;
		accessArgs.Get(pointer); enum_holder = (GLenum *) pointer;
		accessArgs.Get(pointer); string_holder = (GLchar *) pointer;
		
		SelectUniform(program, location);

	}
	else{
		charptr = strstr(funcName,"glProgramUniform");
		if (charptr){
			accessArgs.Get(program);
			accessArgs.Get(location);
			SelectUniform(program, location);
			charptr+=16;
			selected_uniform->Update(charptr, accessArgs);
		}
	}

}


///////////////////////////////////////////////////////////////////////////////
//
void GetUniforms::GLFunctionPost(uint updateID, const char *funcName, uint funcIndex, const FunctionRetValue & retVal)
{
	//DEPRECATED
	if (strcmp("glGetActiveUniform",funcName) == 0){
		selected_uniform->SetType(enum_holder[0]);
		selected_uniform->SetName((char *)string_holder);
		//LOGERR(("GLFunctionPostI: %s", selected_uniform->InfoString())); //name does not exist before executing the function
	}
	//else if (strstr(funcName,"glProgramUniform")){
	//	LOGERR(("GLFunctionPostV function %s: %s", funcName, selected_uniform->ValueString()));
	//}
}

///////////////////////////////////////////////////////////////////////////////
//
void GetUniforms::GLRenderPre(const char *funcName, uint funcIndex, const FunctionArgs & args)
{
}

///////////////////////////////////////////////////////////////////////////////
//
void GetUniforms::GLRenderPost(const char *funcName, uint funcIndex, const FunctionRetValue & retVal)
{

}

///////////////////////////////////////////////////////////////////////////////
//
void GetUniforms::GLFrameEndPre(const char *funcName, uint funcIndex, const FunctionArgs & args )
{

}


///////////////////////////////////////////////////////////////////////////////
//
void GetUniforms::GLFrameEndPost(const char *funcName, uint funcIndex, const FunctionRetValue & retVal)
{
  
  //Check if the enable key state has changed
	if(inputSys.IsAllKeyDown(configData.captureKeys) != enableKeyState)
	{
	//Change the enabled key state
	enableKeyState = !enableKeyState;

	//If the keys were pressed, enable/disable the free camera mode
		if(enableKeyState)
		{
			//Print uniforms data
			LOGERR(("--- CustomGetUniforms uniform data on frame: %i", customgetuniforms_framecounter));
			for (int i = 0; i < uniforms.size(); i++){
				PrintProgramUniforms(uniforms[i]);
			}
			LOGERR(("--- CustomGetUniforms uniform data end"));
		}
	}
	
	customgetuniforms_framecounter++;
}


///////////////////////////////////////////////////////////////////////////////
//
void GetUniforms::OnGLContextSet(HGLRC oldRCHandle, HGLRC newRCHandle)
{
  
}

///////////////////////////////////////////////////////////////////////////////
//
void GetUniforms::OnGLContextCreate(HGLRC rcHandle)
{

}

///////////////////////////////////////////////////////////////////////////////
//
void GetUniforms::OnGLContextDelete(HGLRC rcHandle)
{

}

///////////////////////////////////////////////////////////////////////////////
//
void GetUniforms::OnGLContextShareLists(HGLRC srcHandle, HGLRC dstHandle)
{

}

///////////////////////////////////////////////////////////////////////////////
//
void GetUniforms::OnGLError(const char *funcName, uint funcIndex)
{

}

///////////////////////////////////////////////////////////////////////////////
//
void GetUniforms::Destroy()
{
  LOGERR(("CustomGetUniforms Plugin Destroyed"));
  
  //Don't do this:
  //gliCallBacks->DestroyPlugin();

  //Destroy this plugin
  delete this;
}

void GetUniforms::SelectUniform(unsigned int program,unsigned int location){
	if (uniforms.find(program) == uniforms.end() || uniforms[program].find(location) == uniforms[program].end()){
		selected_uniform = new UniformData(program, location);
		uniforms[program].insert(pair<int, UniformData *>(location, selected_uniform));
	}
	else{
		selected_uniform = uniforms[program][location];
	}
}

void GetUniforms::PrintProgramUniforms(map<int, UniformData *> &program){
	std::map<int,std::vector<int>> loclist;

	for (int i = 0; i < program.size(); i++){
		if (configData.printUnfound ||  program[i]->IsValueSet()){
			LOGERR(("%s", program[i]->UniformString()));
		}
	}
}

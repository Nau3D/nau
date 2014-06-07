#include "GLIntercept.h"
#include "ConfigData.h"
#include "GLDriver.h"


#define EXPORTING_DLL
#include "ConfigDataExport.h"


ConfigData *configDataExport = NULL;
FunctionTable * functionTableExport = NULL;
GLDriver *glDriver = NULL;

void setConfigDataExport(ConfigData *configData, GLDriver *driver){
	configDataExport = configData;
	glDriver = driver;
}

void setFunctionTableExport(FunctionTable * fTable){
	functionTableExport = fTable;
}
  void gliSetLogEnabled(bool logEnabled){
	  configDataExport->logEnabled=logEnabled;
  }

  bool gliIsLogEnabled(){
	  return configDataExport->logEnabled;
  }

  void gliSetLogXMLFormat(bool logXMLFormat){
	  configDataExport->logXMLFormat=logXMLFormat;
  }

  void gliSetLogFlush(bool logFlush){
	  configDataExport->logFlush=logFlush;
  }


  void gliSetLogMaxFrameLoggingEnabled(bool logMaxFrameLoggingEnabled){
	  configDataExport->logMaxFrameLoggingEnabled=logMaxFrameLoggingEnabled;
  }

  void gliSetLogMaxNumLogFrames(unsigned int logMaxNumLogFrames){
	  configDataExport->logMaxNumLogFrames=logMaxNumLogFrames;
  }

  
  void gliSetLogXSLFile(const char *logXSLFile){
	  configDataExport->logXSLFile=logXSLFile;
  }

  void gliSetLogXSLBaseDir(const char *logXSLBaseDir){
	  configDataExport->logXSLBaseDir=logXSLBaseDir;
  }


  void gliSetErrorGetOpenGLChecks(bool errorGetOpenGLChecks){
	  configDataExport->errorGetOpenGLChecks=errorGetOpenGLChecks;
  }

  void gliSetErrorThreadChecking(bool errorThreadChecking){
	  configDataExport->errorThreadChecking=errorThreadChecking;
  }

  void gliSetErrorBreakOnError(bool errorBreakOnError){
	  configDataExport->errorBreakOnError=errorBreakOnError;
  }

  void gliSetErrorLogOnError(bool errorLogOnError){
	  configDataExport->errorLogOnError=errorLogOnError;
  }

  void gliSetErrorExtendedLogError(bool errorExtendedLogError){
	  configDataExport->errorExtendedLogError=errorExtendedLogError;
  }

  void gliSetErrorDebuggerErrorLog(bool errorDebuggerErrorLog){
	  configDataExport->errorDebuggerErrorLog=errorDebuggerErrorLog;
  }


  void gliSetLogPerFrame(bool logPerFrame){
	  configDataExport->logPerFrame=logPerFrame;
  }

  void gliSetLogOneFrameOnly(bool logOneFrameOnly){
	  configDataExport->logOneFrameOnly=logOneFrameOnly;
  }

  void gliAddLogFrameKeys(const char *logFrameKeys){
	  unsigned int key = InputUtils::GetKeyCode(logFrameKeys);
	  configDataExport->logFrameKeys.push_back(key);
  }

  void gliClearLogFrameKeys(){
	  configDataExport->logFrameKeys.clear();
  }


  void gliSetLogPath(const char *logPath){
	  configDataExport->logPath=logPath;
  }

  const char *gliGetLogPath(){
	  return configDataExport->logPath.c_str();
  }

  void gliSetLogName(const char *logName){
	  configDataExport->logName=logName;
  }

  const char *gliGetLogName(){
	  if (configDataExport->logName.length()>0){
		return configDataExport->logName.c_str();
	  }
	  return "gliInterceptLog";
  }

  void gliSetFunctionDataFileName(const char *functionDataFileName){
	  configDataExport->functionDataFileName=functionDataFileName;
  }


  void gliSetImageLogEnabled(bool imageLogEnabled){
	  configDataExport->imageLogEnabled=imageLogEnabled;
  }

  void gliSetImageRenderCallStateLog(bool imageRenderCallStateLog){
	  configDataExport->imageRenderCallStateLog=imageRenderCallStateLog;
  }

  void gliSetImageSaveIcon(bool imageSaveIcon){
	  configDataExport->imageSaveIcon=imageSaveIcon;
  }

  void gliSetImageIconSize(unsigned int imageIconSize){
	  configDataExport->imageIconSize=imageIconSize;
  }

  void gliSetImageIconFormat(const char *imageIconFormat){
	  configDataExport->imageIconFormat=imageIconFormat;
  }


  void gliSetImageSavePNG(bool imageSavePNG){
	  configDataExport->imageSavePNG=imageSavePNG;
  }

  void gliSetImageSaveTGA(bool imageSaveTGA){
	  configDataExport->imageSaveTGA=imageSaveTGA;
  }

  void gliSetImageSaveJPG(bool imageSaveJPG){
	  configDataExport->imageSaveJPG=imageSaveJPG;
  }

  void gliSetImageFlipXAxis(bool imageFlipXAxis){
	  configDataExport->imageFlipXAxis=imageFlipXAxis;
  }

  void gliSetImageCubeMapTile(bool imageCubeMapTile){
	  configDataExport->imageCubeMapTile=imageCubeMapTile;
  }


  void gliSetImageSave1D(bool imageSave1D){
	  configDataExport->imageSave1D=imageSave1D;
  }

  void gliSetImageSave2D(bool imageSave2D){
	  configDataExport->imageSave2D=imageSave2D;
  }

  void gliSetImageSave3D(bool imageSave3D){
	  configDataExport->imageSave3D=imageSave3D;
  }

  void gliSetImageSaveCube(bool imageSaveCube){
	  configDataExport->imageSaveCube=imageSaveCube;
  }

  void gliSetImageSavePBufferTex(bool imageSavePBufferTex){
	  configDataExport->imageSavePBufferTex=imageSavePBufferTex;
  }


  void gliSetShaderLogEnabled(bool shaderLogEnabled){
	  configDataExport->shaderLogEnabled=shaderLogEnabled;
  }

  void gliSetShaderRenderCallStateLog(bool shaderRenderCallStateLog){
	  configDataExport->shaderRenderCallStateLog=shaderRenderCallStateLog;
  }

  void gliSetShaderAttachLogState(bool shaderAttachLogState){
	  configDataExport->shaderAttachLogState=shaderAttachLogState;
  }

  void gliSetShaderValidatePreRender(bool shaderValidatePreRender){
	  configDataExport->shaderValidatePreRender=shaderValidatePreRender;
  }

  void gliSetShaderLogUniformsPreRender(bool shaderLogUniformsPreRender){
	  configDataExport->shaderLogUniformsPreRender=shaderLogUniformsPreRender;
  }


  void gliSetDisplayListLogEnabled(bool displayListLogEnabled){
	  configDataExport->displayListLogEnabled=displayListLogEnabled;
  }



  void gliSetFrameLogEnabled(bool frameLogEnabled){
	  configDataExport->frameLogEnabled=frameLogEnabled;
  }

  void gliAddFrameAdditionalRenderCalls(const char *frameAdditionalRenderCalls){
	  configDataExport->frameAdditionalRenderCalls.push_back(frameAdditionalRenderCalls);
  }

  void gliClearFrameAdditionalRenderCalls(){
	  configDataExport->frameAdditionalRenderCalls.clear();
  }


  void gliSetFrameImageFormat(const char *frameImageFormat){
	  configDataExport->frameImageFormat=frameImageFormat;
  }

  void gliSetFramePreColorSave(bool framePreColorSave){
	  configDataExport->framePreColorSave=framePreColorSave;
  }

  void gliSetFramePostColorSave(bool framePostColorSave){
	  configDataExport->framePostColorSave=framePostColorSave;
  }

  void gliSetFrameDiffColorSave(bool frameDiffColorSave){
	  configDataExport->frameDiffColorSave=frameDiffColorSave;
  }


  void gliSetFramePreDepthSave(bool framePreDepthSave){
	  configDataExport->framePreDepthSave=framePreDepthSave;
  }

  void gliSetFramePostDepthSave(bool framePostDepthSave){
	  configDataExport->framePostDepthSave=framePostDepthSave;
  }

  void gliSetFrameDiffDepthSave(bool frameDiffDepthSave){
	  configDataExport->frameDiffDepthSave=frameDiffDepthSave;
  }


  void gliSetFramePreStencilSave(bool framePreStencilSave){
	  configDataExport->framePreStencilSave=framePreStencilSave;
  }

  void gliSetFramePostStencilSave(bool framePostStencilSave){
	  configDataExport->framePostStencilSave=framePostStencilSave;
  }

  void gliSetFrameDiffStencilSave(bool frameDiffStencilSave){
	  configDataExport->frameDiffStencilSave=frameDiffStencilSave;
  }

  void gliAddFrameStencilColors(unsigned int frameStencilColors){
	  configDataExport->frameStencilColors.push_back(frameStencilColors);
  }
  
  void gliClearFrameStencilColors(){
	  configDataExport->frameStencilColors.clear();
  }

  void gliSetFrameIconSave(bool frameIconSave){
	  configDataExport->frameIconSave=frameIconSave;
  }

  void gliSetFrameIconSize(unsigned int frameIconSize){
	  configDataExport->frameIconSize=frameIconSize;
  }

  void gliSetFrameIconImageFormat(const char *frameIconImageFormat){
	  configDataExport->frameIconImageFormat=frameIconImageFormat;
  }


  void gliSetFrameMovieEnabled(bool frameMovieEnabled){
	  configDataExport->frameMovieEnabled=frameMovieEnabled;
  }

  void gliSetFrameMovieWidth(unsigned int frameMovieWidth){
	  configDataExport->frameMovieWidth=frameMovieWidth;
  }

  void gliSetFrameMovieHeight(unsigned int frameMovieHeight){
	  configDataExport->frameMovieHeight=frameMovieHeight;
  }

  void gliSetFrameMovieRate(unsigned int frameMovieRate){
	  configDataExport->frameMovieRate=frameMovieRate;
  }

  void gliAddFrameMovieCodecs(const char *frameMovieCodecs){
	  configDataExport->frameMovieCodecs.push_back(frameMovieCodecs);
  }

  void gliClearFrameMovieCodecs(){
	  configDataExport->frameMovieCodecs.clear();
  }


  void gliSetTimerLogEnabled(bool timerLogEnabled){
	  configDataExport->timerLogEnabled=timerLogEnabled;
  }

  void gliSetTimerLogCutOff(unsigned int timerLogCutOff){
	  configDataExport->timerLogCutOff=timerLogCutOff;
  }



  void gliSetPluginBasePath (const char *pluginBasePath){
	  configDataExport->pluginBasePath=pluginBasePath;
  }

  
  //void gliSetPluginDataArray(const char *pluginName, const char *pluginDLLName, const char *pluginConfigData, unsigned int index){

	 // printf("Setting Plugins...\n");
	 // ConfigData::PluginData data;
	 // data.pluginName=pluginName;
	 // data.pluginDLLName=pluginDLLName;
	 // data.pluginConfigData=pluginConfigData;

	 // if (configDataExport->pluginDataArray.size()<=index){
		//  configDataExport->pluginDataArray.push_back(data);
	 // }
	 // else{
		//  configDataExport->pluginDataArray[index]=data;
	 // }

	 // printf("Reloading Plugins...\n");

	 // glDriver->ReloadPlugins();
	 // printf("Plugins Reloaded\n");
  //}

  void gliAddPluginData(const char *pluginName, const char *pluginDLLName, const char *pluginConfigData){
	  ConfigData::PluginData data;
      data.pluginName=pluginName;
	  data.pluginDLLName=pluginDLLName;
	  data.pluginConfigData=pluginConfigData;
		  
	  configDataExport->pluginDataArray.push_back(data);
  }

  void gliClearPluginDataArray(){
	  configDataExport->pluginDataArray.clear();
  }

  //void printConfigBasic(){
	 // printf("\nConfig status\n");
	 // printf("Log enabled %s\n",(configDataExport->logEnabled ? "true" : "false"));
  //    printf("Log name: %s\n",configDataExport->logName.c_str());
	 // for (int i=0;i<configDataExport->pluginDataArray.size();i++){
		//  printf("%d - PluginName: %s\n",i,configDataExport->pluginDataArray[i].pluginName);
		//  printf("%d - PluginDLLName: %s\n",i,configDataExport->pluginDataArray[i].pluginDLLName);
		//  if (configDataExport->pluginDataArray[i].pluginConfigData.size())
		//	printf("%d - PluginData: %s\n",i,configDataExport->pluginDataArray[i].pluginConfigData);
	 // }
  //}

  void gliReConfigure(){
	  gliReInit();
  }

  void gliResetConfigure(){
	  gliReset();
  }

  void gliInitGliLog(){
	  gliLogInit();
  }

  void gliKillGliLog(){
	  gliLogKill();
  }

  int gliGetEnumsCount(){
	  return functionTableExport->GetEnumArraySize();
  }

  const char *gliGetEnumsName(unsigned int index){
	  return functionTableExport->GetEnumData(index)->GetName().c_str();
  }
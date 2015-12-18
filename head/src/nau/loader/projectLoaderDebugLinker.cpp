#include <map>
#include <string>
#include "nau/loader/projectLoaderDebugLinker.h"
#include "..\..\GLIntercept\Src\MainLib\ConfigDataExport.h"

using namespace std;

typedef struct functionSet{
	void *function;
	void (*functionClear)();
	unsigned int type;
}functionSet;

map<string,map<string,functionSet>> attributeList;
//FunctionLog
//  LogEnabled      
//  LogFlush        
//  LogPath         
//  LogFileName     
//  AdditionalRenderCalls 
//  LogMaxNumFrames 
//  LogFormat       
//  XMLFormat
//    XSLFile 
//    BaseDir 

//LogPerFrame
//  Enabled         
//  FrameStartKeys  
//  OneFrameOnly    

//InputFiles
//  GLFunctionDefines 
//  GLSystemLib       

//ErrorChecking
//  GLErrorChecking 
//  ThreadChecking  
//  BreakOnError    
//  LogOnError      
//  ExtendedErrorLog 
//  DebuggerErrorLog 

//ImageLog
//  LogEnabled    
//  RenderCallStateLog 
//  SaveFormats    
//  FlipXAxis      
//  TileCubeMaps   
//  SaveGLTypes    
//  SavePbufferTex 
//  ImageIcon
//    Enabled    
//    SaveFormat 
//    Size       

//ShaderLog
//  LogEnabled    
//  RenderCallStateLog 
//  AttachLogState 
//  ValidatePreRender 
//  UniformLogPreRender 

//DisplayListLog
//  LogEnabled

//FrameLog
//  LogEnabled    
//  SaveFormat    
//  FrameIcon->Enabled 
//  FrameIcon->SaveFormat 
//  FrameIcon->Size    
//  FrameMovie->Enabled 
//  FrameMovie->Size    
//  FrameMovie->FrameRate 
//  FrameMovie->Compression  
//  ColorBufferLog 
//  DepthBufferLog 
//  StencilBufferLog 
//  StencilColors 

//TimerLog
//  LogEnabled    
//  LogCutoff    

void activateGLI(){
	gliSetIsGLIActive(true);
}

void addGLInterceptFunction(void *function,void (*functionClear)(),unsigned int type,const char* name,const char* mapName){
	functionSet temp={function,functionClear,type};
	map<string,functionSet> *mapPointer = &(attributeList.find(mapName)->second);
	mapPointer->insert(pair<string, functionSet>(name,temp));
}

bool isGLInterceptFunctionsInit = false;

void initAttributeListMaps(){
	attributeList.clear();
	
	attributeList.insert(pair<string,map<string,functionSet>>("functionlog",map<string,functionSet>()));
	//attributeList.insert(pair<string,map<string,functionSet>>("functionlogxmlFormat",map<string,functionSet>()));
	attributeList.insert(pair<string,map<string,functionSet>>("logperframe",map<string,functionSet>()));
	//attributeList.insert(pair<string,map<string,functionSet>>("inputfiles",map<string,functionSet>()));
	attributeList.insert(pair<string,map<string,functionSet>>("errorchecking",map<string,functionSet>()));
	attributeList.insert(pair<string,map<string,functionSet>>("imagelog",map<string,functionSet>()));
	attributeList.insert(pair<string,map<string,functionSet>>("imagelogimageicon",map<string,functionSet>()));
	attributeList.insert(pair<string,map<string,functionSet>>("shaderlog",map<string,functionSet>()));
	attributeList.insert(pair<string,map<string,functionSet>>("displaylistlog",map<string,functionSet>()));
	attributeList.insert(pair<string,map<string,functionSet>>("framelog",map<string,functionSet>()));
	attributeList.insert(pair<string,map<string,functionSet>>("framelogframeicon",map<string,functionSet>()));
	attributeList.insert(pair<string,map<string,functionSet>>("framelogframemovie",map<string,functionSet>()));
	attributeList.insert(pair<string,map<string,functionSet>>("timerlog",map<string,functionSet>()));
}

void initGLInterceptFunctions(){
	if (!isGLInterceptFunctionsInit){

		initAttributeListMaps();

		addGLInterceptFunction((void*)&gliSetLogEnabled,NULL,GLIEnums::FunctionType::BOOL,"enabled","functionlog");
		//addGLInterceptFunction((void*)&gliSetLogXMLFormat,NULL,GLIEnums::FunctionType::BOOL,"logxmlformat","functionlog");
		//addGLInterceptFunction((void*)&gliSetLogFlush,NULL,GLIEnums::FunctionType::BOOL,"logflush","functionlog");
		addGLInterceptFunction((void*)&gliSetLogMaxFrameLoggingEnabled,NULL,GLIEnums::FunctionType::BOOL,"logmaxframeloggingenabled","functionlog");
		addGLInterceptFunction((void*)&gliSetLogPath,NULL,GLIEnums::FunctionType::STRING,"logpath","functionlog");
		addGLInterceptFunction((void*)&gliSetLogName,NULL,GLIEnums::FunctionType::STRING,"logname","functionlog");
		addGLInterceptFunction((void*)&gliSetLogMaxNumLogFrames,NULL,GLIEnums::FunctionType::UINT,"logmaxnumlogframes","functionlog");
		//addGLInterceptFunction((void*)&gliSetLogXSLFile,NULL,GLIEnums::FunctionType::STRING,"logxslfile","functionlogxmlFormat");
		//addGLInterceptFunction((void*)&gliSetLogXSLBaseDir,NULL,GLIEnums::FunctionType::STRING,"logxslbasedir","functionlogxmlFormat");

		addGLInterceptFunction((void*)&gliSetLogPerFrame,NULL,GLIEnums::FunctionType::BOOL,"logperframe","logperframe");
		addGLInterceptFunction((void*)&gliSetLogOneFrameOnly,NULL,GLIEnums::FunctionType::BOOL,"logoneframeonly","logperframe");
		addGLInterceptFunction((void*)&gliAddLogFrameKeys,&gliClearLogFrameKeys,GLIEnums::FunctionType::STRINGARRAY,"logframekeys","logperframe");

		addGLInterceptFunction((void*)&gliSetErrorGetOpenGLChecks,NULL,GLIEnums::FunctionType::BOOL,"errorgetopenglchecks","errorchecking");
		addGLInterceptFunction((void*)&gliSetErrorThreadChecking,NULL,GLIEnums::FunctionType::BOOL,"errorthreadchecking","errorchecking");
		addGLInterceptFunction((void*)&gliSetErrorBreakOnError,NULL,GLIEnums::FunctionType::BOOL,"errorbreakonerror","errorchecking");
		addGLInterceptFunction((void*)&gliSetErrorLogOnError,NULL,GLIEnums::FunctionType::BOOL,"errorlogonerror","errorchecking");
		addGLInterceptFunction((void*)&gliSetErrorExtendedLogError,NULL,GLIEnums::FunctionType::BOOL,"errorextendedlogerror","errorchecking");
		addGLInterceptFunction((void*)&gliSetErrorDebuggerErrorLog,NULL,GLIEnums::FunctionType::BOOL,"errordebuggererrorlog","errorchecking");

		addGLInterceptFunction((void*)&gliSetImageLogEnabled,NULL,GLIEnums::FunctionType::BOOL,"enabled","imagelog");
		addGLInterceptFunction((void*)&gliSetImageRenderCallStateLog,NULL,GLIEnums::FunctionType::BOOL,"imagerendercallstatelog","imagelog");
		addGLInterceptFunction((void*)&gliSetImageSavePNG,NULL,GLIEnums::FunctionType::BOOL,"imagesavepng","imagelog");
		addGLInterceptFunction((void*)&gliSetImageSaveTGA,NULL,GLIEnums::FunctionType::BOOL,"imagesavetga","imagelog");
		addGLInterceptFunction((void*)&gliSetImageSaveJPG,NULL,GLIEnums::FunctionType::BOOL,"imagesavejpg","imagelog");
		addGLInterceptFunction((void*)&gliSetImageFlipXAxis,NULL,GLIEnums::FunctionType::BOOL,"imageflipxaxis","imagelog");
		addGLInterceptFunction((void*)&gliSetImageCubeMapTile,NULL,GLIEnums::FunctionType::BOOL,"imagecubemaptile","imagelog");
		addGLInterceptFunction((void*)&gliSetImageSave1D,NULL,GLIEnums::FunctionType::BOOL,"imagesave1d","imagelog");
		addGLInterceptFunction((void*)&gliSetImageSave2D,NULL,GLIEnums::FunctionType::BOOL,"imagesave2d","imagelog");
		addGLInterceptFunction((void*)&gliSetImageSave3D,NULL,GLIEnums::FunctionType::BOOL,"imagesave3d","imagelog");
		addGLInterceptFunction((void*)&gliSetImageSaveCube,NULL,GLIEnums::FunctionType::BOOL,"imagesavecube","imagelog");
		addGLInterceptFunction((void*)&gliSetImageSavePBufferTex,NULL,GLIEnums::FunctionType::BOOL,"imagesavepbuffertex","imagelog");
		addGLInterceptFunction((void*)&gliSetImageSaveIcon,NULL,GLIEnums::FunctionType::BOOL,"imagesaveicon","imagelogimageicon");
		addGLInterceptFunction((void*)&gliSetImageIconSize,NULL,GLIEnums::FunctionType::UINT,"imageiconsize","imagelogimageicon");
		addGLInterceptFunction((void*)&gliSetImageIconFormat,NULL,GLIEnums::FunctionType::STRING,"imageiconformat","imagelogimageicon");

		addGLInterceptFunction((void*)&gliSetShaderLogEnabled,NULL,GLIEnums::FunctionType::BOOL,"enabled","shaderlog");
		addGLInterceptFunction((void*)&gliSetShaderRenderCallStateLog,NULL,GLIEnums::FunctionType::BOOL,"shaderrendercallstatelog","shaderlog");
		addGLInterceptFunction((void*)&gliSetShaderAttachLogState,NULL,GLIEnums::FunctionType::BOOL,"shaderattachlogstate","shaderlog");
		addGLInterceptFunction((void*)&gliSetShaderValidatePreRender,NULL,GLIEnums::FunctionType::BOOL,"shadervalidateprerender","shaderlog");
		addGLInterceptFunction((void*)&gliSetShaderLogUniformsPreRender,NULL,GLIEnums::FunctionType::BOOL,"shaderloguniformsprerender","shaderlog");

		addGLInterceptFunction((void*)&gliSetDisplayListLogEnabled,NULL,GLIEnums::FunctionType::BOOL,"displaylistlogenabled","displaylistlog");

		addGLInterceptFunction((void*)&gliSetFrameLogEnabled,NULL,GLIEnums::FunctionType::BOOL,"enabled","framelog");
		addGLInterceptFunction((void*)&gliSetFrameImageFormat,NULL,GLIEnums::FunctionType::STRING,"frameimageformat","framelog");
		addGLInterceptFunction((void*)&gliAddFrameStencilColors,&gliClearFrameStencilColors,GLIEnums::FunctionType::UINTARRAY,"framestencilcolors","framelog");
		addGLInterceptFunction((void*)&gliSetFramePreColorSave,NULL,GLIEnums::FunctionType::BOOL,"frameprecolorsave","framelog");
		addGLInterceptFunction((void*)&gliSetFramePostColorSave,NULL,GLIEnums::FunctionType::BOOL,"framepostcolorsave","framelog");
		addGLInterceptFunction((void*)&gliSetFrameDiffColorSave,NULL,GLIEnums::FunctionType::BOOL,"framediffcolorsave","framelog");
		addGLInterceptFunction((void*)&gliSetFramePreDepthSave,NULL,GLIEnums::FunctionType::BOOL,"framepredepthsave","framelog");
		addGLInterceptFunction((void*)&gliSetFramePostDepthSave,NULL,GLIEnums::FunctionType::BOOL,"framepostdepthsave","framelog");
		addGLInterceptFunction((void*)&gliSetFrameDiffDepthSave,NULL,GLIEnums::FunctionType::BOOL,"framediffdepthsave","framelog");
		addGLInterceptFunction((void*)&gliSetFramePreStencilSave,NULL,GLIEnums::FunctionType::BOOL,"frameprestencilsave","framelog");
		addGLInterceptFunction((void*)&gliSetFramePostStencilSave,NULL,GLIEnums::FunctionType::BOOL,"framepoststencilsave","framelog");
		addGLInterceptFunction((void*)&gliSetFrameDiffStencilSave,NULL,GLIEnums::FunctionType::BOOL,"framediffstencilsave","framelog");
		//addGLInterceptFunction((void*)&gliAddFrameAdditionalRenderCalls,&gliClearFrameAdditionalRenderCalls,GLIEnums::FunctionType::STRINGARRAY,"frameAdditionalRenderCalls","framelog");
		addGLInterceptFunction((void*)&gliSetFrameIconSave,NULL,GLIEnums::FunctionType::BOOL,"frameiconsave","framelogframeicon");
		addGLInterceptFunction((void*)&gliSetFrameIconSize,NULL,GLIEnums::FunctionType::UINT,"frameiconsize","framelogframeicon");
		addGLInterceptFunction((void*)&gliSetFrameIconImageFormat,NULL,GLIEnums::FunctionType::STRING,"frameiconimageformat","framelogframeicon");
		addGLInterceptFunction((void*)&gliSetFrameMovieEnabled,NULL,GLIEnums::FunctionType::BOOL,"framemovieenabled","framelogframemovie");
		addGLInterceptFunction((void*)&gliSetFrameMovieWidth,NULL,GLIEnums::FunctionType::UINT,"framemoviewidth","framelogframemovie");
		addGLInterceptFunction((void*)&gliSetFrameMovieHeight,NULL,GLIEnums::FunctionType::UINT,"framemovieheight","framelogframemovie");
		addGLInterceptFunction((void*)&gliSetFrameMovieRate,NULL,GLIEnums::FunctionType::UINT,"framemovierate","framelogframemovie");
		addGLInterceptFunction((void*)&gliAddFrameMovieCodecs,&gliClearFrameMovieCodecs,GLIEnums::FunctionType::STRINGARRAY,"frameMovieCodecs","framelogframemovie");


		addGLInterceptFunction((void*)&gliSetTimerLogEnabled,NULL,GLIEnums::FunctionType::BOOL,"enabled","timerlog");
		addGLInterceptFunction((void*)&gliSetTimerLogCutOff,NULL,GLIEnums::FunctionType::UINT,"timerlogcutoff","timerlog");
	
		//addGLInterceptFunction((void*)&gliSetFunctionDataFileName,NULL,GLIEnums::FunctionType::STRING,"functiondatafilename","inputfiles");
		//addGLInterceptFunction((void*)&gliSetPluginBasePath,NULL,GLIEnums::FunctionType::STRING,"pluginbasepath","inputfiles");

		isGLInterceptFunctionsInit=true;
	}
}

void useGLIFunction(void *functionSetPointer, void *value){
	functionSet *setPointer = (functionSet*)functionSetPointer;
	switch (setPointer->type)
	{
	case GLIEnums::FunctionType::BOOL:{
			void (*functionPointer)(bool) = (void (*)(bool))setPointer->function;
			functionPointer(*((bool*)value));
			break;
		}
	case GLIEnums::FunctionType::INT:{
			void (*functionPointer)(int) = (void (*)(int))setPointer->function;
			functionPointer(*((int*)value));
			break;
		}
	case GLIEnums::FunctionType::UINT:
	case GLIEnums::FunctionType::UINTARRAY:{
			void (*functionPointer)(unsigned int) = (void (*)(unsigned int))setPointer->function;
			functionPointer(*((unsigned int*)value));
			break;
		}
	case GLIEnums::FunctionType::STRING:
	case GLIEnums::FunctionType::STRINGARRAY:{
			void (*functionPointer)(const char*) = (void (*)(const char*))setPointer->function;
			functionPointer((const char*)value);
			break;
		}
	default:
		break;
	}
}

void useGLIClearFunction(void *functionSetPointer){
	functionSet *setPointer = (functionSet*)functionSetPointer;
	if (setPointer->functionClear){
		setPointer->functionClear();
	}
}

void *getGLIFunction(const char *name, const char *mapName){
	map<string,functionSet> *mapPointer = &(attributeList.find(mapName)->second);
	functionSet *setPointer=NULL;
	map<string,functionSet>::iterator it = mapPointer->find(name);
	if (it!=mapPointer->end()){
		setPointer=&it->second;
	}
	return setPointer;
}

unsigned int getGLIFunctionType(void *functionSetPointer){
	functionSet *setPointer = (functionSet*)functionSetPointer;
	if (setPointer){
		return setPointer->type;
	}
	return GLIEnums::FunctionType::NIL;
}

void startGLIConfiguration(){
	gliReConfigure();
}

void addPlugin(const char *pluginName, const char *pluginDLLName, const char *pluginConfigData){
	gliAddPluginData(pluginName, pluginDLLName, pluginConfigData);
}

void clearPlugins(){
	gliClearPluginDataArray();
}

void startGlilog(){
	gliInitGliLog();
}

bool isDebugLogEnabled(){
	return gliIsLogEnabled();
}

bool debugLogPath(){
	return (gliGetLogPath() != NULL);
}

bool debugLogName(){
	return (gliGetLogName() != NULL);
}


void addMessageToGLILog(const char * message){
	return gliInsertLogMessage(message);
}

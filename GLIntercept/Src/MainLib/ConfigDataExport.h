#ifndef __CONFIG_DATA_EXPORT_H_
#define __CONFIG_DATA_EXPORT_H_




#include <string>
#include <vector>

using namespace std;

#ifdef EXPORTING_DLL
#define dlllibextern extern __declspec(dllexport)
#else
#define dlllibextern extern __declspec(dllimport)
#endif

  dlllibextern void gliSetLogEnabled(bool logEnabled);                                // Flag to indicate if logging is enabled
  dlllibextern bool gliIsLogEnabled();
  dlllibextern void gliSetLogXMLFormat(bool logXMLFormat);                              // If true use XML to log, else use plain text.
  dlllibextern void gliSetLogFlush(bool logFlush);                                  // If true, the logger is flushed after each function name is written (text file only)

  dlllibextern void gliSetLogMaxFrameLoggingEnabled(bool logMaxFrameLoggingEnabled);                 // If a maximum number of frames to log has been specified
  dlllibextern void gliSetLogMaxNumLogFrames(unsigned int logMaxNumLogFrames);                        // The maximum number of frames to log (if logMaxFrameLoggingEnabled is true)
  
  dlllibextern void gliSetLogXSLFile(const char *logXSLFile);                              // The XSL file to use when XML logging is enabled
  dlllibextern void gliSetLogXSLBaseDir(const char *logXSLBaseDir);                           // The base directory where the XSL file is located

  dlllibextern void gliSetErrorGetOpenGLChecks(bool errorGetOpenGLChecks);                      // Flag to indicate if errors are checked for
  dlllibextern void gliSetErrorThreadChecking(bool errorThreadChecking);                       // Flag to indicate if thread checking is performed.
  dlllibextern void gliSetErrorBreakOnError(bool errorBreakOnError);                         // Flag to indicate if to break on an error
  dlllibextern void gliSetErrorLogOnError(bool errorLogOnError);                           // Flag to indicate if log OpenGL errors
  dlllibextern void gliSetErrorExtendedLogError(bool errorExtendedLogError);                     // Flag to indicate if extended data about an error is reported
  dlllibextern void gliSetErrorDebuggerErrorLog(bool errorDebuggerErrorLog);                     // Flag to indicate if the error log is mirrored to the debugger

  dlllibextern void gliSetLogPerFrame(bool logPerFrame);                               // Flag to indicate if we log per-frame or by the entire application
  dlllibextern void gliSetLogOneFrameOnly(bool logOneFrameOnly);                           // Flag to indicate if per-frame logging will only get one frame at a time
  dlllibextern void gliAddLogFrameKeys(const char *logFrameKeys);                      // The key codes used to enable per-frame logging
  dlllibextern void gliClearLogFrameKeys();                      // The key codes used to enable per-frame logging

  dlllibextern void gliSetLogPath(const char *logPath);                                 // The path to write the log files (including trailing seperator)
  dlllibextern const char *gliGetLogPath();
  dlllibextern void gliSetLogName(const char *logName);                                 // The name of the log to write out (without extension)
  dlllibextern const char *gliGetLogName();
  dlllibextern void gliSetFunctionDataFileName(const char *functionDataFileName);                    // The name of the file/path to find the function config data

  dlllibextern void gliSetImageLogEnabled(bool imageLogEnabled);                           // Flag to indicate if the image log is enabled
  dlllibextern void gliSetImageRenderCallStateLog(bool imageRenderCallStateLog);                   // Flag to indicate if the image state is recorded on render calls
  dlllibextern void gliSetImageSaveIcon(bool imageSaveIcon);                             // Save a icon version of the images 
  dlllibextern void gliSetImageIconSize(unsigned int imageIconSize);                             // The size of the icon if saving icons
  dlllibextern void gliSetImageIconFormat(const char *imageIconFormat);                         // The image format of the image icons

  dlllibextern void gliSetImageSavePNG(bool imageSavePNG);                              // Save the images in PNG format
  dlllibextern void gliSetImageSaveTGA(bool imageSaveTGA);                              // Save the images in TGA format
  dlllibextern void gliSetImageSaveJPG(bool imageSaveJPG);                              // Save the images in JPG format
  dlllibextern void gliSetImageFlipXAxis(bool imageFlipXAxis);                            // Flip the images on the X axis before writting out
  dlllibextern void gliSetImageCubeMapTile(bool imageCubeMapTile);                          // Flag to indicate if cube maps are tiled together or saved as six images 

  dlllibextern void gliSetImageSave1D(bool imageSave1D);                               // Flag to indicate if 1D textures are saved
  dlllibextern void gliSetImageSave2D(bool imageSave2D);                               // Flag to indicate if 2D textures are saved (includes rect images)
  dlllibextern void gliSetImageSave3D(bool imageSave3D);                               // Flag to indicate if 3D textures are saved
  dlllibextern void gliSetImageSaveCube(bool imageSaveCube);                             // Flag to indicate if Cube textures are saved
  dlllibextern void gliSetImageSavePBufferTex(bool imageSavePBufferTex);                       // Flag to indicate if textures that are bound to p-buffers are saved

  dlllibextern void gliSetShaderLogEnabled(bool shaderLogEnabled);                          // Flag to indicate if the shader log is enabled
  dlllibextern void gliSetShaderRenderCallStateLog(bool shaderRenderCallStateLog);                  // Flag to indicate if the shader state is recorded on render calls
  dlllibextern void gliSetShaderAttachLogState(bool shaderAttachLogState);                      // Flag to indicate if the shader log data is to be written.
  dlllibextern void gliSetShaderValidatePreRender(bool shaderValidatePreRender);                   // Flag to indicate if the shader is validated before each render.
  dlllibextern void gliSetShaderLogUniformsPreRender(bool shaderLogUniformsPreRender);                // Flag to indicate if the shader is to log the uniforms before each render.

  dlllibextern void gliSetDisplayListLogEnabled(bool displayListLogEnabled);                     // Flag to indicate if the display list log is enabled


  dlllibextern void gliSetFrameLogEnabled(bool frameLogEnabled);                           // Flag to indicate if the frame log is enabled
  dlllibextern void gliAddFrameAdditionalRenderCalls(const char *frameAdditionalRenderCalls);      // Additional functions for which to dump framebuffer
  dlllibextern void gliClearFrameAdditionalRenderCalls();

  dlllibextern void gliSetFrameImageFormat(const char *frameImageFormat);                        // The format to save frame images in.
  dlllibextern void gliSetFramePreColorSave(bool framePreColorSave);                         // Save pre-color frame images
  dlllibextern void gliSetFramePostColorSave(bool framePostColorSave);                        // Save post-color frame images
  dlllibextern void gliSetFrameDiffColorSave(bool frameDiffColorSave);                        // Save diff-color frame images

  dlllibextern void gliSetFramePreDepthSave(bool framePreDepthSave);                         // Save pre-depth frame images
  dlllibextern void gliSetFramePostDepthSave(bool framePostDepthSave);                        // Save post-depth frame images
  dlllibextern void gliSetFrameDiffDepthSave(bool frameDiffDepthSave);                        // Save diff-depth frame images

  dlllibextern void gliSetFramePreStencilSave(bool framePreStencilSave);                       // Save pre-stencil frame images
  dlllibextern void gliSetFramePostStencilSave(bool framePostStencilSave);                      // Save post-stencil frame images
  dlllibextern void gliSetFrameDiffStencilSave(bool frameDiffStencilSave);                      // Save diff-stencil frame images
  dlllibextern void gliAddFrameStencilColors(unsigned int frameStencilColors);                // The stencil color values to use when logging a stancil frame
  dlllibextern void gliClearFrameStencilColors();

  dlllibextern void gliSetFrameIconSave(bool frameIconSave);                             // If icon images of the frame buffer are saved
  dlllibextern void gliSetFrameIconSize(unsigned int frameIconSize);                             // The save size of the frame buffer icon images
  dlllibextern void gliSetFrameIconImageFormat(const char *frameIconImageFormat);                    // The image format of the frame buffer icon images

  dlllibextern void gliSetFrameMovieEnabled(bool frameMovieEnabled);                         // If frame movies are enabled
  dlllibextern void gliSetFrameMovieWidth(unsigned int frameMovieWidth);                           // The frame movie width
  dlllibextern void gliSetFrameMovieHeight(unsigned int frameMovieHeight);                          // The frame movie height
  dlllibextern void gliSetFrameMovieRate(unsigned int frameMovieRate);                            // The frame movie rate
  dlllibextern void gliAddFrameMovieCodecs(const char *frameMovieCodecs);                // The fraem movie codecs for compression
  dlllibextern void gliClearFrameMovieCodecs();

  dlllibextern void gliSetTimerLogEnabled(bool timerLogEnabled);                           // Flag to indicate if the timer log is enabled
  dlllibextern void gliSetTimerLogCutOff(unsigned int timerLogCutOff);                            // The cutoff value for the timer log.


  dlllibextern void gliSetPluginBasePath (const char *pluginBasePath);                          // The root path to the plugin directory
  
  dlllibextern void gliAddPluginData(const char *pluginName, const char *pluginDLLName, const char *pluginConfigData);  
  dlllibextern void gliClearPluginDataArray();
  
  //dlllibextern void printConfigBasic(); //Removed

  dlllibextern void gliReConfigure();
  dlllibextern void gliResetConfigure();
  dlllibextern void gliInitGliLog();
  dlllibextern void gliKillGliLog();

  dlllibextern int gliGetEnumsCount();
  dlllibextern const char *gliGetEnumsName(unsigned int index);
#endif


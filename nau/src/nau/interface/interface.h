#ifndef INTERFACE
#define INTERFACE

#include <AntTweakBar.h>

#include <map>
#include <string>
#include <vector>

#define INTERFACE_MANAGER nau::inter::ToolBar::GetInstance()

namespace nau {

	namespace inter {

		class ToolBar {

		protected:

			typedef struct {
				std::string type, context, component;
				int id;
				std::string luaScript, luaScriptFile;
			} NauVar;

			std::vector< NauVar *> m_ClientDataVec;

			static TwStructMember Vec2Members[2], Vec3Members[3], Vec4Members[4],
				UIVec2Members[2], UIVec3Members[3], 
				IVec2Members[2], IVec3Members[3],
				BVec4Members[4], 
				Mat3Members[9], Mat4Members[16];

			static TwType Vec2, Vec3, Vec4,
				UIVec2, UIVec3,
				IVec2, IVec3,
				BVec4,
				Mat3, Mat4;

			static void TW_CALL SetPipelineCallBack(const void * value, void * clientData);
			static void TW_CALL GetPipelineCallBack(void * value, void * clientData);

			static void TW_CALL SetColorCallBack(const void * value, void * clientData);
			static void TW_CALL GetColorCallBack(void *value, void *clientData);

			static void TW_CALL SetDirCallBack(const void * value, void * clientData);
			static void TW_CALL GetDirCallBack(void *value, void *clientData);

			static void TW_CALL SetCallBack(const void *value, void *clientData);
			static void TW_CALL SetBoolCallBack(const void *value, void *clientData);
			static void TW_CALL SetIntCallBack(const void *value, void *clientData);
			static void TW_CALL SetFloatCallBack(const void *value, void *clientData);
			static void TW_CALL SetUIntCallBack(const void *value, void *clientData);

			static void TW_CALL GetIntCallBack(void *value, void *clientData);

			static void TW_CALL GetFloatCallBack(void *value, void *clientData);
			static void TW_CALL GetVec2CallBack(void *value, void *clientData);
			static void TW_CALL GetVec3CallBack(void *value, void *clientData);
			static void TW_CALL GetVec4CallBack(void *value, void *clientData);

			static void TW_CALL GetUIntCallBack(void *value, void *clientData);
			static void TW_CALL GetUIVec2CallBack(void *value, void *clientData);
			static void TW_CALL GetUIVec3CallBack(void *value, void *clientData);

			static void TW_CALL GetBoolCallBack(void *value, void *clientData);
			static void TW_CALL GetBVec4CallBack(void *value, void *clientData);

			static void TW_CALL GetMat3CallBack(void *value, void *clientData);
			static void TW_CALL GetMat4CallBack(void *value, void *clientData);

			static ToolBar *Instance;

			// maps from the window label to pair<window label, window pointer>
			std::map<std::string, std::pair<std::string,TwBar *>> m_Windows;
			
			ToolBar();



		public:

			static ToolBar *GetInstance();

			~ToolBar();

			void clear();
			void init();

			void render();

			static void TW_CALL ErrorHandler(const char * errorMessage);

			bool createWindow(const std::string &label);

			bool addColor(const std::string &windowName, const std::string &varLabel,
				const std::string &varType, const std::string &varContext,
				const std::string &component, int id = 0, const std::string &luaScript = "", const std::string &luaScriptFile = "");
			
			bool addDir(const std::string &windowName, const std::string &varLabel,
				const std::string &varType, const std::string &varContext,
				const std::string &component, int id = 0, const std::string &luaScript = "", const std::string &luaScriptFile = "");

			bool addEnum(const std::string &windowName, const std::string &varLabel,
				const std::string &varType, const std::string &varContext,
				const std::string &component, const std::string &enums, int id = 0, 
				const std::string def = "", const std::string &luaScript = "", const std::string &luaScriptFile = "");

			bool addVar(const std::string &windowName, const std::string &varLabel,
				const std::string &varType, const std::string &varContext,
				const std::string &component, int id = 0, const std::string def = "", 
				const std::string &luaScript = "", const std::string &luaScriptFile = "");

			bool addPipelineList(const std::string &windowName, const std::string &label, const std::string &luaScript = "", const std::string &luaScriptFile = "");

			void resize(unsigned int width, unsigned int heigth);
			//bool setWindowProp(const std::string &name, const std::string &prop);
			//bool setVarProp(const std::string &name, const std::string &var, const std::string &prop);
		};
	};
	
};


#endif
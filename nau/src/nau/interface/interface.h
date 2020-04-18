#ifndef INTERFACE
#define INTERFACE

#include <map>
#include <string>
#include <vector>

#include <nau/attribute.h>

#define INTERFACE_MANAGER nau::inter::ToolBar::GetInstance()

namespace nau {

	namespace inter {

		class ToolBar {
		public:

			enum varClass {NAU_VAR, PIPELINE_LIST, CUSTOM_ENUM};
			struct NauVar {
				varClass aClass;
				std::string label;
				std::string type, context, component;
				int id;
				Enums::DataType dt;
				std::string luaScript, luaScriptFile;
				Attribute::Semantics semantics;
				// only valid for int fields. Allows the usage of strings instead fo numbers 
				std::vector<char> options;
				float min, max, step;
			};
			typedef std::vector< NauVar> Items;

		protected:

			static ToolBar *Instance;

			// maps from the window label to pair<window label, window pointer>
			std::map<std::string, Items> m_Windows;
			
			ToolBar();
			void parse(const std::string& s, float& max, float& min, float& step);

		public:

			static ToolBar *GetInstance();

			~ToolBar();

			void clear();
			// just for compatibility with AntTweakBar solution
			void render();

			const std::map<std::string, ToolBar::Items>& getWindows();

			bool createWindow(const std::string &label);

			bool addColor(const std::string &windowName, const std::string &varLabel,
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
		};
	};
	
};


#endif
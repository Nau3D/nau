#ifndef IGLOBALSTATE_H
#define IGLOBALSTATE_H

#include <map>
#include <string>
#include <vector>


namespace nau
{
	namespace render
	{
		class IGlobalState 
		{
		public:	
			static IGlobalState *Create();

			void addStateEnum(std::string &name, std::string &functionName,
				unsigned int penumValue, unsigned int pLength);
			void getStateEnumNames(std::vector<std::string> *enumNames);
			std::string getState(std::string enumName);

		protected:

			IGlobalState();

			// Get types
			enum FunctionType{
				BOOLEANV,
				DOUBLEV,
				FLOATV,
				INTEGERV,
				INTEGERI_V,
				INTEGER64V
			};

			// stores global state variables
			struct StateVariable{
				StateVariable();
				StateVariable(std::string pfunctionName, unsigned int penumValue, 
														 unsigned int plength = 1);
				std::string functionName;
				int length;
				int enumValue;
			};

			// map to store global state variables
			std::map<std::string, StateVariable> stateVariablesMap;
			// map from function name to function type
			static std::map<std::string, FunctionType> FunctionMap;

			virtual std::string getStateValue(unsigned int enumValue, FunctionType type) = 0;
			virtual std::string getStateValue(unsigned int enumValue, 
											FunctionType type, 
											unsigned int length) = 0;

		};
	};
};

#endif
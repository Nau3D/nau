#include "../nau/src/nau.h"

#include <vector>
#include <string>
#include <stdio.h>

#include <windows.h>
#include <DbgHelp.h>
#pragma comment(lib,"Dbghelp")

#define _CRTDBG_MAP_ALLOC
#define _CRTDBG_MAP_ALLOC_NEW
#include <stdlib.h>
#include <crtdbg.h>

#ifdef _DEBUG
#ifndef DBG_NEW
#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#define new DBG_NEW
#endif
#endif  // _DEBUG

#ifdef GLINTERCEPTDEBUG
#include "..\..\..\GLIntercept\Src\MainLib\ConfigDataExport.h"
#endif

using namespace nau;

int main(int argc, char** argv) {

	AttribSet* a;

	std::vector<std::string> types;
	Enums::DataType dt;
	const std::vector<std::string> &data = Enums::GetDataTypeToString();


	NAU->getValidObjectTypes(&types);

	for (auto &t : types) {
		printf("%s\n", t.c_str());

		a = NAU->getAttribs(t);
		std::map<std::string, std::unique_ptr<Attribute>>& attrs = a->getAttributes();
		for (auto &attr : attrs) {
			printf("\t%s", attr.first.c_str());

			dt = attr.second->getType();
			if (attr.second->getReadOnlyFlag())
				printf("\t(read only)");
			printf("\t%s ", data[dt].c_str());
			if (dt == Enums::ENUM) {
				printf("\t{");


				const std::vector<std::string> &options = a->getListString(attr.second->getId());
				for (auto option : options) {
					printf("%s ", option.c_str());
				}
				printf("}");
				std::shared_ptr<nau::math::Data> &def = attr.second->getDefault();
				int x = *(int*)def->getPtr();
				if (x) {
					printf("\tdefault: %s", attr.second->getOptionString(x).c_str());
				}
				printf("\n");
			}
			else {
				std::shared_ptr<nau::math::Data>& min = attr.second->getMin();
				std::shared_ptr<nau::math::Data>& max = attr.second->getMax();
				if (min != NULL)
					printf("\tRange: [%s, ", Enums::valueToString(dt, min.get()).c_str());
				else if (max != NULL)
					printf("\tRange: [ , ");
				if (max != NULL)
					printf("%s] ", Enums::valueToString(dt, max.get()).c_str());
				else if (min != NULL)
					printf("] ");
				std::shared_ptr<nau::math::Data>& def = attr.second->getDefault();
				if (def)
					printf("\tdefault: %s", Enums::valueToString(dt, def.get()).c_str());
				printf("\n");
			}
			printf("\n");
		}
	}

}


/*
int main(int argc, char **argv) {

	_CrtDumpMemoryLeaks();	
	_CrtMemState s1,s2,s3;
	_CrtMemCheckpoint(&s1); 
	
	char *www = (char *)malloc(10);
	_CrtDumpMemoryLeaks();
	www = (char *)malloc(13);


	std::map<std::string, Attribute> attrs;
	AttribSet *a;
	std::vector<std::string> contexts = NAU->getContextList();
	std::vector<std::string> options;
	Enums::DataType dt;

	for (auto c : contexts) {

		printf("%s\n",c.c_str());
		a = NAU->getAttribs(c);
		attrs = a->getAttributes();
		for (auto attr : attrs) {
		
			dt = attr.second.getType();
			printf("\t%s", attr.second.getName().c_str());
			if (attr.second.getReadOnlyFlag())
				printf("\t(read only)");
			printf("\t%s ", Enums::DataTypeToString[dt].c_str());
			if (dt == Enums::ENUM) {
				printf("\t{");
				options = a->getListString(attr.second.getId());
				for (auto option : options) {
					printf("%s ", option.c_str());
				}
				printf("}");
				int *def = (int *)attr.second.getDefault();
				if (def)
					printf("\tdefault: %s", attr.second.getOptionString(*def).c_str());
				printf("\n");
			}
			else {
				void *min = attr.second.getMin();
				void *max = attr.second.getMax();
				if (min != NULL)
					printf("\tRange: [%s, ", Enums::valueToString(dt, min).c_str());
				else if (max != NULL)
					printf("\tRange: [ , ");
				if (max != NULL)
					printf("%s] ", Enums::valueToString(dt, max).c_str());
				else if (min != NULL)
					printf("] ");
				void *def = attr.second.getDefault();
				if (def)
					printf("\tdefault: %s", Enums::valueToString(dt, def).c_str());
				printf("\n");
			}
		}
	}

	delete NAU;
	_CrtMemCheckpoint(&s2);
	if (_CrtMemDifference(&s3, &s1, &s2))
		_CrtMemDumpStatistics(&s3);

}

*/
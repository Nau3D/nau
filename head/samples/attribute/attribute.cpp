#include "nau.h"

#include <vector>
#include <string>
#include <stdio.h>


using namespace nau;



int main(int argc, char **argv) {

	std::map<std::string, Attribute> attrs;
	AttribSet *a;
	std::vector<std::string> types;
	NAU->getObjTypeList(&types);
	std::vector<std::string> options;
	Enums::DataType dt;

	printf("Object Types:\n");
	for (auto &c : types) {
		printf("%s  ", c.c_str());
	}

	printf("\n<BR><BR>\nObject Type and Components:\n<BR><BR>");
	for (auto &c : types) {
		printf("<B>%s</B>\n<ul>\n", c.c_str());
		a = NAU->getAttribs(c);
		std::map<std::string, std::unique_ptr<Attribute>> &attrs = a->getAttributes();
		for (auto& attr : attrs) {
			dt = attr.second->getType();
			printf("<li><code>%s</code> ", attr.second->getName().c_str());
			if (attr.second->getReadOnlyFlag())
				printf("(read only) - ");
			else
				printf("(read/write) - ");
			printf("%s ", Enums::DataTypeToString[dt].c_str());
			if (dt == Enums::ENUM) {
				printf("{");
				options = a->getListString(attr.second->getId());
				for (auto option : options) {
					printf("%s   ", option.c_str());
				}
				printf("}");
				int *def = (int *)attr.second->getDefault()->getPtr();
				if (def)
					printf(" default: %s", attr.second->getOptionString(*def).c_str());
				
			}
			else {
				void *min = NULL, *max = NULL;
				std::shared_ptr<Data> &aux = attr.second->getMin();
				if (aux)
					min = aux->getPtr();
				std::shared_ptr<Data> &aux2 = attr.second->getMax();
				if (aux2)
					max = aux2->getPtr();
				if (min != NULL)
					printf("[%s, ", Enums::pointerToString(dt, min).c_str());
				else if (max != NULL)
					printf("[ , ");
				if (max != NULL)
					printf("%s]", Enums::pointerToString(dt, max).c_str());
				else if (min != NULL)
					printf("]");
				Data *def = attr.second->getDefault().get();
				if (def)
					printf(" default: %s", Enums::valueToString(dt, def).c_str());
			}
			printf("</li>\n");
		}
		printf("</ul>\n");
	}
	printf("</body></html>>");

	//for (auto c : contexts) {

	//	printf("<B>%s</B>\n",c.c_str());
	//	a = NAU->getAttribs(c);
	//	attrs = a->getAttributes();
	//	printf("<UL>");
	//	for (auto attr : attrs) {
	//		printf("<LI>");
	//		dt = attr.second.getType();
	//		printf("%s", attr.second.getName().c_str());
	//		if (attr.second.getReadOnlyFlag())
	//			printf("\t(read only)");
	//		printf("\t%s ", Enums::DataTypeToString[dt].c_str());
	//		if (dt == Enums::ENUM) {
	//			printf("\t{");
	//			options = a->getListString(attr.second.getId());
	//			for (auto option : options) {
	//				printf("%s ", option.c_str());
	//			}
	//			printf("}");
	//			int *def = (int *)attr.second.getDefault();
	//			if (def)
	//				printf("\tdefault: %s", attr.second.getOptionString(*def).c_str());
	//			printf("\n");
	//		}
	//		else {
	//			void *min = attr.second.getMin();
	//			void *max = attr.second.getMax();
	//			if (min != NULL)
	//				printf("\tRange: [%s, ", Enums::valueToString(dt, min).c_str());
	//			else if (max != NULL)
	//				printf("\tRange: [ , ");
	//			if (max != NULL)
	//				printf("%s] ", Enums::valueToString(dt, max).c_str());
	//			else if (min != NULL)
	//				printf("] ");
	//			void *def = attr.second.getDefault();
	//			if (def)
	//				printf("\tdefault: %s", Enums::valueToString(dt, def).c_str());
	//			printf("\n");
	//			printf("</LI>");
	//		}
	//	}
	//	printf("</UL>");
	//}

}
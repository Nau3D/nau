#include "nau.h"

#include "nau/material/iTexture.h"

#include <vector>
#include <string>
#include <stdio.h>


//#ifdef GLINTERCEPTDEBUG
//#include "../../../GLIntercept\Src\MainLib\ConfigDataExport.h"
//#endif

using namespace nau;



int main(int argc, char **argv) {

	std::map<std::string, Attribute> attrs;
	AttribSet *a;
	std::vector<std::string> contexts = NAU->getContextList();
	std::vector<std::string> options;
	Enums::DataType dt;

	printf("<html><body>");
	for (auto c : contexts) {
		printf("<style type='text / css'>");
		printf("	.tg{ border - collapse:collapse; border - spacing:0; }");
		printf(".tg td{ font - family:Arial, sans - serif; font - size:14px; padding:10px 5px; border - style:solid; border - width:1px; overflow:hidden; word - break:normal; }");
		printf(".tg th{ font - family:Arial, sans - serif; font - size:14px; font - weight:normal; padding:10px 5px; border - style:solid; border - width:1px; overflow:hidden; word - break:normal; }");
		printf("	.tg.tg - yw4l{ text-align:left; }");
		printf(".tg.tg - wr1b{ background - color:#343434; color:#ffffff; vertical - align:top }");
		printf("	</style>");
		printf("	<table class = 'tg'>");
		printf("	<tr>");
		printf("	<th class = 'tg-yw4l' colspan = '5' style='text-align:left; '>");
		printf("<B>%s</B></th>  </tr>\n", c.c_str());
		a = NAU->getAttribs(c);
		std::map<std::string, std::unique_ptr<Attribute>> &attrs = a->getAttributes();
		printf("<tr>");
		for (auto& attr : attrs) {
			
			dt = attr.second->getType();
			printf("<td class='tg - wr1b'>%s</td>", attr.second->getName().c_str());
			if (attr.second->getReadOnlyFlag())
				printf("<td class='tg - wr1b'>yes</td>");
			else
				printf("<td class='tg - wr1b'>no</td>");
			printf("<td class='tg - wr1b'>%s</td>", Enums::DataTypeToString[dt].c_str());
			if (dt == Enums::ENUM) {
				printf("<td class='tg - wr1b'>");
				options = a->getListString(attr.second->getId());
				for (auto option : options) {
					printf("%s<br> ", option.c_str());
				}
				printf("</td>");
				int *def = (int *)attr.second->getDefault()->getPtr();
				if (def)
					printf("<td class='tg - wr1b'>%s</td>", attr.second->getOptionString(*def).c_str());
				printf("\n");
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
					printf("<td class='tg - wr1b'> [%s, ", Enums::pointerToString(dt, min).c_str());
				else if (max != NULL)
					printf("<td class='tg - wr1b'> [ , ");
				if (max != NULL)
					printf("%s]</td> ", Enums::pointerToString(dt, max).c_str());
				else if (min != NULL)
					printf("]</td> ");
				void *def = attr.second->getDefault()->getPtr();
				if (def)
					printf("<td class='tg - wr1b'>%s</td>", Enums::valueToString(dt, def).c_str());
				printf("\n");
				printf("");
			}
			printf("</tr>");
		}
		printf("</table>");
	}
	printf("</body</html>>");

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

	delete NAU;
}
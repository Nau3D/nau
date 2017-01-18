#include "nau/loader/bufferLoader.h"

#include "nau.h"
#include "nau/material/iBuffer.h"
#include "nau/system/file.h"

#include <stdlib.h>

using namespace nau::loader;
using namespace nau::material;
using namespace nau::system;


int BufferLoader::LoadBuffer(IBuffer *aBuffer, std::string &aFilename) {

	File::FixSlashes(aFilename);
	FILE *fp = fopen(aFilename.c_str(),"rt");

	if (fp == NULL) {
		NAU_THROW("unable to open buffer file: %s", aFilename.c_str());
	}

	int bufferSize = aBuffer->getPropui(IBuffer::SIZE);
	char *data = (char *)malloc(bufferSize);
	char *dataPtr = data;
	int count = 0, lines = 0, itensRead = 0, expectedItens;
	std::vector<Enums::DataType> &structure = aBuffer->getStructure();

	bool bufferFull = false;

	while (!feof(fp) && !bufferFull) {

		for (int i = 0; i < structure.size(); ++i) {

			int elementSize = Enums::getSize(structure[i]);
			if (count + elementSize > bufferSize) {
				bufferFull = true;
				break;
			}
			count += elementSize;

			switch (structure[i]) {

			case Enums::INT:
				expectedItens = 1;
				itensRead = fscanf(fp, "%d", (int *)dataPtr);
				dataPtr += elementSize;
				break;
			case Enums::UINT:
				expectedItens = 1;
				itensRead = fscanf(fp, "%u", (unsigned int *)dataPtr);
				dataPtr += elementSize;
				break;
			case Enums::FLOAT:
				expectedItens = 1;
				itensRead = fscanf(fp, "%f", (float *)dataPtr);
				dataPtr += elementSize;
				break;
			case Enums::DOUBLE:
				expectedItens = 1;
				itensRead = fscanf(fp, "%lf", (double *)dataPtr);
				dataPtr += elementSize;
				break;
			default:
				NAU_THROW("Buffer %s structure must contain only INT, UNSIGNED INT, FLOAT or DOUBLE", 
					aBuffer->getLabel().c_str());
			}
			//if (expectedItens != itensRead)
			//	NAU_THROW("Buffer %s\nFile %s\nIncomplete line at or before line %d", 
			//		aBuffer->getLabel().c_str(), aFilename.c_str(), lines);
		}
		lines++;
	}
	aBuffer->setData(count, data);
	fclose(fp);
	return lines;
}
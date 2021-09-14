#include "nau/loader/bufferLoader.h"

#include "nau.h"
#include "nau/slogger.h"
#include "nau/material/iBuffer.h"
#include "nau/system/file.h"

#include <stdlib.h>

using namespace nau::loader;
using namespace nau::material;
using namespace nau::system;


int 
BufferLoader::LoadBuffer(IBuffer *aBuffer, std::string &aFilename) {

	File::FixSlashes(aFilename);
	std::string ext = File::GetExtension(aFilename);
	int bufferSize = aBuffer->getPropui(IBuffer::SIZE);
	char *data = (char *)malloc(bufferSize);
	char* dataPtr = data;
	size_t count = 0;


	if (ext == "nbb") {

		FILE* fp = fopen(aFilename.c_str(), "rb");
		if (fp == NULL) {
			NAU_THROW("unable to open buffer file: %s", aFilename.c_str());
		}

		count = fread(data, sizeof(char), bufferSize, fp);
		std::fclose(fp);

		SLOG("Buffer read (binary - %s:%d buffer size: %d bytes read", aFilename.c_str(), bufferSize, (int)count);
	}
	else {

		FILE* fp = fopen(aFilename.c_str(), "rt");
		if (fp == NULL) {
			NAU_THROW("unable to open buffer file: %s", aFilename.c_str());
		}

		int lines = 0, itensRead = 0, expectedItens;
		std::vector<Enums::DataType>& structure = aBuffer->getStructure();

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
					itensRead = fscanf(fp, "%d", (int*)dataPtr);
					dataPtr += elementSize;
					break;
				case Enums::UINT:
					expectedItens = 1;
					itensRead = fscanf(fp, "%u", (unsigned int*)dataPtr);
					dataPtr += elementSize;
					break;
				case Enums::FLOAT:
					expectedItens = 1;
					itensRead = fscanf(fp, "%f", (float*)dataPtr);
					dataPtr += elementSize;
					break;
				case Enums::DOUBLE:
					expectedItens = 1;
					itensRead = fscanf(fp, "%lf", (double*)dataPtr);
					dataPtr += elementSize;
					break;
				default:
					NAU_THROW("Buffer %s structure must contain only INT, UNSIGNED INT, FLOAT or DOUBLE",
						aBuffer->getLabel().c_str());
				}
			}
			lines++;
		}
		std::fclose(fp);
		SLOG("Buffer read (text) - %s: buffer size %d - %d bytes read", aFilename.c_str(), bufferSize, (int)count);
	}
	aBuffer->setData(count, data);
	return (int)count;
}


int
BufferLoader::SaveBuffer(IBuffer *aBuffer) {

	void *data;

	unsigned int bsize = aBuffer->getPropui(IBuffer::SIZE);
	data = malloc(bsize);
	aBuffer->getData(0, bsize, data);
	std::vector<Enums::DataType> &structure = aBuffer->getStructure();

	char s[200];
	sprintf(s, "%s.%d.txt", aBuffer->getLabel().c_str(), RENDERER->getPropui(IRenderer::FRAME_COUNT));
	std::string sname = nau::system::File::Validate(s);
	File::FixSlashes(sname);

	if (structure.size()) {
		// save as text
		int pointerIndex = 0;
		std::string value;
		FILE *fp = fopen(sname.c_str(), "wb");
		while (pointerIndex < (int)bsize) {
			for (auto t : structure) {
				value = Enums::pointerToString(t, (char *)data + pointerIndex);
				std::fprintf(fp, "%s ", value.c_str());
				pointerIndex += Enums::getSize(t);
			}
			std::fprintf(fp, "\n");
		}
		std::fclose(fp);
	}
	else {
		// save as binary
		FILE *fp = fopen(sname.c_str(), "wb");
		std::fwrite(data, bsize, 1, fp);
		std::fclose(fp);
	}
	return 0;	
}

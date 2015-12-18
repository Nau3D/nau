#include "nau/loader/iTextureLoader.h"

#include "nau.h"
#include "nau/loader/devilTextureLoader.h"
#include "nau/system/file.h"

#include <ctime>

using namespace nau::loader;

ITextureLoader*
ITextureLoader::create (void)
{
	return new DevILTextureLoader();
}


void
ITextureLoader::Save(ITexture *t, FileType ft) {

	if (t == NULL)
		return;

	//ITexImage *ti = RESOURCEMANAGER->createTexImage(t);
	ITexImage *ti = ITexImage::create(t);

	nau::loader::ITextureLoader *loader = nau::loader::ITextureLoader::create();

	char s[200];
	if (ft == PNG)
		sprintf(s,"%s.%d.png", t->getLabel().c_str(), RENDERER->getPropui(IRenderer::FRAME_COUNT));
	else 
		sprintf(s,"%s.%d.hdr", t->getLabel().c_str(), RENDERER->getPropui(IRenderer::FRAME_COUNT));

	std::string sname = nau::system::File::Validate(s);
	loader->save(ti,sname);
	delete loader;
	delete ti;
}


void
ITextureLoader::Save(int width, int height, char *data, std::string filename) {


	if (filename == "") {
		time_t rawtime;
		struct tm * timeinfo;
		char buffer[256];
		char buffer2[256];
	
		time(&rawtime);
		timeinfo = localtime(&rawtime);
		float k = (float)clock();
		strftime(buffer, 80, "%Y-%m-%d_%H-%M-%S", timeinfo);
		sprintf(buffer2, "%s.%f.png", buffer, k);
		filename = std::string(buffer2);
	}

	nau::loader::ITextureLoader *loader = nau::loader::ITextureLoader::create();
	loader->save(width, height, data, filename);

}


void 
ITextureLoader::SaveRaw(ITexture *texture, std::string filename){

	//ITexImage *ti = RESOURCEMANAGER->createTexImage(texture);
	ITexImage *ti = ITexImage::create(texture);

	void *data = ti->getData();

	int w = ti->getWidth();
	int h = ti->getHeight();
	int d = ti->getDepth();
	int n = ti->getNumComponents();
	std::string type = ti->getType();

	// WHEN ADDING MORE TYPES MAKE SURE THEY EXIST IN GLTEXIMAGE.CPP
	float *fData = NULL;
	unsigned int *uiData = NULL;
	unsigned short *usData = NULL;
	unsigned char *ubData = NULL;
	short *sData = NULL;
	char *cData = NULL;
	int *iData = NULL;
	if (type == "FLOAT")
		fData = (float *)data;
	else if (type == "UNSIGNED_BYTE" || type == "UNSIGNED_INT_8_8_8_8_REV")
		ubData = (unsigned char *)data;
	else if (type == "UNSIGNED_SHORT")
		usData = (unsigned short *)data;
	else if (type == "UNSIGNED_INT")
		uiData = (unsigned int *)data;
	else if (type == "SHORT")
		sData = (short *)data;
	else if (type == "BYTE")
		cData = (char *)data;
	else if (type == "INT")
		iData = (int *)data;

	FILE *fp;
	std::string sname = nau::system::File::Validate(filename);

	fp = fopen(sname.c_str(), "wt+");

	for (int g = 0; g < d; ++g) {

		for (int i = 0; i < w; ++i) {

			for (int j = 0; j < h; ++j) {

				for (int k = 0; k < n; ++k) {

					if (type == "FLOAT")
						fprintf(fp, "%f ", fData[(g*w*h + i*h + j)*n + k]);
					else if (type == "UNSIGNED_BYTE" || type == "UNSIGNED_INT_8_8_8_8_REV")
						fprintf(fp, "%u ", ubData[(g*w*h + i*h + j)*n + k]);
					else if (type == "UNSIGNED_SHORT")
						fprintf(fp, "%u ", usData[(g*w*h + i*h + j)*n + k]);
					else if (type == "UNSIGNED_INT")
						fprintf(fp, "%u ", uiData[(g*w*h + i*h + j)*n + k]);
					else if (type == "SHORT")
						fprintf(fp, "%i ", sData[(g*w*h + i*h + j)*n + k]);
					else if (type == "BYTE")
						fprintf(fp, "%i ", cData[(g*w*h + i*h + j)*n + k]);
					else if (type == "INT")
						fprintf(fp, "%d ", iData[(g*w*h + i*h + j)*n + k]);

				}
				fprintf(fp, "; ");
			}
			fprintf(fp, "\n");
		}
	}
	fclose(fp);
	delete ti;
}
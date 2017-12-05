#include "nau/geometry/terrain.h"

#include "nau/math/vec3.h"
#include "nau/geometry/vertexData.h"
#include "nau/material/materialGroup.h"
#include "nau/loader/iTextureLoader.h"

using namespace nau::geometry;
using namespace nau::loader;
using namespace nau::material;
using namespace nau::math;
using namespace nau::render;



Terrain::Terrain(void) : Primitive() {

}


Terrain::~Terrain(void) {

}


std::string
Terrain::getClassName() {

	return "Terrain";
}


void
Terrain::setHeightMap(const std::string &name) {

	m_HeightMap = name;
}


void 
Terrain::build() {

	ITextureLoader *tl = ITextureLoader::create(m_HeightMap);
	// load image but keep its original format
	tl->loadImage(false);
	// convert image to luminance, usefull when image is RGB or RGBA
	tl->convertToFloatLuminance();

	float *data = (float *)tl->getData();
	int width = tl->getWidth();
	int height = tl->getHeight();

	std::shared_ptr<std::vector<VertexData::Attr>> pos =
		std::shared_ptr<std::vector<VertexData::Attr>>(new std::vector<VertexData::Attr>(width*height*sizeof(VertexData::Attr)));
	std::shared_ptr<std::vector<VertexData::Attr>> normal =
		std::shared_ptr<std::vector<VertexData::Attr>>(new std::vector<VertexData::Attr>(width*height*sizeof(VertexData::Attr)));
	std::shared_ptr<std::vector<VertexData::Attr>> texCoord =
		std::shared_ptr<std::vector<VertexData::Attr>>(new std::vector<VertexData::Attr>(width*height*sizeof(VertexData::Attr)));
	for (int h = 0; h < height; ++h) {
		for (int w = 0; w < width; ++w) {
			int index = (h * width + w);
			pos->at(index).x = w - width * 0.5f;
			pos->at(index).z = -height * 0.5f + h;
			pos->at(index).y = data[index];
			pos->at(index).w = 0.0f;

			texCoord->at(index).x = h * 1.0f / width;
			texCoord->at(index).y = w * 1.0f / height;
			texCoord->at(index).z = texCoord->at(index).w = 0.0f;
		}
	}
	float h1, h2, length;
	for (int h = 0; h < height; ++h) {
		for (int w = 0; w < width; ++w) {
			int index = (h * width + w);
			pos->at(index).x = w - width * 0.5f;
			pos->at(index).z = -height * 0.5f + h;
			pos->at(index).y = data[index];
			pos->at(index).w = 1.0f;

			texCoord->at(index).x = h * 1.0f / width;
			texCoord->at(index).y = w * 1.0f / height;
			texCoord->at(index).z = texCoord->at(index).w = 0.0f;

			if (w == 0 || w == width - 1)
				h1 = 0;
			else
				h1 = data[h*width + w + 1] - data[h*width + w - 1];
			if (h == 0 || h == height - 1)
				h2 = 0;
			else
				h2 = data[(h + 1)*width + w] - data[(h - 1)*width + w];

			normal->at(index).x = - 2.0f * h1;
			normal->at(index).y = 4.0f;
			normal->at(index).z = -2.0f * h2;
			normal->at(index).w = 0.0f;
			length = sqrt(normal->at(index).x * normal->at(index).x +
				normal->at(index).y * normal->at(index).y +
				normal->at(index).z*normal->at(index).z);
			normal->at(index).x /= length;
			normal->at(index).y /= length;
			normal->at(index).z /= length;
		}
	}

	std::shared_ptr<VertexData> &vertexData = getVertexData();

	vertexData->setDataFor(VertexData::GetAttribIndex(std::string("position")), pos);
	vertexData->setDataFor(VertexData::GetAttribIndex(std::string("texCoord0")), texCoord);
	vertexData->setDataFor(VertexData::GetAttribIndex(std::string("normal")), normal);

	std::shared_ptr<MaterialGroup> aMatGroup = MaterialGroup::Create(this, "__Light Grey");


	std::shared_ptr<std::vector<unsigned int>> iArr =
		std::shared_ptr<std::vector<unsigned int>>(new std::vector<unsigned int>);
	iArr->resize((width - 1)*(height - 1) * 2 * 3);

	for (int h = 0; h < height - 1; ++h) {

		for (int w = 0; w < width - 1; ++w) {

			int index = (h * (width - 1) + w) * 6;
			// first triangle
			(*iArr)[index] = h * width + w;
			(*iArr)[index + 1] = (h + 1) * width + w;
			(*iArr)[index + 2] = (h + 1) * width + w + 1;

			// second triangle
			(*iArr)[index + 3] = h * width + w;
			(*iArr)[index + 4] = (h + 1) * width + w + 1;
			(*iArr)[index + 5] = h * width + w + 1;
		}
	}

	aMatGroup->setIndexList(iArr);
	addMaterialGroup(aMatGroup);
}



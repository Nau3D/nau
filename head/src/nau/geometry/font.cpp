#include "nau/geometry/font.h"

#include "nau.h"
#include "nau/geometry/mesh.h"
#include "nau/material/materialGroup.h"
#include "nau/scene/sceneObjectFactory.h"

#include <assert.h>


using namespace nau::resource;
using namespace nau::render;
using namespace nau::geometry;


#ifdef _DEBUG
   #ifndef DBG_NEW
      #define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
      #define new DBG_NEW
   #endif
#endif  // _DEBUG

/*

NEED TO SOLVE PROBLEM OF MATERIAL NAME

*/

Font::Font(): 
		mFontName(""),
		mNumChars(0),
		mMaterialName(""),
		mFixedSize(true)
{}


Font::~Font()
{
	mChars.clear();
}



void 
Font::setName(std::string fontName) 
{
	mFontName = fontName;
}

void
Font::setFontHeight(unsigned int height) 
{
	mHeight = height;
}

void
Font::setMaterialName(std::string aMatName) {

	mMaterialName = aMatName;
}

const std::string &
Font::getMaterialName()
{
	return mMaterialName;
}

void 
Font::addChar(char code, int width, float x1, float x2, float y1, float y2, int A, int C)
{
	Char aChar;

	aChar.A = A;
	aChar.C = C;
	aChar.width = width;
	aChar.x1 = x1;
	aChar.x2 = x2;
	aChar.y1 = y1;
	aChar.y2 = y2;
	mChars[code] = aChar;
	mNumChars++;
}

void
Font::createSentenceRenderable(IRenderable &renderable, std::string sentence)
{
	assert(mMaterialName != "");

	size_t aux = sentence.length();
	int size = 0;

	for (unsigned int count = 0; count < aux; count++) {
	
		// if char exists in the font definition
		if (mChars.count(sentence[count])) 
			size++;
	}

	assert(size);

	// need to clear previous mesh 
	//Mesh *renderable = (Mesh *)RESOURCEMANAGER->createRenderable("Mesh", sentence, "Sentence");
	//renderable->setDrawingPrimitive(nau::render::IRenderer::TRIANGLES);

	std::vector<VertexData::Attr> *vertices = new std::vector<VertexData::Attr>(size*6);
	std::vector<VertexData::Attr> *texCoords = new std::vector<VertexData::Attr>(size*6);
	std::vector<VertexData::Attr> *normals = new std::vector<VertexData::Attr>(size*6);

	int i = 0;
	float hDisp = 0.0f, vDisp = 0.0f;

	for (size_t count = 0; count < aux; count++) {
	
		// get char at position count
		char c = sentence[count];
		if (c == ' ') {
			if (mFixedSize)
				hDisp += mChars[c].C + mChars[c].A; 
			else
				hDisp += mChars[c].C;	
		}
		// if char exists in the font definition
		else if (mChars.count(c)) {
			vertices->at(6*i  ).set(hDisp, vDisp + mHeight, 0.0f, 1.0f);
			vertices->at(6*i+1).set(hDisp + mChars[c].width, vDisp, 0.0f, 1.0f);
			vertices->at(6*i+2).set(hDisp, vDisp, 0.0f, 1.0f);			
			vertices->at(6*i+3).set(hDisp + mChars[c].width, vDisp, 0.0f, 1.0f);			
			vertices->at(6*i+4).set(hDisp, vDisp + mHeight,0.0f, 1.0f);			
			vertices->at(6*i+5).set(hDisp + mChars[c].width, vDisp + mHeight, 0.0f, 1.0f);		

			normals->at(6*i    ).set(0,0,1);
			normals->at(6*i+1  ).set(0,0,1);
			normals->at(6*i+2  ).set(0,0,1);
			normals->at(6*i+3  ).set(0,0,1);
			normals->at(6*i+4  ).set(0,0,1);
			normals->at(6*i+5  ).set(0,0,1);

			texCoords->at(6*i  ).set(mChars[c].x1, 1-mChars[c].y2, 0.0f, 1.0f);
			texCoords->at(6*i+1).set(mChars[c].x2, 1-mChars[c].y1, 0.0f, 1.0f);
			texCoords->at(6*i+2).set(mChars[c].x1, 1-mChars[c].y1, 0.0f, 1.0f);
			texCoords->at(6*i+3).set(mChars[c].x2, 1-mChars[c].y1, 0.0f, 1.0f);
			texCoords->at(6*i+4).set(mChars[c].x1, 1-mChars[c].y2, 0.0f, 1.0f);
			texCoords->at(6*i+5).set(mChars[c].x2, 1-mChars[c].y2, 0.0f, 1.0f);

			if (mFixedSize)
				hDisp += mChars[c].C + mChars[c].A; 
			else
				hDisp += mChars[c].C;
			i++;
		}
		// newline
		else if (c == '\n') {
			vDisp += mHeight;
			hDisp = 0.0f;
		}
	}

	VertexData &vertexData = renderable.getVertexData();
	vertexData.setDataFor (VertexData::GetAttribIndex(std::string("position")), vertices);
	vertexData.setDataFor (VertexData::GetAttribIndex(std::string("normal")), normals);
	vertexData.setDataFor (VertexData::GetAttribIndex(std::string("texCoord0")), texCoords);

	std::shared_ptr<std::vector<unsigned int>> indices = 
		std::shared_ptr<std::vector<unsigned int>>(new std::vector<unsigned int>(size*6));
	for (int j = 0; j < size*6 ; j++)
		indices->push_back(j);

	MaterialGroup* auxMG;

	std::vector<MaterialGroup *> aMatG = renderable.getMaterialGroups();
	if (aMatG.size()) {
		auxMG = (MaterialGroup *)aMatG[0];
		auxMG->setIndexList (indices);
	}
	else {
		auxMG = MaterialGroup::Create(&renderable, mMaterialName);
//		auxMG->setMaterialName(mMaterialName);
//		auxMG->setParent(&renderable);
		auxMG->setIndexList (indices);
		renderable.addMaterialGroup(auxMG);
	}
}


const std::string &
Font::getFontName()
{
	return mFontName;
}

bool
Font::getFixedSize()
{
	return(mFixedSize);
}

void
Font::setFixedSize(bool f)
{
	mFixedSize = f;
}
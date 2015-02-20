#include "nau/render/irenderer.h"

#include "nau/math/matrix.h"

using namespace nau;
using namespace nau::math;

std::map<std::string, IRenderable::DrawPrimitive> IRenderer::PrimitiveTypes = {
	{ "TRIANGLES", IRenderable::TRIANGLES },
	{ "TRIANGLE_STRIP", IRenderable::TRIANGLE_STRIP },
	{ "TRIANGLE_FAN", IRenderable::TRIANGLE_FAN },
	{ "LINES", IRenderable::LINES },
	{ "LINE_LOOP", IRenderable::LINE_LOOP },
	{ "POINTS", IRenderable::POINTS },
	{ "TRIANGLES_ADJACENCY", IRenderable::TRIANGLES_ADJACENCY }
#if NAU_OPENGL_VERSION >= 400	
	, { "PATCHES", IRenderable::PATCHES }
#endif
};

int IRenderer::MaxTextureUnits;
int IRenderer::MaxColorAttachments;

AttribSet IRenderer::MatrixAttribs;
bool IRenderer::Inited = Init();

bool
IRenderer::Init() {
	MatrixAttribs.add(Attribute(PROJECTION,	"PROJECTION", Enums::DataType::MAT4, true, new mat4()));
	MatrixAttribs.add(Attribute(MODEL, "MODEL", Enums::DataType::MAT4, true, new mat4()));
	MatrixAttribs.add(Attribute(VIEW, "VIEW", Enums::DataType::MAT4, true, new mat4()));
	MatrixAttribs.add(Attribute(TEXTURE, "TEXTURE", Enums::DataType::MAT4, true, new mat4()));
	MatrixAttribs.add(Attribute(VIEW_MODEL, "VIEW_MODEL", Enums::DataType::MAT4, true, new mat4()));
	MatrixAttribs.add(Attribute(PROJECTION_VIEW_MODEL, "PROJECTION_VIEW_MODEL", Enums::DataType::MAT4, true, new mat4()));
	MatrixAttribs.add(Attribute(PROJECTION_VIEW, "PROJECTION_VIEW", Enums::DataType::MAT4, true, new mat4()));
	MatrixAttribs.add(Attribute(TS05_PVM, "TS05_PVM", Enums::DataType::MAT4, true, new mat4()));
	MatrixAttribs.add(Attribute(NORMAL, "NORMAL", Enums::DataType::MAT3, true, new mat3()));

	// MOVE TO irenderable.h
	//Attribs.add(Attribute(DRAW_PRIMITIVE, "DRAW_PRIMITIVE", Enums::DataType::ENUM, true));
	//Attribs.listAdd("DRAW_PRIMITIVE", "TRIANGLES", TRIANGLES);
	//Attribs.listAdd("DRAW_PRIMITIVE", "TRIANGLE_STRIP", TRIANGLE_STRIP);
	//Attribs.listAdd("DRAW_PRIMITIVE", "TRIANGLE_FAN", TRIANGLE_FAN);
	//Attribs.listAdd("DRAW_PRIMITIVE", "LINES", LINES);
	//Attribs.listAdd("DRAW_PRIMITIVE", "LINE_LOOP", LINE_LOOP);
	//Attribs.listAdd("DRAW_PRIMITIVE", "POINTS", POINTS);
	//Attribs.listAdd("DRAW_PRIMITIVE", "TRIANGLES_ADJACENCY", TRIANGLES_ADJACENCY);
	//Attribs.listAdd("DRAW_PRIMITIVE", "PATCHES", PATCH);
	return true;
}


void 
IRenderer::setPrope(EnumProperty prop, int value){
	assert(MatrixAttribs.getName(prop, Enums::DataType::ENUM) != "" && "invalid option for an enum") ;
	m_EnumProps[prop] = value;
}


void 
IRenderer::setProp(int prop, Enums::DataType type, void *value) {

	switch (type) {
	case Enums::ENUM:
		// prop must exist
		assert(m_EnumProps.count(prop) && "invalid property");
		// value must be in the list of valid enums
		assert(MatrixAttribs.getName(prop, Enums::DataType::ENUM) != "" && "invalid value");
		m_EnumProps[prop] = *(int *)value;
		break;
	default:
		assert(false && "invalid enum type");
	}
}


// ATOMIC COUNTERS

#if NAU_OPENGL_VERSION >= 400

void
IRenderer::addAtomic(std::string buffer, unsigned int offset, std::string name) {

	std::pair<std::string, unsigned int> p = std::pair<std::string, unsigned int>(buffer, offset);
	if (m_AtomicLabels.count(p) == 0) {
		++m_AtomicCount;
		m_AtomicBufferPrepared = false;
		m_AtomicLabels[p] = name;
		//if (id > m_AtomicMaxID)
		//	m_AtomicMaxID = id;
	}
}


//int
//IRenderer::getAtomicID(std::string name) {
//
//	for (auto atom: m_AtomicLabels) {
//
//		if (atom.second == name)
//			return (atom.first);
//	}
//	return -1;
//
//}

#endif

// -------------------



const std::string IRenderer::MatrixTypeString[] = {"PROJECTION", "MODEL", 
												"VIEW", "TEXTURE",
												"VIEW_MODEL", "PROJECTION_VIEW_MODEL", 
												"PROJECTION_VIEW", "TS05_PVM", "NORMAL"};




void 
IRenderer::getPropId(std::string &s, int *id){

	// value returned in case of an invalid string
	*id = -1;

	for (int i = 0; i < COUNT_MATRIXTYPE; i++) {

		if (s == MatrixTypeString[i]) {
		
			*id = i;
			return;
		}
	}
}


const std::string &
IRenderer::getPropMatrixTypeString(MatrixType mode) 
{
	return MatrixTypeString[mode];
}



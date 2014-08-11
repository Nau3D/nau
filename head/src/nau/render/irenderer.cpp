#include <nau/render/irenderer.h>

using namespace nau;

int IRenderer::MaxTextureUnits;
int IRenderer::MaxColorAttachments;

AttribSet IRenderer::Attribs;
bool IRenderer::Inited = Init();

bool
IRenderer::Init() {
	//Attribs.add(Attribute(MATRIX, "MATRIX", Enums::DataType::ENUM, true, new mat4()));
	//Attribs.listAdd("MATRIX", "PROJECTION", PROJECTION);
	//Attribs.listAdd("MATRIX", "MODEL", MODEL);
	//Attribs.listAdd("MATRIX", "VIEW", VIEW);
	//Attribs.listAdd("MATRIX", "TEXTURE", TEXTURE);
	//Attribs.listAdd("MATRIX", "VIEW_MODEL", VIEW_MODEL);
	//Attribs.listAdd("MATRIX", "PROJECTION_VIEW_MODEL", PROJECTION_VIEW_MODEL);
	//Attribs.listAdd("MATRIX", "PROJECTION_VIEW", PROJECTION_VIEW);
	//Attribs.listAdd("MATRIX", "TS05_PVM", TS05_PVM);
	//Attribs.listAdd("MATRIX", "NORMAL", NORMAL);
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
	assert(Attribs.getName(prop, Enums::DataType::ENUM) != "" && "invalid option for an enum") ;
	m_EnumProps[prop] = value;
}


void 
IRenderer::setProp(int prop, Enums::DataType type, void *value) {

	switch (type) {
	case Enums::ENUM:
		// prop must exist
		assert(m_EnumProps.count(prop) && "invalid property");
		// value must be in the list of valid enums
		assert(Attribs.getName(prop, Enums::DataType::ENUM) != "" && "invalid value");
		m_EnumProps[prop] = *(int *)value;
		break;
	default:
		assert(false && "invalid enum type");
	}
}


// ATOMIC COUNTERS


void
IRenderer::addAtomic(unsigned int id, std::string name) {

	if (m_AtomicLabels.count(id) == 0) {
		++m_AtomicCount;
		m_AtomicBufferPrepared = false;
		m_AtomicLabels[id] = name;
		if (id > m_AtomicMaxID)
			m_AtomicMaxID = id;
	}
}



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


int 
IRenderer::translateStringToStencilOp(std::string s) {

	if (s == "KEEP")
		return(KEEP);
	if (s == "ZERO")
		return(ZERO);
	if (s == "REPLACE")
		return(REPLACE);
	if (s == "INCR")
		return(INCR);
	if (s == "INCR_WRAP")
		return(INCR_WRAP);
	if (s == "DECR")
		return(DECR);
	if (s == "DECR_WRAP")
		return(DECR_WRAP);
	if (s == "INVERT")
		return(INVERT);

	return(-1);
}

int 
IRenderer::translateStringToStencilFunc(std::string s) {

	if (s == "NEVER")
		return(NEVER);
	if (s == "ALWAYS")
		return(ALWAYS);
	if (s == "LESS")
		return(LESS);
	if (s == "LEQUAL")
		return(LEQUAL);
	if (s == "GEQUAL")
		return(GEQUAL);
	if (s == "GREATER")
		return(GREATER);
	if (s == "EQUAL")
		return(EQUAL);
	if (s == "NOT_EQUAL")
		return(NOT_EQUAL);

	return(-1);
}
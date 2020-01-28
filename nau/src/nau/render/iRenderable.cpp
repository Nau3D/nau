#include "nau/render/iRenderable.h"

#include "nau.h"

using namespace nau::render;


bool
IRenderable::Init() {

//	// ENUM
//	Attribs.add(Attribute(PRIMITIVE_TYPE, "PRIMITIVE_TYPE", Enums::ENUM, false, new int(TRIANGLES)));
//	Attribs.listAdd("PRIMITIVE_TYPE", "TRIANGLES", TRIANGLES);
//	Attribs.listAdd("PRIMITIVE_TYPE", "TRIANGLE_STRIP", TRIANGLE_STRIP);
//	Attribs.listAdd("PRIMITIVE_TYPE", "TRIANGLE_FAN", TRIANGLE_FAN);
//	Attribs.listAdd("PRIMITIVE_TYPE", "LINES", LINES);
//	Attribs.listAdd("PRIMITIVE_TYPE", "LINE_LOOP", LINE_LOOP);
//	Attribs.listAdd("PRIMITIVE_TYPE", "POINTS", POINTS);
//	Attribs.listAdd("PRIMITIVE_TYPE", "TRIANGLES_ADJACENCY", TRIANGLES_ADJACENCY);
//#if NAU_OPENGL_VERSION >= 400
//	Attribs.listAdd("PRIMITIVE_TYPE", "PATCHES", PATCHES);
//#endif

//#ifndef _WINDLL
//	NAU->registerAttributes("RENDERABLE", &Attribs);
//#endif

	return true;
}


AttribSet IRenderable::Attribs;
bool IRenderable::Inited = Init();


IRenderable::IRenderable(): m_DrawPrimitive(TRIANGLES) {

	registerAndInitArrays(Attribs);
}


std::string
IRenderable::getDrawingPrimitiveString() {

	switch (m_DrawPrimitive) {
	case TRIANGLES: return "Triangles";
	case TRIANGLE_STRIP: return "Triangle strip";
	case TRIANGLE_FAN: return "Traingle fan";
	case LINES: return "Lines";
	case LINE_LOOP: return "Line loop";
	case POINTS: return "Points";
	case TRIANGLES_ADJACENCY: return "Triangle adjacency";
	case PATCHES: return "Patches";
	default: return "unsupported";

	}
}
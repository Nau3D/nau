#include "nau/render/irenderer.h"

#include "nau.h"
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

AttribSet IRenderer::Attribs;
bool IRenderer::Inited = Init();

bool
IRenderer::Init() {
	// MAT4
	Attribs.add(Attribute(PROJECTION,	"PROJECTION", Enums::DataType::MAT4, true, new mat4()));
	Attribs.add(Attribute(MODEL, "MODEL", Enums::DataType::MAT4, true, new mat4()));
	Attribs.add(Attribute(VIEW, "VIEW", Enums::DataType::MAT4, true, new mat4()));
	Attribs.add(Attribute(TEXTURE, "TEXTURE", Enums::DataType::MAT4, true, new mat4()));
	Attribs.add(Attribute(VIEW_MODEL, "VIEW_MODEL", Enums::DataType::MAT4, true, new mat4()));
	Attribs.add(Attribute(PROJECTION_VIEW_MODEL, "PROJECTION_VIEW_MODEL", Enums::DataType::MAT4, true, new mat4()));
	Attribs.add(Attribute(PROJECTION_VIEW, "PROJECTION_VIEW", Enums::DataType::MAT4, true, new mat4()));
	Attribs.add(Attribute(TS05_PVM, "TS05_PVM", Enums::DataType::MAT4, true, new mat4()));

	// MAT3
	Attribs.add(Attribute(NORMAL, "NORMAL", Enums::DataType::MAT3, true, new mat3()));

	// INT
	Attribs.add(Attribute(TEXTURE_COUNT, "TEXTURE_COUNT", Enums::DataType::INT, true, new int(0)));
	Attribs.add(Attribute(LIGHT_COUNT, "LIGHT_COUNT", Enums::DataType::INT, true, new int(0)));

	// VEC2
	Attribs.add(Attribute(MOUSE_CLICK, "MOUSE_CLICK", Enums::DataType::IVEC2, false, new ivec2(0)));

	NAU->registerAttributes("RENDERER", &Attribs);
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



// ATOMIC COUNTERS

#if NAU_OPENGL_VERSION >= 400

void
IRenderer::addAtomic(std::string buffer, unsigned int offset, std::string name) {

	std::pair<std::string, unsigned int> p = std::pair<std::string, unsigned int>(buffer, offset);

	if (m_AtomicLabels.count(p) == 0) {

		++m_AtomicCount;
		m_AtomicBufferPrepared = false;
		m_AtomicLabels[p] = name;
	}
	else {

		assert(false && "Adding an atomic that already exists");
	}

}



#endif


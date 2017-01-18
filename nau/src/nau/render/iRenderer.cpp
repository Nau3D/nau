#include "nau/render/iRenderer.h"

#include "nau.h"
#include "nau/math/number.h"
#include "nau/math/matrix.h"
#include "nau/render/iAPISupport.h"

#include <ctime>

using namespace nau;
using namespace nau::math;

std::map<std::string, IRenderable::DrawPrimitive> IRenderer::PrimitiveTypes = {
	{ "TRIANGLES", IRenderable::TRIANGLES },
	{ "TRIANGLE_STRIP", IRenderable::TRIANGLE_STRIP },
	{ "TRIANGLE_FAN", IRenderable::TRIANGLE_FAN },
	{ "LINES", IRenderable::LINES },
	{ "LINE_LOOP", IRenderable::LINE_LOOP },
	{ "POINTS", IRenderable::POINTS },
	{ "TRIANGLES_ADJACENCY", IRenderable::TRIANGLES_ADJACENCY },
	{ "PATCHES", IRenderable::PATCHES }
};

int IRenderer::MaxTextureUnits;
int IRenderer::MaxColorAttachments;

AttribSet IRenderer::Attribs;
bool IRenderer::Inited = Init();


AttribSet &
IRenderer::GetAttribs() {
	return Attribs;
}


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
	Attribs.add(Attribute(TEXTURE_COUNT, "TEXTURE_COUNT", Enums::DataType::INT, true, new NauInt(0)));
	Attribs.add(Attribute(LIGHT_COUNT, "LIGHT_COUNT", Enums::DataType::INT, true, new NauInt(0)));

	// VEC2
	Attribs.add(Attribute(MOUSE_LEFT_CLICK, "MOUSE_LEFT_CLICK", Enums::DataType::IVEC2, false, new ivec2(0)));
	Attribs.add(Attribute(MOUSE_MIDDLE_CLICK, "MOUSE_MIDDLE_CLICK", Enums::DataType::IVEC2, false, new ivec2(0)));
	Attribs.add(Attribute(MOUSE_RIGHT_CLICK, "MOUSE_RIGHT_CLICK", Enums::DataType::IVEC2, false, new ivec2(0)));

	//UINT		
	Attribs.add(Attribute(INSTANCE_COUNT, "INSTANCE_COUNT", Enums::DataType::UINT, false, new NauUInt(0)));
	Attribs.add(Attribute(BUFFER_DRAW_INDIRECT, "BUFFER_DRAW_INDIRECT", Enums::DataType::UINT, false, new NauUInt(0)));
	Attribs.add(Attribute(FRAME_COUNT, "FRAME_COUNT", Enums::DataType::UINT, true, new NauUInt(0)));

	// BOOL
	Attribs.add(Attribute(DEBUG_DRAW_CALL, "DEBUG_DRAW_CALL", Enums::DataType::BOOL, true, new NauInt(false)));

	// FLOAT
	Attribs.add(Attribute(TIMER, "TIMER", Enums::DataType::FLOAT, true, new NauFloat(0.0f)));

	//#ifndef _WINDLL
	NAU->registerAttributes("RENDERER", &Attribs);
	//#endif

	return true;
}


// SHADER DEBUG INFO

nau::util::Tree *
IRenderer::getShaderDebugTree() {

	return &m_ShaderDebugTree;
}

// API SUPPORT

bool
IRenderer::primitiveTypeSupport(std::string primitive) {

	IAPISupport *sup = IAPISupport::GetInstance();

	if (primitive == "PATCHES" && !sup->apiSupport(IAPISupport::TESSELATION_SHADERS))
		return false;
	else if (PrimitiveTypes.count(primitive))
		return true;
	else
		return false;
}


// ATOMIC COUNTERS

void
IRenderer::addAtomic(std::string buffer, unsigned int offset, std::string name) {

	if (APISupport->apiSupport(IAPISupport::BUFFER_ATOMICS)) {
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
	else
		assert(false && "atomics not supported");
}


float 
IRenderer::getPropf(FloatProperty prop) {

	switch (prop) {
	case TIMER:
		m_FloatProps[TIMER] = (float)clock();// *1000.0 / CLOCKS_PER_SEC;
		return m_FloatProps[TIMER];

	default:return(AttributeValues::getPropf(prop));
	}

}


void 
IRenderer::setPropb(BoolProperty prop, bool value) {

	switch (prop) {

		case IRenderer::DEBUG_DRAW_CALL:
			if (value == true && m_BoolProps[prop] == false) {
				m_BoolProps[prop] = value;
				m_ShaderDebugTree.clear();
			}
			else if (value == false)
				m_BoolProps[prop] = false;
			break;
		default:
			AttributeValues::setPropb(prop, value);
	}
}

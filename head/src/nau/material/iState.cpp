#include "nau/material/iState.h"
#include "nau/config.h"

#ifdef NAU_OPENGL
#include "nau/render/opengl/glState.h"	
#elif NAU_DIRECTX
#include "nau/render/dx/dxstate.h"
#endif

#include "nau.h"

using namespace nau::render;
using namespace nau;

bool
IState::Init() {

	// BOOL
	Attribs.add(Attribute(BLEND, "BLEND", Enums::DataType::BOOL, false, new NauInt(false)));
	Attribs.add(Attribute(DEPTH_TEST, "DEPTH_TEST", Enums::DataType::BOOL, false, new NauInt(true)));
	Attribs.add(Attribute(CULL_FACE, "CULL_FACE", Enums::DataType::BOOL, false, new NauInt(true)));
	Attribs.add(Attribute(DEPTH_MASK, "DEPTH_MASK", Enums::DataType::BOOL, false, new NauInt(true)));

	// ENUM
	Attribs.add(Attribute(DEPTH_FUNC, "DEPTH_FUNC", Enums::DataType::ENUM, false));
	Attribs.add(Attribute(CULL_TYPE, "CULL_TYPE", Enums::DataType::ENUM, false));
	Attribs.add(Attribute(ORDER_TYPE, "ORDER_TYPE", Enums::DataType::ENUM, false));
	Attribs.add(Attribute(BLEND_SRC, "BLEND_SRC", Enums::DataType::ENUM, false));
	Attribs.add(Attribute(BLEND_DST, "BLEND_DST", Enums::DataType::ENUM, false));
	Attribs.add(Attribute(BLEND_EQUATION, "BLEND_EQUATION", Enums::DataType::ENUM, false));

	// INT
	Attribs.add(Attribute(ORDER, "ORDER", Enums::DataType::INT, false, new NauInt(0)));

	// FLOAT4
	Attribs.add(Attribute(BLEND_COLOR, "BLEND_COLOR", Enums::DataType::VEC4, false, new vec4(0.0f, 0.0f, 0.0f, 0.0f)));

	// BOOL4
	Attribs.add(Attribute(COLOR_MASK_B4, "COLOR_MASK_B4", Enums::DataType::BVEC4, false, new bvec4(true, true, true, true)));

#ifndef _WINDLL
	NAU->registerAttributes("STATE", &Attribs);
#endif

	return true;
}


AttribSet IState::Attribs;
bool IState::Inited = Init();


IState*
IState::create() {

#ifdef NAU_OPENGL
	return new GLState ;
#elif NAU_DIRECTX
	return new DXState;
#endif
}


IState::IState():
		m_defColor(0.0f, 0.0f, 0.0f, 1.0f)
{
	registerAndInitArrays(Attribs);
	m_Name = "default";
}

			
IState* 
IState::clone() {

	IState *res = IState::create();
	res->copy(this);

	return res;
}


void 
IState::setDefault() {

	initArrays();
	m_Name = "default";
}

//
// GET methods
//


std::string
IState::getName() {

	return(m_Name);
}


void 
IState::setName(std::string aName) {

	m_Name = aName;
}



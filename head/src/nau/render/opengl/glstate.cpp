#include <nau/render/opengl/glstate.h>
#include <nau/config.h>

#include <GL/glew.h>

using namespace nau::render;

bool
GlState::Init() {

	// ENUM
	Attribs.setDefault("DEPTH_FUNC", new int(GL_LESS));
	Attribs.listAdd("DEPTH_FUNC", "LESS", GL_LESS);
	Attribs.listAdd("DEPTH_FUNC", "NEVER", GL_NEVER);
	Attribs.listAdd("DEPTH_FUNC", "ALWAYS", GL_ALWAYS);
	Attribs.listAdd("DEPTH_FUNC", "LEQUAL", GL_LEQUAL);
	Attribs.listAdd("DEPTH_FUNC", "EQUAL", GL_EQUAL);
	Attribs.listAdd("DEPTH_FUNC", "GEQUAL", GL_GEQUAL);
	Attribs.listAdd("DEPTH_FUNC", "GREATER", GL_GREATER);
	Attribs.listAdd("DEPTH_FUNC", "NOT_EQUAL", GL_NOTEQUAL);

	Attribs.setDefault("CULL_TYPE", new int(GL_BACK));
	Attribs.listAdd("CULL_TYPE", "FRONT", GL_FRONT);
	Attribs.listAdd("CULL_TYPE", "BACK", GL_BACK);
	Attribs.listAdd("CULL_TYPE", "FRONT_AND_BACK", GL_FRONT_AND_BACK);
	
	Attribs.setDefault("ORDER_TYPE", new int(FRONT_TO_BACK));
	Attribs.listAdd("ORDER_TYPE", "FRONT_TO_BACK", FRONT_TO_BACK);
	Attribs.listAdd("ORDER_TYPE", "BACK_TO_FRONT", BACK_TO_FRONT);

	Attribs.setDefault("BLEND_SRC", new int(GL_ONE));
	Attribs.listAdd("BLEND_SRC", "ZERO", GL_ZERO);
	Attribs.listAdd("BLEND_SRC", "ONE", GL_ONE);
	Attribs.listAdd("BLEND_SRC", "SRC_COLOR", GL_SRC_COLOR);
	Attribs.listAdd("BLEND_SRC", "ONE_MINUS_SRC_COLOR", GL_ONE_MINUS_SRC_COLOR);
	Attribs.listAdd("BLEND_SRC", "DST_COLOR", GL_DST_COLOR);
	Attribs.listAdd("BLEND_SRC", "ONE_MINUS_DST_COLOR", GL_ONE_MINUS_DST_COLOR);
	Attribs.listAdd("BLEND_SRC", "SRC_ALPHA", GL_SRC_ALPHA);
	Attribs.listAdd("BLEND_SRC", "ONE_MINUS_SRC_ALPHA", GL_ONE_MINUS_SRC_ALPHA);
	Attribs.listAdd("BLEND_SRC", "DST_ALPHA", GL_DST_ALPHA);
	Attribs.listAdd("BLEND_SRC", "ONE_MINUS_DST_ALPHA", GL_ONE_MINUS_DST_ALPHA);
	Attribs.listAdd("BLEND_SRC", "SRC_ALPHA_SATURATE", GL_SRC_ALPHA_SATURATE);
	Attribs.listAdd("BLEND_SRC", "CONSTANT_COLOR", GL_CONSTANT_COLOR);
	Attribs.listAdd("BLEND_SRC", "ONE_MINUS_CONSTANT_COLOR", GL_ONE_MINUS_CONSTANT_COLOR);
	Attribs.listAdd("BLEND_SRC", "CONSTANT_ALPHA", GL_CONSTANT_ALPHA);
	Attribs.listAdd("BLEND_SRC", "ONE_MINUS_CONSTANT_ALPHA", GL_ONE_MINUS_CONSTANT_ALPHA);

	Attribs.setDefault("BLEND_DST", new int(GL_ZERO));
	Attribs.listAdd("BLEND_DST", "ZERO", GL_ZERO);
	Attribs.listAdd("BLEND_DST", "ONE", GL_ONE);
	Attribs.listAdd("BLEND_DST", "SRC_COLOR", GL_SRC_COLOR);
	Attribs.listAdd("BLEND_DST", "ONE_MINUS_SRC_COLOR", GL_ONE_MINUS_SRC_COLOR);
	Attribs.listAdd("BLEND_DST", "DST_COLOR", GL_DST_COLOR);
	Attribs.listAdd("BLEND_DST", "ONE_MINUS_DST_COLOR", GL_ONE_MINUS_DST_COLOR);
	Attribs.listAdd("BLEND_DST", "SRC_ALPHA", GL_SRC_ALPHA);
	Attribs.listAdd("BLEND_DST", "ONE_MINUS_SRC_ALPHA", GL_ONE_MINUS_SRC_ALPHA);
	Attribs.listAdd("BLEND_DST", "DST_ALPHA", GL_DST_ALPHA);
	Attribs.listAdd("BLEND_DST", "ONE_MINUS_DST_ALPHA", GL_ONE_MINUS_DST_ALPHA);
	Attribs.listAdd("BLEND_DST", "SRC_ALPHA_SATURATE", GL_SRC_ALPHA_SATURATE);
	Attribs.listAdd("BLEND_DST", "CONSTANT_COLOR", GL_CONSTANT_COLOR);
	Attribs.listAdd("BLEND_DST", "ONE_MINUS_CONSTANT_COLOR", GL_ONE_MINUS_CONSTANT_COLOR);
	Attribs.listAdd("BLEND_DST", "CONSTANT_ALPHA", GL_CONSTANT_ALPHA);
	Attribs.listAdd("BLEND_DST", "ONE_MINUS_CONSTANT_ALPHA", GL_ONE_MINUS_CONSTANT_ALPHA);

	Attribs.setDefault("BLEND_EQUATION", new int(GL_FUNC_ADD));
	Attribs.listAdd("BLEND_EQUATION", "ADD", GL_FUNC_ADD);
	Attribs.listAdd("BLEND_EQUATION", "SUBTRACT", GL_FUNC_SUBTRACT);
	Attribs.listAdd("BLEND_EQUATION", "REVERSE_SUBTRACT", GL_FUNC_REVERSE_SUBTRACT);
	Attribs.listAdd("BLEND_EQUATION", "MIN", GL_MIN);
	Attribs.listAdd("BLEND_EQUATION", "MAX", GL_MAX);

	// NON CORE SETTINGS
#if NAU_CORE_OPENGL == 0
	Attribs.setDefault("ALPHA_FUNC", new int(GL_ALWAYS));
	Attribs.listAdd("ALPHA_FUNC", "LESS", GL_LESS);
	Attribs.listAdd("ALPHA_FUNC", "NEVER", GL_NEVER);
	Attribs.listAdd("ALPHA_FUNC", "ALWAYS", GL_ALWAYS);
	Attribs.listAdd("ALPHA_FUNC", "LEQUAL", GL_LEQUAL);
	Attribs.listAdd("ALPHA_FUNC", "EQUAL", GL_EQUAL);
	Attribs.listAdd("ALPHA_FUNC", "GEQUAL", GL_GEQUAL);
	Attribs.listAdd("ALPHA_FUNC", "GREATER", GL_GREATER);
	Attribs.listAdd("ALPHA_FUNC", "NOT_EQUAL", GL_NOTEQUAL);

	Attribs.setDefault("FOG_MODE", new int(GL_EXP));
	Attribs.listAdd("FOG_MODE", "LINEAR", GL_LINEAR);
	Attribs.listAdd("FOG_MODE", "EXP", GL_EXP);
	Attribs.listAdd("FOG_MODE", "EXP2", GL_EXP2);

	Attribs.setDefault("FOG_COORD_SRC", new int(GL_FRAGMENT_DEPTH));
	Attribs.listAdd("FOG_COORD_SRC", "FOG_COORD", GL_FOG_COORD);
	Attribs.listAdd("FOG_COORD_SRC", "FRAGMENT_DEPTH", GL_FRAGMENT_DEPTH);
#endif

	return true;
}

bool GlState::Inited = Init();


//
//Constructor & Destructor
//
GlState::GlState(): IState()   {

}


GlState::~GlState() {

}


 //Set Full States

void
GlState::set() {

	std::map< int, int>::iterator iterInt;
	iterInt = m_IntProps.begin();

	for ( ; iterInt != m_IntProps.end(); ++iterInt) {
	
		switch(iterInt->first) {
			case ORDER: break;
		}
	}


	std::map< int, bool>::iterator iterBool;
	iterBool = m_EnableProps.begin();
	for ( ; iterBool != m_EnableProps.end(); ++iterBool) {
	
		switch(iterBool->first) {
#if NAU_CORE_OPENGL == 0
			case FOG:	
						if (iterBool->second)
							glEnable(GL_FOG);
						else
							glDisable(GL_FOG);
					break;
			case ALPHA_TEST: 
						if (iterBool->second)
							glEnable(GL_ALPHA_TEST);
						else
							glDisable(GL_ALPHA_TEST);
					break;
#endif
			case BLEND: 
						if (iterBool->second)
							glEnable(GL_BLEND);
						else
							glDisable(GL_BLEND);
					break;
			case DEPTH_TEST:
						if (iterBool->second)
							glEnable(GL_DEPTH_TEST);
						else
							glDisable(GL_DEPTH_TEST);
					break;
			case DEPTH_MASK:
						glDepthMask(iterBool->second);
						break;
			case CULL_FACE: 
						if (iterBool->second)
							glEnable(GL_CULL_FACE);
						else
							glDisable(GL_CULL_FACE);
					break;
			case COLOR_MASK: 
					break;
		}
	}

#if NAU_CORE_OPENGL  == 0
	std::map< int, float>::iterator iterFloat;
	iterFloat = m_FloatProps.begin();
	for ( ; iterFloat != m_FloatProps.end(); ++iterFloat) {
	
		switch(iterFloat->first) {
		case FOG_START: 
				glFogf(GL_FOG_START, iterFloat->second);
				break;
			case FOG_END: 
				glFogf(GL_FOG_END, iterFloat->second);
				break;
			case FOG_DENSITY: 
				glFogf(GL_FOG_DENSITY, iterFloat->second);
				break;

			case ALPHA_VALUE: 
				// it is the responsability of someone else to ensure that
				//both fields are defined
				glAlphaFunc(translate((Func)m_EnumProps[ALPHA_FUNC]),iterFloat->second);
				break;
		}
	}
#endif		

	std::map< int, vec4>::iterator iterVec4;
	iterVec4 = m_Float4Props.begin();
	for ( ; iterVec4 != m_Float4Props.end(); ++iterVec4) {
	
		switch(iterVec4->first) {
#if NAU_CORE_OPENGL  == 0
			case FOG_COLOR: 
				glFogfv(GL_FOG_COLOR, toFloatPtr(iterVec4->second));
				break;
#endif	
			case BLEND_COLOR: 
				glBlendColor(iterVec4->second.x,
					 iterVec4->second.y,
					  iterVec4->second.z,
					  iterVec4->second.w);;
				break;
		}
	}

	std::map< int, bvec4>::iterator iterBool4;
	iterBool4 = m_Bool4Props.begin();
	for ( ; iterBool4 != m_Bool4Props.end(); ++iterBool4) {
	
		switch(iterBool4->first) {
			case COLOR_MASK:
				glColorMask(iterBool4->second.x,
					iterBool4->second.y,
					iterBool4->second.z,
					iterBool4->second.w);
				break;
		}
	}

	iterInt = m_EnumProps.begin();
	for ( ; iterInt != m_EnumProps.end(); ++iterInt) {
	
		switch(iterInt->first) {
#if NAU_CORE_OPENGL  == 0
			case FOG_MODE: 
				glFogi(GL_FOG_MODE, iterInt->second));
				break;
			case FOG_COORD_SRC: 
				glFogi(GL_FOG_COORD_SRC,iterInt->second));
				break;
#endif
			case DEPTH_FUNC: 
				glDepthFunc(iterInt->second);
				break;
			case CULL_TYPE: 
				glCullFace(iterInt->second);
				break;
			case ORDER_TYPE: 
				break;
			case BLEND_SRC: 
			case BLEND_DST:
				int s,d;
				if (!m_EnumProps.count(BLEND_SRC))
					s = GL_ONE;
				else
					s =  m_EnumProps[BLEND_SRC];
				if (!m_EnumProps.count(BLEND_DST))
					d = GL_ZERO;
				else
					d =  m_EnumProps[BLEND_DST];
				
				glBlendFunc(s,d);
				break;
			//case BLEND_DST: 
			//	break;
			case BLEND_EQUATION: 
				glBlendEquation( iterInt->second);
				break;
			// done for the float props
			//case ALPHA_FUNC: 
			//	break;
		}
	}
}


void 
GlState::setDiff(IState *def, IState *aState) {

	GlState s;
	GlState *d = (GlState *)def;
	GlState *a = (GlState *)aState;

	std::map< int, int>::iterator iterInt;
	iterInt = m_IntProps.begin();
	// for all properties (with current values)
	for ( ; iterInt != m_IntProps.end(); ++iterInt) {
		// if a is setting a property	
		if (a->m_IntProps.count(iterInt->first)) {
			// if the value in a is different from the current value
			if (a->m_IntProps[iterInt->first] != iterInt->second) {
				// add property to new state
				s.m_IntProps[iterInt->first] = a->m_IntProps[iterInt->first];
				// change current state accordingly
				m_IntProps[iterInt->first] = a->m_IntProps[iterInt->first];
			}
		}

		// if a is not setting this property and the default value is different from the current value
		else if (d->m_IntProps.count(iterInt->first) && d->m_IntProps[iterInt->first] != iterInt->second) {
			// add the default value to the new state
			s.m_IntProps[iterInt->first] = d->m_IntProps[iterInt->first];
			// change the current state accordingly
			m_IntProps[iterInt->first] = d->m_IntProps[iterInt->first];
		}
	}

	std::map< int, bool>::iterator iterBool;
	iterBool = m_EnableProps.begin();
	for ( ; iterBool != m_EnableProps.end(); ++iterBool) {
	
		if (a->m_EnableProps.count(iterBool->first)) {
			if (a->m_EnableProps[iterBool->first] != iterBool->second) {
				// add property to new state
				s.m_EnableProps[iterBool->first] = a->m_EnableProps[iterBool->first];
				// change current state accordingly
				m_EnableProps[iterBool->first] = a->m_EnableProps[iterBool->first];
			}
		}

		else if (d->m_EnableProps.count(iterBool->first) && d->m_EnableProps[iterBool->first] != iterBool->second) {

			s.m_EnableProps[iterBool->first] = d->m_EnableProps[iterBool->first];
			m_EnableProps[iterBool->first] = d->m_EnableProps[iterBool->first];
		}
	}

	std::map< int, float>::iterator iterFloat;
	iterFloat = m_FloatProps.begin();
	for ( ; iterFloat != m_FloatProps.end(); ++iterFloat) {

		if (a->m_FloatProps.count(iterFloat->first)) {
			if (a->m_FloatProps[iterFloat->first] != iterFloat->second) {
				// add property to new state
				s.m_FloatProps[iterFloat->first] = a->m_FloatProps[iterFloat->first];
				// change current state accordingly
				m_FloatProps[iterFloat->first] = a->m_FloatProps[iterFloat->first];
			}
		}

		else if (d->m_FloatProps.count(iterFloat->first) && d->m_FloatProps[iterFloat->first] != iterFloat->second) {

			s.m_FloatProps[iterFloat->first] = d->m_FloatProps[iterFloat->first];
			m_FloatProps[iterFloat->first] = d->m_FloatProps[iterFloat->first];
		}
	}

	std::map< int, vec4>::iterator iterVec4;
	iterVec4 = m_Float4Props.begin();
	for ( ; iterVec4 != m_Float4Props.end(); ++iterVec4) {

		if (a->m_Float4Props.count(iterVec4->first)) {
			if (a->m_Float4Props[iterVec4->first] != iterVec4->second) {
				// add property to new state
				s.m_Float4Props[iterVec4->first] = a->m_Float4Props[iterVec4->first];
				// change current state accordingly
				m_Float4Props[iterVec4->first] = a->m_Float4Props[iterVec4->first];
			}
		}

		else if (d->m_Float4Props.count(iterVec4->first) && d->m_Float4Props[iterVec4->first] != iterVec4->second) {

			s.m_Float4Props[iterVec4->first] = d->m_Float4Props[iterVec4->first];
			m_Float4Props[iterVec4->first] = d->m_Float4Props[iterVec4->first];
		}
	}

	std::map< int, bvec4>::iterator iterBool4;
	iterBool4 = m_Bool4Props.begin();
	for ( ; iterBool4 != m_Bool4Props.end(); ++iterBool4) {

		if (a->m_Bool4Props.count(iterBool4->first)) {
			if (a->m_Bool4Props[iterBool4->first] != iterBool4->second) {
				// add property to new state
				s.m_Bool4Props[iterBool4->first] = a->m_Bool4Props[iterBool4->first];
				// change current state accordingly
				m_Bool4Props[iterBool4->first] = a->m_Bool4Props[iterBool4->first];
			}
		}

		else if (d->m_Bool4Props.count(iterBool4->first) && d->m_Bool4Props[iterBool4->first] != iterBool4->second) {

			s.m_Bool4Props[iterBool4->first] = d->m_Bool4Props[iterBool4->first];
			m_Bool4Props[iterBool4->first] = d->m_Bool4Props[iterBool4->first];
		}
	}

	iterInt = m_EnumProps.begin();
	for ( ; iterInt != m_EnumProps.end(); ++iterInt) {
	
		if (a->m_EnumProps.count(iterInt->first)) {
			if (a->m_EnumProps[iterInt->first] != iterInt->second) {
				// add property to new state
				s.m_EnumProps[iterInt->first] = a->m_EnumProps[iterInt->first];
				// change current state accordingly
				m_EnumProps[iterInt->first] = a->m_EnumProps[iterInt->first];
			}
		}

		else if (d->m_EnumProps.count(iterInt->first) && d->m_EnumProps[iterInt->first] != iterInt->second) {

			s.m_EnumProps[iterInt->first] = d->m_EnumProps[iterInt->first];
			m_EnumProps[iterInt->first] = d->m_EnumProps[iterInt->first];
		}
	}
	// set the difference
	s.set();
}


float *
GlState::toFloatPtr(vec4& v) {

	float *res = (float *)malloc(sizeof(float) * 4);

	res[0] = v.x;
	res[1] = v.y;
	res[2] = v.z;
	res[3] = v.w;

	return res;
}


bool 
GlState::difColor(vec4& a, vec4& b) {

	if (a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w) {
		a.x = b.x;
		a.y = b.y;
		a.z = b.z;
		a.w = b.w;
		return true;
	}
	else
		return false;
}


bool 
GlState::difBoolVector(bool* a, bool* b) {

	if (a[0] != b[0] || a[1] != b[1] || a[2] != b[2] || a[3] != b[3]) {
		a[0] = b[0];
		a[1] = b[1];
		a[2] = b[2];
		a[3] = b[3];
		return true;
	}
	else
		return false;
}


#include "nau/render/opengl/glState.h"

#include "nau/config.h"

#include <glbinding/gl/gl.h>
using namespace gl;
//#include <GL/glew.h>

using namespace nau::render;


bool
GLState::InitGL() {

	// ENUM
	NauInt ni6((int)GL_LESS);
	Attribs.setDefault("DEPTH_FUNC", ni6);
	Attribs.listAdd("DEPTH_FUNC", "LESS", (int)GL_LESS);
	Attribs.listAdd("DEPTH_FUNC", "NEVER", (int)GL_NEVER);
	Attribs.listAdd("DEPTH_FUNC", "ALWAYS", (int)GL_ALWAYS);
	Attribs.listAdd("DEPTH_FUNC", "LEQUAL", (int)GL_LEQUAL);
	Attribs.listAdd("DEPTH_FUNC", "EQUAL", (int)GL_EQUAL);
	Attribs.listAdd("DEPTH_FUNC", "GEQUAL", (int)GL_GEQUAL);
	Attribs.listAdd("DEPTH_FUNC", "GREATER", (int)GL_GREATER);
	Attribs.listAdd("DEPTH_FUNC", "NOT_EQUAL", (int)GL_NOTEQUAL);

	NauInt ni5((int)GL_BACK);
	Attribs.setDefault("CULL_TYPE", ni5);
	Attribs.listAdd("CULL_TYPE", "FRONT", (int)GL_FRONT);
	Attribs.listAdd("CULL_TYPE", "BACK", (int)GL_BACK);
	Attribs.listAdd("CULL_TYPE", "FRONT_AND_BACK", (int)GL_FRONT_AND_BACK);
	
	NauInt ni4((int)FRONT_TO_BACK);	
	Attribs.setDefault("ORDER_TYPE", ni4);
	Attribs.listAdd("ORDER_TYPE", "FRONT_TO_BACK", FRONT_TO_BACK);
	Attribs.listAdd("ORDER_TYPE", "BACK_TO_FRONT", BACK_TO_FRONT);

	NauInt ni3((int)GL_ONE);
	Attribs.setDefault("BLEND_SRC", ni3);
	Attribs.listAdd("BLEND_SRC", "ZERO", (int)GL_ZERO);
	Attribs.listAdd("BLEND_SRC", "ONE", (int)GL_ONE);
	Attribs.listAdd("BLEND_SRC", "SRC_COLOR", (int)GL_SRC_COLOR);
	Attribs.listAdd("BLEND_SRC", "ONE_MINUS_SRC_COLOR", (int)GL_ONE_MINUS_SRC_COLOR);
	Attribs.listAdd("BLEND_SRC", "DST_COLOR", (int)GL_DST_COLOR);
	Attribs.listAdd("BLEND_SRC", "ONE_MINUS_DST_COLOR", (int)GL_ONE_MINUS_DST_COLOR);
	Attribs.listAdd("BLEND_SRC", "SRC_ALPHA", (int)GL_SRC_ALPHA);
	Attribs.listAdd("BLEND_SRC", "ONE_MINUS_SRC_ALPHA", (int)GL_ONE_MINUS_SRC_ALPHA);
	Attribs.listAdd("BLEND_SRC", "DST_ALPHA", (int)GL_DST_ALPHA);
	Attribs.listAdd("BLEND_SRC", "ONE_MINUS_DST_ALPHA", (int)GL_ONE_MINUS_DST_ALPHA);
	Attribs.listAdd("BLEND_SRC", "SRC_ALPHA_SATURATE", (int)GL_SRC_ALPHA_SATURATE);
	Attribs.listAdd("BLEND_SRC", "CONSTANT_COLOR", (int)GL_CONSTANT_COLOR);
	Attribs.listAdd("BLEND_SRC", "ONE_MINUS_CONSTANT_COLOR", (int)GL_ONE_MINUS_CONSTANT_COLOR);
	Attribs.listAdd("BLEND_SRC", "CONSTANT_ALPHA", (int)GL_CONSTANT_ALPHA);
	Attribs.listAdd("BLEND_SRC", "ONE_MINUS_CONSTANT_ALPHA", (int)GL_ONE_MINUS_CONSTANT_ALPHA);

	NauInt ni2((int)GL_ZERO);
	Attribs.setDefault("BLEND_DST", ni2);
	Attribs.listAdd("BLEND_DST", "ZERO", (int)GL_ZERO);
	Attribs.listAdd("BLEND_DST", "ONE", (int)GL_ONE);
	Attribs.listAdd("BLEND_DST", "SRC_COLOR", (int)GL_SRC_COLOR);
	Attribs.listAdd("BLEND_DST", "ONE_MINUS_SRC_COLOR", (int)GL_ONE_MINUS_SRC_COLOR);
	Attribs.listAdd("BLEND_DST", "DST_COLOR", (int)GL_DST_COLOR);
	Attribs.listAdd("BLEND_DST", "ONE_MINUS_DST_COLOR", (int)GL_ONE_MINUS_DST_COLOR);
	Attribs.listAdd("BLEND_DST", "SRC_ALPHA", (int)GL_SRC_ALPHA);
	Attribs.listAdd("BLEND_DST", "ONE_MINUS_SRC_ALPHA", (int)GL_ONE_MINUS_SRC_ALPHA);
	Attribs.listAdd("BLEND_DST", "DST_ALPHA", (int)GL_DST_ALPHA);
	Attribs.listAdd("BLEND_DST", "ONE_MINUS_DST_ALPHA", (int)GL_ONE_MINUS_DST_ALPHA);
	Attribs.listAdd("BLEND_DST", "SRC_ALPHA_SATURATE", (int)GL_SRC_ALPHA_SATURATE);
	Attribs.listAdd("BLEND_DST", "CONSTANT_COLOR", (int)GL_CONSTANT_COLOR);
	Attribs.listAdd("BLEND_DST", "ONE_MINUS_CONSTANT_COLOR", (int)GL_ONE_MINUS_CONSTANT_COLOR);
	Attribs.listAdd("BLEND_DST", "CONSTANT_ALPHA", (int)GL_CONSTANT_ALPHA);
	Attribs.listAdd("BLEND_DST", "ONE_MINUS_CONSTANT_ALPHA", (int)GL_ONE_MINUS_CONSTANT_ALPHA);

	NauInt ni((int)GL_FUNC_ADD);
	Attribs.setDefault("BLEND_EQUATION", ni);
	Attribs.listAdd("BLEND_EQUATION", "ADD", (int)GL_FUNC_ADD);
	Attribs.listAdd("BLEND_EQUATION", "SUBTRACT", (int)GL_FUNC_SUBTRACT);
	Attribs.listAdd("BLEND_EQUATION", "REVERSE_SUBTRACT", (int)GL_FUNC_REVERSE_SUBTRACT);
	Attribs.listAdd("BLEND_EQUATION", "MIN", (int)GL_MIN);
	Attribs.listAdd("BLEND_EQUATION", "MAX", (int)GL_MAX);

	return true;
}

bool GLState::Inited = InitGL();







//
//Constructor & Destructor
//
GLState::GLState(): IState()   {

}


GLState::~GLState() {

}


 //Set Full States

void
GLState::set() {

	std::map< int, int>::iterator iterInt;
	iterInt = m_IntProps.begin();

	for ( ; iterInt != m_IntProps.end(); ++iterInt) {
	
		switch(iterInt->first) {
			case ORDER: break;
		}
	}


	std::map< int, bool>::iterator iterBool;
	iterBool = m_BoolProps.begin();
	for (; iterBool != m_BoolProps.end(); ++iterBool) {
	
		switch(iterBool->first) {
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
						glDepthMask((GLboolean)iterBool->second);
						break;
			case CULL_FACE: 
						if (iterBool->second)
							glEnable(GL_CULL_FACE);
						else
							glDisable(GL_CULL_FACE);
					break;
			case SAMPLE_SHADING:
				if (iterBool->second)
					glEnable(GL_SAMPLE_SHADING);
				else
					glDisable(GL_SAMPLE_SHADING);
				break;
		}
	}


	std::map< int, float>::iterator iterFloat;
	iterFloat = m_FloatProps.begin();
	for (; iterFloat != m_FloatProps.end(); ++iterFloat) {

		switch (iterFloat->first) {
		case MIN_SAMPLE_SHADING:
			glMinSampleShading(iterFloat->second);

			break;
		}
	}

	std::map< int, vec4>::iterator iterVec4;
	iterVec4 = m_Float4Props.begin();
	for ( ; iterVec4 != m_Float4Props.end(); ++iterVec4) {
	
		switch(iterVec4->first) {
			case BLEND_COLOR: 
				glBlendColor(iterVec4->second.x,
					 iterVec4->second.y,
					  iterVec4->second.z,
					  iterVec4->second.w);
				break;
		}
	}

	std::map< int, bvec4>::iterator iterBool4;
	iterBool4 = m_Bool4Props.begin();
	for ( ; iterBool4 != m_Bool4Props.end(); ++iterBool4) {
	
		switch(iterBool4->first) {
			case COLOR_MASK_B4:
				glColorMask((GLboolean)iterBool4->second.x,
					(GLboolean)iterBool4->second.y,
					(GLboolean)iterBool4->second.z,
					(GLboolean)iterBool4->second.w);
				break;
		}
	}

	iterInt = m_EnumProps.begin();
	for ( ; iterInt != m_EnumProps.end(); ++iterInt) {
	
		switch(iterInt->first) {
			case DEPTH_FUNC: 
				glDepthFunc((GLenum)iterInt->second);
				break;
			case CULL_TYPE: 
				glCullFace((GLenum)iterInt->second);
				break;
			case ORDER_TYPE: 
				break;
			case BLEND_SRC: 
			case BLEND_DST:
				GLenum s,d;
				if (!m_EnumProps.count(BLEND_SRC))
					s = GL_ONE;
				else
					s =  (GLenum)m_EnumProps[BLEND_SRC];
				if (!m_EnumProps.count(BLEND_DST))
					d = GL_ZERO;
				else
					d =  (GLenum)m_EnumProps[BLEND_DST];
				
				glBlendFunc(s,d);
				break;
			//case BLEND_DST: 
			//	break;
			case BLEND_EQUATION: 
				glBlendEquation( (GLenum)iterInt->second);
				break;
			// done for the float props
			//case ALPHA_FUNC: 
			//	break;
		}
	}
}


void 
GLState::setDiff(IState *def, IState *aState) {

	static GLState s;
	GLState *d = (GLState *)def;
	GLState *a = (GLState *)aState;

	s.clearArrays();
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

		// if a is not setting this property and 
		// the default value is different from the current value
		else if (d->m_IntProps.count(iterInt->first) 
				&& d->m_IntProps[iterInt->first] != iterInt->second) {
			// add the default value to the new state
			s.m_IntProps[iterInt->first] = d->m_IntProps[iterInt->first];
			// change the current state accordingly
			m_IntProps[iterInt->first] = d->m_IntProps[iterInt->first];
		}
	}

	std::map< int, bool>::iterator iterBool;
	iterBool = m_BoolProps.begin();
	for (; iterBool != m_BoolProps.end(); ++iterBool) {
	
		if (a->m_BoolProps.count(iterBool->first)) {
			if (a->m_BoolProps[iterBool->first] != iterBool->second) {
				// add property to new state
				s.m_BoolProps[iterBool->first] = a->m_BoolProps[iterBool->first];
				// change current state accordingly
				m_BoolProps[iterBool->first] = a->m_BoolProps[iterBool->first];
			}
		}

		else if (d->m_BoolProps.count(iterBool->first) && 
				d->m_BoolProps[iterBool->first] != iterBool->second) {

			s.m_BoolProps[iterBool->first] = d->m_BoolProps[iterBool->first];
			m_BoolProps[iterBool->first] = d->m_BoolProps[iterBool->first];
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

		else if (d->m_FloatProps.count(iterFloat->first) && 
				d->m_FloatProps[iterFloat->first] != iterFloat->second) {

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

		else if (d->m_Float4Props.count(iterVec4->first) && 
				d->m_Float4Props[iterVec4->first] != iterVec4->second) {

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

		else if (d->m_Bool4Props.count(iterBool4->first) && 
				d->m_Bool4Props[iterBool4->first] != iterBool4->second) {

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

		else if (d->m_EnumProps.count(iterInt->first) && 
				d->m_EnumProps[iterInt->first] != iterInt->second) {

			s.m_EnumProps[iterInt->first] = d->m_EnumProps[iterInt->first];
			m_EnumProps[iterInt->first] = d->m_EnumProps[iterInt->first];
		}
	}
	// set the difference
	s.set();
}


float *
GLState::toFloatPtr(vec4& v) {

	float *res = (float *)malloc(sizeof(float) * 4);

	res[0] = v.x;
	res[1] = v.y;
	res[2] = v.z;
	res[3] = v.w;

	return res;
}


bool 
GLState::difColor(vec4& a, vec4& b) {

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
GLState::difBoolVector(bool* a, bool* b) {

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


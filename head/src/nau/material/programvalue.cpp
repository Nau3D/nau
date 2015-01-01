#include <nau/material/programvalue.h>

#include <nau.h>

#include <nau/math/simpletransform.h>
#include <nau/system/textutil.h>

using namespace nau::material;
using namespace nau::math;
using namespace nau::render;
using namespace nau::scene;
using namespace nau::system;


const std::string ProgramValue::semanticTypeString[] = {"CAMERA", "LIGHT", "TEXTURE",
											"DATA", "PASS", "CURRENT", "IMAGE_TEXTURE"};

std::string 
ProgramValue::getSemanticTypeString(SEMANTIC_TYPE s) 
{
	return (semanticTypeString[s]);
}




bool
ProgramValue::Validate(std::string type,std::string context,std::string component)
{
		int id;
		nau::Enums::DataType dt;

	if (type == "CAMERA") {

		Camera::Attribs.getPropTypeAndId(component, &dt, &id);
		return (id != -1);
	}
	else if (type == "LIGHT") {

		Light::Attribs.getPropTypeAndId(component, &dt, &id);
		return (id != -1);
	}
	else if (type == "TEXTURE") {

		Texture::Attribs.getPropTypeAndId(component, &dt, &id);
		return (id != -1);
	}
	else if (type == "DATA") {
		return (Enums::isValidType(context));
	}
	else if (type == "PASS") {
		Pass::Attribs.getPropTypeAndId(component, &dt, &id);
		return (id != -1);
	}

	else if (type == "CURRENT" && context == "MATRIX") {

		IRenderer::getPropId(component, &id);
		return (id != -1);
	}
	else if (type == "CURRENT" && context == "COLOR") {

		ColorMaterial::Attribs.getPropTypeAndId(component, &dt, &id);
		return (id != -1);
	}
	else if (type == "CURRENT" && context == "TEXTURE") {

		if (component == "COUNT" || component == "UNIT")
			return true;

		Texture::Attribs.getPropTypeAndId(component, &dt, &id);
		return (id != -1);

	}
#if NAU_OPENGL_VERSION >=  420

	else if (type == "CURRENT" && context == "IMAGE_TEXTURE") {

		if (component == "UNIT")
			return true;

		ImageTexture::Attribs.getPropTypeAndId(component, &dt, &id);
		return (id != -1);

	}
#endif
	else if (type == "CURRENT" && context == "LIGHT") {

		if (component == "COUNT")
			return true;

		Light::Attribs.getPropTypeAndId(component, &dt, &id);
		return (id != -1);
	}
	else if (type == "CURRENT" && context == "CAMERA") {

		Camera::Attribs.getPropTypeAndId(component, &dt, &id);
		return (id != -1);
	}
	else if (type == "CURRENT" && context == "STATE") {

		IState::Attribs.getPropTypeAndId(component, &dt, &id);
		return (id != -1);
	}

	else if (type == "CURRENT" && context == "VIEWPORT") {

		Viewport::Attribs.getPropTypeAndId(component, &dt, &id);
		return (id != -1);
	}
	else if (type == "CURRENT" && context == "PASS") {
	
		Pass::Attribs.getPropTypeAndId(component, &dt, &id);
		return (id != -1);
	}
	return false;
}


ProgramValue::ProgramValue () {
}


ProgramValue::ProgramValue (std::string name, std::string type,std::string context,std::string valueof, int id, bool inSpecML) : m_Cardinality (0)
{
	int attr;
	nau::Enums::DataType dt;
	m_Values = NULL;
//	m_IntValue = NULL;
	m_InSpecML = inSpecML;

	m_Name = name;
	m_Context = context;
	m_Id = id;

	if (0 == type.compare ("CAMERA")) {

		m_Type = CAMERA;

		Camera::Attribs.getPropTypeAndId(valueof, &dt, &attr);
		m_ValueOf = attr;
		m_ValueType = dt;
		m_Cardinality = Enums::getCardinality(dt);
		m_Values = (void *)malloc(Enums::getSize(dt));
		return;
	} 
	else if (0 == type.compare ("LIGHT")) {

		m_Type = LIGHT;

		Light::Attribs.getPropTypeAndId(valueof, &dt, &attr);
		m_ValueOf = attr;
		m_ValueType = dt;
		m_Cardinality = Enums::getCardinality(dt);
		m_Values = (void *)malloc(Enums::getSize(dt));

		return;
	} 
	else if (0 == type.compare ("TEXTURE")) {

		m_Type = TEXTURE;

		Texture::Attribs.getPropTypeAndId(valueof, &dt, &attr);
		m_ValueOf = attr;
		m_ValueType = dt;
		m_Cardinality = Enums::getCardinality(dt);
		m_Values = (void *)malloc(Enums::getSize(dt));

		return;
	}
	else if (0 == type.compare ("DATA")) {
		m_Type = DATA;

		m_ValueOf = USERDATA;
		m_ValueType = nau::Enums::getType(context);
		m_Cardinality = nau::Enums::getCardinality(m_ValueType);

		m_Values = textutil::ParseFloats(valueof, m_Cardinality);
	}
	else if (0 == type.compare("PASS")) {
	
		m_Type = PASS;

		Pass::Attribs.getPropTypeAndId(valueof, &dt, &attr);
		m_ValueOf = attr;
		m_ValueType = dt;
		m_Cardinality = Enums::getCardinality(dt);
		m_Values = (void *)malloc(Enums::getSize(dt));
	}
	else if (0 == type.compare("CURRENT")) {

		m_Type = CURRENT;

		if (0 == context.compare("COLOR")) {

			ColorMaterial::Attribs.getPropTypeAndId(valueof, &dt, &attr);
			m_ValueOf = attr;
			m_ValueType = dt;
			m_Cardinality = Enums::getCardinality(dt);
			m_Values = (void *)malloc(Enums::getSize(dt));
		}

		else if (0 == context.compare("MATRIX")) {

			IRenderer::MatrixAttribs.getPropTypeAndId(valueof, &dt, &attr);
			m_ValueOf = attr;
			m_ValueType = dt;
			m_Cardinality = Enums::getCardinality(dt);
			m_Values = (void *)malloc(Enums::getSize(dt));
			return;
		}

		else if (0 == context.compare("LIGHT")) {
		
			if (0 == valueof.compare("COUNT")) {

				m_ValueOf = COUNT;
				m_ValueType = Enums::INT;
				m_Cardinality = 1;
				m_Values = (void *)malloc(Enums::getSize(m_ValueType));
				//m_IntValue = (int *)malloc(sizeof(int));
				return;
			}
			Light::Attribs.getPropTypeAndId(valueof, &dt, &attr);
			m_ValueOf = attr;
			m_ValueType = dt;
			m_Cardinality = Enums::getCardinality(dt);
			m_Values = (void *)malloc(Enums::getSize(dt));
			return;
		} 

		else if (0 == context.compare("CAMERA")) {

			Camera::Attribs.getPropTypeAndId(valueof, &dt, &attr);
			m_ValueOf = attr;
			m_ValueType = dt;
			m_Cardinality = Enums::getCardinality(dt);
			m_Values = (void *)malloc(Enums::getSize(dt));

			return;
		}

		else if (0 == context.compare("STATE")) {

			IState::Attribs.getPropTypeAndId(valueof, &dt, &attr);
			m_ValueOf = attr;
			m_ValueType = dt;
			m_Cardinality = Enums::getCardinality(dt);
			m_Values = (void *)malloc(Enums::getSize(dt));

			return;
		}

		else if (0 == context.compare("PASS")) {

			Pass::Attribs.getPropTypeAndId(valueof, &dt, &attr);
			m_ValueOf = attr;
			m_ValueType = dt;
			m_Cardinality = Enums::getCardinality(dt);
			m_Values = (void *)malloc(Enums::getSize(dt));

			return;
		}

		else if (0 == context.compare("VIEWPORT")) {

			Viewport::Attribs.getPropTypeAndId(valueof, &dt, &attr);
			m_ValueOf = attr;
			m_ValueType = dt;
			m_Cardinality = Enums::getCardinality(dt);
			m_Values = (void *)malloc(Enums::getSize(dt));

			return;
		}

		else if (0 == context.compare("TEXTURE")) {
		
			if (0 == valueof.compare("COUNT")) {

				m_ValueOf = COUNT;
				m_ValueType = Enums::INT;
				m_Cardinality = 1;
				m_Values = (void *)malloc(Enums::getSize(Enums::INT));
				return;
			}
			else if (0 == valueof.compare("UNIT")) {

				m_ValueOf = UNIT;
				m_ValueType = Enums::SAMPLER;
				m_Cardinality = 1;
				m_Values = (void *)malloc(Enums::getSize(Enums::SAMPLER));
				memcpy(m_Values, &m_Id, Enums::getSize(Enums::SAMPLER));
				return;
			}
			Texture::Attribs.getPropTypeAndId(valueof, &dt, &attr);
			m_ValueOf = attr;
			m_ValueType = dt;
			m_Cardinality = Enums::getCardinality(dt);
			m_Values = (void *)malloc(Enums::getSize(dt));

			return;
		} 
#if NAU_OPENGL_VERSION >=  420
		else if (0 == context.compare("IMAGE_TEXTURE")) {
		
			if (0 == valueof.compare("COUNT")) {

				m_ValueOf = COUNT;
				m_ValueType = Enums::INT;
				m_Cardinality = 1;
				m_Values = (void *)malloc(Enums::getSize(Enums::INT));
				return;
			}
			else if (0 == valueof.compare("UNIT")) {

				m_ValueOf = UNIT;
				m_ValueType = Enums::SAMPLER;
				m_Cardinality = 1;
				m_Values = (void *)malloc(Enums::getSize(Enums::SAMPLER));
				memcpy(m_Values, &m_Id, Enums::getSize(Enums::SAMPLER));
				return;
			}
			ImageTexture::Attribs.getPropTypeAndId(valueof, &dt, &attr);
			m_ValueOf = attr;
			m_ValueType = dt;
			m_Cardinality = Enums::getCardinality(dt);
			m_Values = (void *)malloc(Enums::getSize(dt));

			return;
		} 
#endif
		else if (0 == context.compare("PASS")) {
		
			m_Param = valueof;
			return;
		}

	}




}

ProgramValue::~ProgramValue () {

	//if (m_Value) {
	//	free (m_Value);
	//	m_Value = NULL;
	//}
	//if (m_IntValue) {
	//	free (m_IntValue);
	//	m_IntValue = NULL;
	//}
}


void 
ProgramValue::clone(ProgramValue &pv) 
{
	m_Type = pv.m_Type;
	m_Id = pv.m_Id;
	m_ValueOf = pv.m_ValueOf;
	m_ValueType = pv.m_ValueType;
	m_Context = pv.m_Context;
	m_Cardinality = pv.m_Cardinality;
	m_InSpecML = pv.m_InSpecML;
	m_Values = (void *)malloc(Enums::getSize(m_ValueType));
	memcpy(m_Values, pv.m_Values, Enums::getSize(m_ValueType));
}


int 
ProgramValue::getId() 
{
	return m_Id;
}

void
ProgramValue::setId(int id)
{
	m_Id = id;
}


std::string 
ProgramValue::getContext() {

	return(m_Context);
}

bool
ProgramValue::isInSpecML() 
{
	return m_InSpecML;
}
	 
void 
ProgramValue::setContext(std::string s) {

	m_Context = s;
}


std::string 
ProgramValue::getName() {

	return(m_Name);
}

int
ProgramValue::getCardinality (void)
{
   return m_Cardinality;
}



void*
ProgramValue::getValues (void)
{
   switch (m_Type) {

	   case CAMERA: {
		  Camera *cam = RENDERMANAGER->getCamera (m_Context);
		  m_Values = cam->getProp(m_ValueOf, m_ValueType);
		  return m_Values;
	   }
		break;
	   case LIGHT: {
			Light *light = RENDERMANAGER->getLight (m_Context);
			m_Values = light->getProp(m_ValueOf, m_ValueType);
			return m_Values;
	   }
		break;
	   case TEXTURE: {
			Texture *t = RESOURCEMANAGER->getTexture (m_Context);
			m_Values = t->getProp(m_ValueOf, m_ValueType);
			return m_Values;
		}
		break;
	   case DATA: {
		   return m_Values;
		}
	   case PASS: {		
		   Pass *p = RENDERMANAGER->getPass(m_Context);
		   m_Values = p->getProp(m_ValueOf, m_ValueType);
		   return m_Values;

		}
	   case CURRENT: {
		   if ("MATRIX" == m_Context) {
			   m_Values = (void *)RENDERER->getMatrix((IRenderer::MatrixType)m_ValueOf);
			   return m_Values;
		   }
		   else if ("COLOR" == m_Context) {	
			   m_Values = RENDERER->getColorProp(m_ValueOf, m_ValueType);
			   return m_Values;
		   }
		   else if ("TEXTURE" == m_Context) {

			   if (m_ValueOf == COUNT) {
				   int m = RENDERER->getTextureCount();
					memcpy(m_Values,&(m), sizeof(int));
				}
			   else if (m_ValueOf == UNIT) {

				   memcpy(m_Values, &m_Id, sizeof(int));
			   }
			   else if (m_Id < RENDERER->getTextureCount()) {
				   Texture *t = RENDERER->getTexture(m_Id);
				   m_Values = t->getProp(m_ValueOf, m_ValueType);
			   }
			   return m_Values;

		   }
#if NAU_OPENGL_VERSION >=  420
		   else if ("IMAGE_TEXTURE" == m_Context) {

			   if (m_ValueOf == COUNT) {
				   int m = RENDERER->getImageTextureCount();
				   memcpy(m_Values, &m, sizeof(int));
				}
			   else if (m_ValueOf == UNIT) {

				   memcpy(m_Values, &m_Id, sizeof(int));
			   }
			   else if (m_Id < RENDERER->getImageTextureCount()) {
				   ImageTexture *t = RENDERER->getImageTexture(m_Id);
				   m_Values = t->getProp(m_ValueOf, m_ValueType);
			   }
			   return m_Values;
		   }
#endif
		   else if ("LIGHT" == m_Context) {

			   if (m_ValueOf == COUNT) {
				   int m = RENDERER->getLightCount();
				   memcpy(m_Values, &m, sizeof(int));
				   return m_Values;
				}
			   else if (m_Id < RENDERER->getLightCount()) {
				   Light *l = RENDERER->getLight(m_Id);
				   m_Values = l->getProp(m_ValueOf, m_ValueType);
			   }
			   return m_Values;
		   }
		   else if ("CAMERA" == m_Context) {

			  Camera *cam = RENDERER->getCamera();				
			  return cam->getProp(m_ValueOf, m_ValueType);
		   }
		   else if ("STATE" == m_Context) {

			   IState *s = RENDERER->getState();
			   return s->getProp(m_ValueOf, m_ValueType);
		   }
		   else if ("VIEWPORT" == m_Context) {

			   Viewport *s = RENDERER->getViewport();
			   return s->getProp(m_ValueOf, m_ValueType);
		   }
		   else if ("PASS" == m_Context) {
		   
			   Pass *p = RENDERMANAGER->getCurrentPass();
			   return p->getProp(m_ValueOf, m_ValueType);
		   }
		   else return 0;
		}
	   default:
		   return 0;
		
   }
   assert("Getting the Value of an Invalid ProgramValue");
   return 0;
}


ProgramValue::SEMANTIC_TYPE 
ProgramValue::getSemanticType() {

	return m_Type;
}

	 
void 
ProgramValue::setSemanticType(SEMANTIC_TYPE s) {

	m_Type = s;
}

int 
ProgramValue::getSemanticValueOf() {

	return m_ValueOf;
}

void 
ProgramValue::setSemanticValueOf(int s) {

	m_ValueOf = s;
}


nau::Enums::DataType
ProgramValue::getValueType()
{
	//int pType;
	//if (m_Type == PASS) {
	//	pType = RENDERMANAGER->getPassParamType(m_Context, m_Param);
	//	switch (pType) {
	//		case Enums::FLOAT:
	//			m_ValueType = Enums::FLOAT;
	//			break;
	//	}
	//}
	//else if (m_Type == CURRENT && m_Context == "PASS") {
	//	pType = RENDERMANAGER->getCurrentPassParamType(m_Param);
	//	switch (pType) {
	//		case Enums::FLOAT:
	//			m_ValueType = Enums::FLOAT;
	//			break;
	//	}
	//}

	return m_ValueType;
}


void 
ProgramValue::setValueType(nau::Enums::DataType s) {

	m_ValueType = s;
}


void
ProgramValue::setValueOfUniform(void *values) {

	memcpy(m_Values, values, Enums::getSize(m_ValueType));
	//for (int i = 0; i < m_Cardinality; i++) {
	//	m_IntValue[i] = (int)values[i];
	//}
}


void
ProgramValue::setLoc(int l) {

	m_Loc = l;
}


int
ProgramValue::getLoc() {

	return m_Loc;
}


//void 
//ProgramValue::setValueOfUniform (int *values) { 
//
//	for (int i = 0 ; i < m_Cardinality; i++) {
//		m_IntValue[i] = (int)values[i];	
//	}
//}
//
//// this version is more generic than the previous one
//// this is on purpose!
//void 
//ProgramValue::setValueOfUniform (float *values) { 
//
//	switch (m_ValueType) {
//		case Enums::INT:
//		case Enums::IVEC2:
//		case Enums::IVEC3:
//		case Enums::IVEC4:
//		case Enums::BOOL:
//		case Enums::BVEC2:
//		case Enums::BVEC3:
//		case Enums::BVEC4:
//		case Enums::SAMPLER:
//		case Enums::ENUM:
//			for (int i = 0 ; i < m_Cardinality; i++) {
//				m_IntValue[i] = (int)values[i];	
//			}
//			break;
//		default:
//			for (int i = 0 ; i < m_Cardinality; i++) {
//				m_Value[i] = values[i];	
//			}
//	}
//
//}



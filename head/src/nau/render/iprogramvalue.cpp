#include <nau/render/iprogramvalue.h>

#ifdef NAU_OPENGL
#include <nau/render/opengl/glProgramValue.h>
#endif

#include <nau.h>
#include <nau/material/colormaterial.h>

#include <nau/math/simpletransform.h>
#include <nau/system/textutil.h>

using namespace nau::material;
using namespace nau::math;
using namespace nau::render;
using namespace nau::scene;
using namespace nau::system;


bool
IProgramValue::Init() {

	// INT
	Attribs.add(Attribute(PROPERTY, "PROPERTY", Enums::DataType::INT, true, new int (-1)));
	Attribs.add(Attribute(SIZE, "SIZE", Enums::DataType::INT, true, new int (0)));
	Attribs.add(Attribute(CARDINALITY, "CARDINALITY", Enums::DataType::INT, true, new int(1)));
	Attribs.add(Attribute(ID, "ID", Enums::DataType::INT, true, new int(-1)));
	// BOOL
	Attribs.add(Attribute(CURRENT, "CURRENT", Enums::DataType::BOOL, true, new bool(false)));
	// ENUM
	Attribs.add(Attribute(SEMANTICS, "SEMANTICS", Enums::DataType::ENUM, true));
	Attribs.listAdd("SEMANTICS", "CAMERA", CAMERA);
	Attribs.listAdd("SEMANTICS", "LIGHT", LIGHT);
	Attribs.listAdd("SEMANTICS", "TEXTURE", TEXTURE);
	Attribs.listAdd("SEMANTICS", "IMAGE_TEXTURE", IMAGE_TEXTURE);
	Attribs.listAdd("SEMANTICS", "PASS", PASS);
	Attribs.listAdd("SEMANTICS", "COLOR", COLOR);
	return true;
}


AttribSet IProgramValue::Attribs;
bool IProgramValue::Inited = Init();


void
IProgramValue::initArrays() {

	Attribs.initAttribInstanceEnumArray(m_EnumProps);
	Attribs.initAttribInstanceIntArray(m_IntProps);
	Attribs.initAttribInstanceBoolArray(m_BoolProps);
}


bool
IProgramValue::Validate(std::string type,std::string context,std::string component)
{
	int id;
	nau::Enums::DataType dt;

	if (type == "COLOR") {

		ColorMaterial::Attribs.getPropTypeAndId(component, &dt, &id);
		return (id != -1);
	}
	else if (type == "CAMERA") {

		Camera::Attribs.getPropTypeAndId(component, &dt, &id);
		return (id != -1);
	}
	else if (type == "LIGHT") {

		if (context == "CURRENT" && component == "COUNT" )
			return true;
		else {
			Light::Attribs.getPropTypeAndId(component, &dt, &id);
			return (id != -1);
		}
	}
	else if (type == "TEXTURE") {

		if (context == "CURRENT" && (component == "COUNT" || component == "UNIT"))
			return true;
		else {
			Texture::Attribs.getPropTypeAndId(component, &dt, &id);
			return (id != -1);
		}
	}
	else if (type == "PASS") {
		// there is no way to validate this before loading the passes.
		// if a variable is not defined it will later assume a zero value
		return true;
	}

	else if (type == "RENDERER" && context == "MATRIX") {

		int id;
		IRenderer::getPropId(component, &id);
		return (id != -1);
	}
#if NAU_OPENGL_VERSION >=  420

	else if (type == "IMAGE_TEXTURE") {

		if (context == "CURRENT" && (component == "COUNT" || component == "UNIT"))
			return true;
		else {
			ImageTexture::Attribs.getPropTypeAndId(component, &dt, &id);
			return (id != -1);
		}
	}
#endif
	else
		return false;
}


IProgramValue::IProgramValue (): m_Value(0) {

}



IProgramValue *
IProgramValue::Create(unsigned int programID, std::string name, std::string type, std::string context, std::string component, int id) {

#ifdef NAU_OPENGL
	if (IProgramValue::Validate(type, context, component))
		return new GLProgramValue(unsigned int programID, std::string name, std::string type, std::string context, std::string valueof, int id);
#endif

	return NULL;
}


void
IProgramValue::set(std::string name, std::string type,std::string context,std::string component, int id) {

	int attr;
	nau::Enums::DataType dt;

	m_Value = NULL;
	m_Name = name;
	m_Context = context;
	m_Id = id;

	if (type == "CAMERA") {

		m_Type = CAMERA;
		Camera::Attribs.getPropTypeAndId(component, &dt, &attr);
		m_Component = attr;
		m_ValueType = dt;
		m_Cardinality = Enums::getCardinality(dt);
		m_Value = allocate(dt, m_Cardinality);
		return;
	} 
	else if (type == "LIGHT") {

		m_Type = LIGHT;
		if (context == "CURRENT" && component == "COUNT") {
			m_Component = COUNT;
			m_ValueType = Enums::INT;
		}
		else {
			Light::Attribs.getPropTypeAndId(component, &dt, &attr);
			m_Component = attr;
			m_ValueType = dt;
		}
		m_Cardinality = Enums::getCardinality(dt);
		m_Value = allocate(m_ValueType, m_Cardinality);
		return;
	} 
	else if (type == "TEXTURE") {

		m_Type = TEXTURE;
		if (context == "CURRENT") {
			if (component == "COUNT") {
				m_Component = COUNT;
				m_ValueType = Enums::INT;
			}
			else if (component == "UNIT") {
				m_Component = UNIT;
				m_ValueType = Enums::SAMPLER;
			}
		}
		else {
			Texture::Attribs.getPropTypeAndId(valueof, &dt, &attr);
			m_Component = attr;
			m_ValueType = dt;
		}
		m_Cardinality = Enums::getCardinality(dt);
		m_Value = allocate(m_ValueType, m_Cardinality);
		return;
	}
	else if (type =="DATA") {

		m_Type = DATA;
		m_Component = USERDATA;
		m_ValueType = nau::Enums::getType(context);
		m_Cardinality = nau::Enums::getCardinality(m_ValueType);

		m_Value = (float *)malloc(sizeof(float) * m_Cardinality);
		switch(m_ValueType) {
			case Enums::INT:
			case Enums::SAMPLER:
			case Enums::ENUM:
			case Enums::BOOL:
			case Enums::IVEC2:
			case Enums::IVEC3:
			case Enums::IVEC4:
			case Enums::BVEC2:
			case Enums::BVEC3:
			case Enums::BVEC4:
				m_IntValue = textutil::ParseInts(valueof, m_Cardinality);
				break;
			default:
				m_Value = textutil::ParseFloats(valueof, m_Cardinality);
		}
	}
	else if (0 == type.compare("PASS")) {
	
		m_Type = PASS;
		m_Param = valueof;
		m_Context = context;
	}
	else if (0 == type.compare("CURRENT")) {

		m_Type = CURRENT;

		if (0 == context.compare("COLOR")) {

			ColorMaterial::ColorComponent attr;
			Enums::DataType dt;

			ColorMaterial::Attribs::getComponentTypeAndId(valueof, &dt, &attr);
			m_ValueType = dt;
			m_ValueOf = attr;
			m_Cardinality = Enums::getCardinality(dt);
			m_Value = (float *)malloc(sizeof(float) * m_Cardinality);
			return;
		}

		else if (0 == context.compare("MATRIX")) {

			int attr;
			IRenderer::getPropId(valueof, &attr);
			m_ValueOf = attr;

			if (m_ValueOf == IRenderer::NORMAL ) {
				m_ValueType = nau::Enums::MAT3;
				m_Cardinality = Enums::getCardinality(Enums::MAT3);
			}
			else {
				m_ValueType = nau::Enums::MAT4;
				m_Cardinality = Enums::getCardinality(Enums::MAT4);
			}

			m_Value = (float *)malloc(sizeof(float) * m_Cardinality); 
			return;
		}

		else if (0 == context.compare("LIGHT")) {
		
			if (0 == valueof.compare("COUNT")) {

				m_ValueOf = COUNT;
				m_ValueType = Enums::INT;
				m_Cardinality = 1;
				m_IntValue = (int *)malloc(sizeof(int));
				return;
			}
			int attr;
			nau::Enums::DataType dt;
			Light::Attribs.getPropTypeAndId(valueof, &dt, &attr);
			m_ValueOf = attr;
			m_ValueType = dt;
			m_Cardinality = Enums::getCardinality(dt);

			switch(dt) {
				case Enums::INT:
				case Enums::SAMPLER:
				case Enums::BOOL:
				case Enums::ENUM:
				case Enums::IVEC2:
				case Enums::IVEC3:
				case Enums::IVEC4:
				case Enums::BVEC2:
				case Enums::BVEC3:
				case Enums::BVEC4:
					m_IntValue = (int *)malloc(sizeof(int) * m_Cardinality);
				default:
					m_Value = (float *)malloc(sizeof(float) * m_Cardinality); 
			}
			return;
		} 


		else if (0 == context.compare("CAMERA")) {


			int attr;
			nau::Enums::DataType dt;
			Camera::Attribs.getPropTypeAndId(valueof, &dt, &attr);
			m_ValueOf = attr;
			m_ValueType = dt;
			m_Cardinality = Enums::getCardinality(dt);

			switch(dt) {
				case Enums::INT:
				case Enums::SAMPLER:
				case Enums::ENUM:
				case Enums::BOOL:
				case Enums::IVEC2:
				case Enums::IVEC3:
				case Enums::IVEC4:
				case Enums::BVEC2:
				case Enums::BVEC3:
				case Enums::BVEC4:
					m_IntValue = (int *)malloc(sizeof(int) * m_Cardinality);
				default:
					m_Value = (float *)malloc(sizeof(float) * m_Cardinality); 
			}
			return;
		}

		else if (0 == context.compare("TEXTURE")) {
		
			if (0 == valueof.compare("COUNT")) {

				m_ValueOf = COUNT;
				m_ValueType = Enums::INT;
				m_Cardinality = 1;
				m_IntValue = (int *)malloc(sizeof(int));
				return;
			}
			else if (0 == valueof.compare("UNIT")) {

				m_ValueOf = UNIT;
				m_ValueType = Enums::DataType::SAMPLER;
				m_Cardinality = 1;
				m_IntValue = (int *)malloc(sizeof(int));
				m_IntValue[0] = m_Id;
				return;
			}
			int attr;
			nau::Enums::DataType dt;
			Texture::Attribs.getPropTypeAndId(valueof, &dt, &attr);
			m_ValueOf = attr;
			m_ValueType = dt;
			m_Cardinality = Enums::getCardinality(dt);

			switch(dt) {
				case Enums::INT:
				case Enums::SAMPLER:
				case Enums::BOOL:
				case Enums::ENUM:
				case Enums::IVEC2:
				case Enums::IVEC3:
				case Enums::IVEC4:
				case Enums::BVEC2:
				case Enums::BVEC3:
				case Enums::BVEC4:
					m_IntValue = (int *)malloc(sizeof(int) * m_Cardinality);
				default:
					m_Value = (float *)malloc(sizeof(float) * m_Cardinality); 
			}
			return;
		} 
#if NAU_OPENGL_VERSION >=  420
		else if (0 == context.compare("IMAGE_TEXTURE")) {
		
			if (0 == valueof.compare("COUNT")) {

				m_ValueOf = COUNT;
				m_ValueType = Enums::INT;
				m_Cardinality = 1;
				m_IntValue = (int *)malloc(sizeof(int));
				return;
			}
			else if (0 == valueof.compare("UNIT")) {

				m_ValueOf = UNIT;
				m_ValueType = Enums::DataType::SAMPLER;
				m_Cardinality = 1;
				m_IntValue = (int *)malloc(sizeof(int));
				m_IntValue[0] = m_Id;
				return;
			}
			int attr;
			nau::Enums::DataType dt;
			ImageTexture::Attribs.getPropTypeAndId(valueof, &dt, &attr);
			m_ValueOf = attr;
			m_ValueType = dt;
			m_Cardinality = Enums::getCardinality(dt);

			switch(dt) {
				case Enums::INT:
				case Enums::SAMPLER:
				case Enums::BOOL:
				case Enums::ENUM:
				case Enums::IVEC2:
				case Enums::IVEC3:
				case Enums::IVEC4:
				case Enums::BVEC2:
				case Enums::BVEC3:
				case Enums::BVEC4:
					m_IntValue = (int *)malloc(sizeof(int) * m_Cardinality);
				default:
					m_Value = (float *)malloc(sizeof(float) * m_Cardinality); 
			}
			return;
		} 
#endif
		else if (0 == context.compare("PASS")) {
		
			//get type
			//int dType = RENDERMANAGER->getCurrentPassParamType(valueof);
			//switch (dType) {
			//case Pass::FLOAT: 
			//	m_ValueType = Enums::FLOAT;
			//	m_Cardinality = 1;
			//	m_Value = (float *)malloc(sizeof(float) * m_Cardinality);
				m_Param = valueof;
			//	break;

			//}
			return;
		}

	}



}

IProgramValue::~IProgramValue () {

	if (m_Value)
		free (m_Value);
}


void *
IProgramValue::allocate(nau::Enums::DataType dt, int cardinality) {

	void *v;

	switch (dt) {
		case Enums::INT:
		case Enums::SAMPLER:
		case Enums::BOOL:
		case Enums::ENUM:
		case Enums::IVEC2:
		case Enums::IVEC3:
		case Enums::IVEC4:
		case Enums::BVEC2:
		case Enums::BVEC3:
		case Enums::BVEC4:
			v = (void *)malloc(sizeof(int) * m_Cardinality);
		default:
			v = (void *)malloc(sizeof(float) * m_Cardinality);
	}
	return v;
}


void 
IProgramValue::clone(IProgramValue *pv) 
{
	m_Type = pv->m_Type;
	m_Id = pv->m_Id;
	m_ValueOf = pv->m_ValueOf;
	m_ValueType = pv->m_ValueType;
	m_Context = pv->m_Context;
	m_Cardinality = pv->m_Cardinality;
	m_InSpecML = pv->m_InSpecML;

	if (pv->m_Value) {
		for (int i = 0 ; i < m_Cardinality; i++)
			m_Value[i] = pv->m_Value[i];
	}
	if (pv->m_IntValue) {
		for (int i = 0 ; i < m_Cardinality; i++)
			m_IntValue[i] = pv->m_IntValue[i];
	}
}


int 
IProgramValue::getId() 
{
	return m_Id;
}

void
IProgramValue::setId(int id)
{
	m_Id = id;
}


std::string 
IProgramValue::getContext() {

	return(m_Context);
}

bool
IProgramValue::isInSpecML() 
{
	return m_InSpecML;
}
	 
void 
IProgramValue::setContext(std::string s) {

	m_Context = s;
}


std::string 
IProgramValue::getName() {

	return(m_Name);
}

int
ProgramValue::getCardinality (void)
{
   return m_Cardinality;
}



void*
IProgramValue::getValues (void)
{
   static SimpleTransform returnTransform; //What an ugly, ugly thing to do...

   switch (m_Type) {

	   case CAMERA: {
		  Camera *cam = RENDERMANAGER->getCamera (m_Context);
				
		  switch(m_ValueType) {
				case Enums::VEC4:
					return(float *)(&(cam->getPropf4((Camera::Float4Property)m_ValueOf))); 

				case Enums::FLOAT:
					m_Value[0] = cam->getPropf((Camera::FloatProperty)m_ValueOf);
					return m_Value;

				case Enums::MAT4:
					return (float *)(&(cam->getPropm4((Camera::Mat4Property)m_ValueOf)));
					 
				default: return 0; 
		   
		   }

	   }
		break;
	   case LIGHT: {
			Light *light = RENDERMANAGER->getLight (m_Context);
			   switch(m_ValueType) {
					case Enums::VEC4:
						return(float *)(&(light->getPropf4((Light::Float4Property)m_ValueOf))); 
	
					case Enums::FLOAT:
						m_Value[0] = light->getPropf((Light::FloatProperty)m_ValueOf);
						return m_Value;

					case Enums::ENUM:
						m_IntValue[0] = light->getPrope((Light::EnumProperty)m_ValueOf);
						return m_IntValue;

					case Enums::BOOL:
						m_IntValue[0] = light->getPropb((Light::BoolProperty)m_ValueOf);
						return m_IntValue;

					default: return 0; 
			   
			   }
	   }
		break;
	   case TEXTURE: {
			Texture *t = RESOURCEMANAGER->getTexture (m_Context);

			switch(m_ValueType) {
					case Enums::INT:
						m_IntValue[0] = t->getPropi((Texture::IntProperty)m_ValueOf);
						return m_IntValue;

					case Enums::SAMPLER:
//						m_IntValue[0] = t->getProps((Texture::SamplerProperty)m_ValueOf);
						return 0;//m_IntValue;
			}
		}
		break;
	   case DATA: {
		   switch (m_ValueType) {
				case Enums::FLOAT:
				case Enums::VEC2:
				case Enums::VEC3:
				case Enums::VEC4:
				case Enums::MAT2:
				case Enums::MAT3:
				case Enums::MAT4:
					return m_Value;
				case Enums::SAMPLER:
				case Enums::INT:
				case Enums::ENUM:
				case Enums::BOOL:
				case Enums::IVEC2:
				case Enums::BVEC2:
				case Enums::IVEC3:
				case Enums::BVEC3:
				case Enums::IVEC4:
				case Enums::BVEC4:
					return m_IntValue;
				default:
					return 0;
		   }
		}
	   case PASS: {
				  
		
		   int pType = RENDERMANAGER->getPassParamType(m_Context, m_Param);
		   switch (pType) {
		   
				case Enums::FLOAT:
					return RENDERMANAGER->getPassParamf(m_Context, m_Param);
		   }

		}
	   case CURRENT: {
		   if ("MATRIX" == m_Context) {

			   return const_cast<float*>(RENDERER->getMatrix((IRenderer::MatrixType)m_ValueOf));

					//case NORMALMATRIX: return const_cast<float*>(RENDERER->getNormalMatrix());				
					//case VIEWMATRIX: return const_cast<float*>(RENDERER->getMatrix(IRenderer::VIEW));
					//case MODELMATRIX: return const_cast<float*>(RENDERER->getMatrix(IRenderer::MODEL));
					//case PROJECTIONMATRIX: return const_cast<float*>(RENDERER->getMatrix(IRenderer::PROJECTION));
					//case TEXTUREMATRIX: return const_cast<float*>(RENDERER->getMatrix(IRenderer::TEXTURE));
					//case VIEWMODELMATRIX: return const_cast<float*>(RENDERER->getMatrix(IRenderer::VIEWMODEL));
					//case PROJECTIONVIEWMODELMATRIX: return const_cast<float*>(RENDERER->getMatrix(IRenderer::PROJECTIONVIEWMODEL));
					//case Camera::TS05_PVM_MATRIX: return const_cast<float*>(RENDERER->getMatrix(IRenderer::TS05_PVM));
				//	default: return 0;
				//}
		   }
		   else if ("COLOR" == m_Context) {
					
			   return const_cast<float*>(RENDERER->getColor((ColorMaterial::ColorComponent)m_ValueOf));

			   //switch (m_ValueOf) {
				//	case DIFFUSE: (RENDERER->getColor(IRenderer::DIFFUSE));
				//	case AMBIENT: return const_cast<float*>(RENDERER->getColor(IRenderer::AMBIENT));
				//	case SPECULAR: return const_cast<float*>(RENDERER->getColor(IRenderer::SPECULAR));
				//	case EMISSION: return const_cast<float*>(RENDERER->getColor(IRenderer::EMISSION));
				//	case SHININESS: return const_cast<float*>( RENDERER->getColor(IRenderer::SHININESS));
				//	default: return 0;
			 //  } 
		   }
		   else if ("TEXTURE" == m_Context) {

			   if (m_ValueOf == COUNT) {
					m_IntValue[0] = RENDERER->getTextureCount();
					return m_IntValue;
				}

				switch(m_ValueType) {
					case Enums::INT:
						m_IntValue[0] = RENDERER->getPropi((IRenderer::TextureUnit)m_Id,(Texture::IntProperty)m_ValueOf);
						return m_IntValue;

					case Enums::SAMPLER:
						return m_IntValue;//&m_Id;

						//m_IntValue[0] = RENDERER->getProps((IRenderer::TextureUnit)m_Id, (Texture::SamplerProperty)m_ValueOf);
						//return m_IntValue;
				}
		   }
#if NAU_OPENGL_VERSION >=  420
		   else if ("IMAGE_TEXTURE" == m_Context) {

			   if (m_ValueOf == COUNT) {
					m_IntValue[0] = RENDERER->getImageTextureCount();
					return m_IntValue;
				}
				switch(m_ValueType) {

					case Enums::SAMPLER:
						return m_IntValue;//&m_Id;

						//m_IntValue[0] = RENDERER->getProps((IRenderer::TextureUnit)m_Id, (Texture::SamplerProperty)m_ValueOf);
						//return m_IntValue;
				}
		   }
#endif
		   else if ("LIGHT" == m_Context) {

			   if (m_ValueOf == COUNT) {
					m_IntValue[0] = RENDERER->getLightCount();
					return m_IntValue;
				}
			   if (RENDERER->getLightCount() == 0)
				   return 0;

			   switch(m_ValueType) {
					case Enums::VEC4:
						return((float *)&(RENDERER->getLight(m_Id)->getPropf4((Light::Float4Property)m_ValueOf)));//	getLightfvComponent(m_Id, (Light::Float4Property)m_ValueOf)); 
	
					case Enums::FLOAT:
						m_Value[0] = RENDERER->getLight(m_Id)->getPropf((Light::FloatProperty)m_ValueOf);
						return m_Value;

					case Enums::ENUM:
						m_IntValue[0] = RENDERER->getLight(m_Id)->getPrope((Light::EnumProperty)m_ValueOf);
						return m_IntValue;

					case Enums::BOOL:
						m_IntValue[0] = RENDERER->getLight(m_Id)->getPropb((Light::BoolProperty)m_ValueOf);
						return m_IntValue;

					default: return 0; 
			   
			   }
		   }
		   else if ("CAMERA" == m_Context) {

			  Camera *cam = RENDERER->getCamera ();
				
			  switch(m_ValueType) {
					case Enums::VEC4:
						return(float *)(&(cam->getPropf4((Camera::Float4Property)m_ValueOf))); 

					case Enums::FLOAT:
						m_Value[0] = cam->getPropf((Camera::FloatProperty)m_ValueOf);
						return m_Value;

					case Enums::MAT4:
						return (float *)(&(cam->getPropm4((Camera::Mat4Property)m_ValueOf)));
					 
					default: return 0; 
		   
			   }
		   }
		   else if ("PASS" == m_Context) {
		   
			   int type = RENDERMANAGER->getCurrentPassParamType(m_Param);
			   switch(type) {
				case Enums::FLOAT:
					m_ValueType = Enums::FLOAT;
					m_Value = RENDERMANAGER->getCurrentPassParamf(m_Param);
					return m_Value;
			   }	
		   }
		   else return 0;
		}
	   default:
		   return 0;
		
   }
   assert("Getting the Value of an Invalid ProgramValue");
   return 0;
}


IProgramValue::SEMANTIC_TYPE 
IProgramValue::getSemanticType() {

	return m_Type;
}

	 
void 
IProgramValue::setSemanticType(SEMANTIC_TYPE s) {

	m_Type = s;
}

int 
IProgramValue::getSemanticValueOf() {

	return m_ValueOf;
}

void 
IProgramValue::setSemanticValueOf(int s) {

	m_ValueOf = s;
}


nau::Enums::DataType
IProgramValue::getValueType()
{
	int pType;
	if (m_Type == PASS) {
		pType = RENDERMANAGER->getPassParamType(m_Context, m_Param);
		switch (pType) {
			case Enums::FLOAT:
				m_ValueType = Enums::FLOAT;
				break;
		}
	}
	else if (m_Type == CURRENT && m_Context == "PASS") {
		pType = RENDERMANAGER->getCurrentPassParamType(m_Param);
		switch (pType) {
			case Enums::FLOAT:
				m_ValueType = Enums::FLOAT;
				break;
		}
	}

	return m_ValueType;
}


void 
IProgramValue::setValueType(nau::Enums::DataType s) {

	m_ValueType = s;
}

void 
IProgramValue::setValueOfUniform (int *values) { 

	for (int i = 0 ; i < m_Cardinality; i++) {
		m_IntValue[i] = (int)values[i];	
	}
}

// this version is more generic than the previous one
// this is on purpose!
void 
IProgramValue::setValueOfUniform (float *values) { 

	switch (m_ValueType) {
		case Enums::INT:
		case Enums::IVEC2:
		case Enums::IVEC3:
		case Enums::IVEC4:
		case Enums::BOOL:
		case Enums::BVEC2:
		case Enums::BVEC3:
		case Enums::BVEC4:
		case Enums::SAMPLER:
		case Enums::ENUM:
			for (int i = 0 ; i < m_Cardinality; i++) {
				m_IntValue[i] = (int)values[i];	
			}
			break;
		default:
			for (int i = 0 ; i < m_Cardinality; i++) {
				m_Value[i] = values[i];	
			}
	}

}



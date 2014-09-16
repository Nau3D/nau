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
	if (type == "CAMERA") {

		int id;
		nau::Enums::DataType dt;
		Camera::Attribs.getPropTypeAndId(component, &dt, &id);
		return (id != -1);
	}
	else if (type == "LIGHT") {

		int id;
		nau::Enums::DataType dt;
		Light::Attribs.getPropTypeAndId(component, &dt, &id);
		return (id != -1);
	}
	else if (type == "TEXTURE") {

		//if (component == "UNIT")
		//	return 0;

		int id;
		nau::Enums::DataType dt;
		Texture::Attribs.getPropTypeAndId(component, &dt, &id);
		return (id != -1);
	}
	else if (type == "DATA") {
		return (Enums::isValidType(context));
	}
	else if (type == "PASS") {
		// not much we can do here
		return true;
	}

	else if (type == "CURRENT" && context == "MATRIX") {

		int id;
		IRenderer::getPropId(component, &id);
		return (id != -1);
	}
	else if (type == "CURRENT" && context == "COLOR") {

		return ColorMaterial::validateComponent(component);
	}
	else if (type == "CURRENT" && context == "TEXTURE") {

		if (component == "COUNT" || component == "UNIT")
			return true;

		int id;
		nau::Enums::DataType dt;
		Texture::Attribs.getPropTypeAndId(component, &dt, &id);
		return (id != -1);

	}
#if NAU_OPENGL_VERSION >=  420

	else if (type == "CURRENT" && context == "IMAGE_TEXTURE") {

		if (component == "UNIT")
			return true;

		int id;
		nau::Enums::DataType dt;
		ImageTexture::Attribs.getPropTypeAndId(component, &dt, &id);
		return (id != -1);

	}
#endif
	else if (type == "CURRENT" && context == "LIGHT") {

		if (component == "COUNT")
			return true;
		int id;
		nau::Enums::DataType dt;
		Light::Attribs.getPropTypeAndId(component, &dt, &id);
		return (id != -1);
	}
	else if (type == "CURRENT" && context == "CAMERA") {

		int id;
		nau::Enums::DataType dt;
		Camera::Attribs.getPropTypeAndId(component, &dt, &id);
		return (id != -1);
	}
	else if (type == "CURRENT" && context == "PASS") {
	
		// there is no way to validate this before loading the passes.
		// if a variable is not defined it will later assume a zero value
		return true;
	}
	return false;
}


ProgramValue::ProgramValue () {m_Value = 0; m_IntValue = 0; m_InSpecML = false;};


ProgramValue::ProgramValue (std::string name, std::string type,std::string context,std::string valueof, int id, bool inSpecML) : m_Cardinality (0)
{
	m_Value = NULL;
	m_IntValue = NULL;
	m_InSpecML = inSpecML;

	m_Name = name;
	m_Context = context;
	m_Id = id;

	if (0 == type.compare ("CAMERA")) {

		m_Type = CAMERA;

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
	else if (0 == type.compare ("LIGHT")) {

		m_Type = LIGHT;

		int attr;
		nau::Enums::DataType dt;
		Light::Attribs.getPropTypeAndId(valueof, &dt, &attr);
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
	else if (0 == type.compare ("TEXTURE")) {

		m_Type = TEXTURE;

		int attr;
		nau::Enums::DataType dt;
		Texture::Attribs.getPropTypeAndId(valueof, &dt, &attr);
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
	else if (0 == type.compare ("DATA")) {
		m_Type = DATA;

		m_ValueOf = USERDATA;
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

			ColorMaterial::getComponentTypeAndId(valueof, &dt, &attr);
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


	//if (0 == valueof.compare ("ID")) {
	//	m_ValueOf = ID;
	//	m_ValueType = Enums::INT;
	//	m_Cardinality = 1;
	//	m_IntValue = (int *)malloc(sizeof(int));
	//	assert(false);
	//} 
	//else if (0 == valueof.compare ("UNIT")) {
	//	m_ValueOf = UNIT;
	//	m_ValueType = Enums::SAMPLER;
	//	m_Cardinality = 1;
	//	m_IntValue = (int *)malloc(sizeof(int));
	//	assert(false);
	//} 
	//else if (0 == valueof.compare("COUNT")) {
	//	m_ValueOf = COUNT;
	//	m_ValueType = Enums::INT;
	//	m_Cardinality = 1;
	//	m_IntValue = (int *)malloc(sizeof(int) * m_Cardinality);
	//	assert(false);
	//}

	//else if (0 == valueof.compare ("POSITION")) {
	//	m_ValueOf = POSITION;
	//	m_ValueType = Enums::VEC3;
	//	m_Cardinality = 3;
	//	m_Value = (float *)malloc(sizeof(float) * m_Cardinality);
	//} 
	//else if (0 == valueof.compare ("DIRECTION")) {
	//	m_ValueOf = DIRECTION;
	//	m_ValueType = Enums::VEC4;
	//	m_Cardinality = 4;
	//	m_Value = (float *)malloc(sizeof(float) * m_Cardinality);
	//} 
	//else if (0 == valueof.compare ("VIEW")) {
	//	m_ValueOf = VIEW;
	//	m_ValueType = Enums::VEC3;
	//	m_Cardinality = 3;
	//	m_Value = (float *)malloc(sizeof(float) * m_Cardinality);
	//} 
	//else if (0 == valueof.compare ("UP")) {
	//	m_ValueOf = UP;
	//	m_ValueType = Enums::VEC3;
	//	m_Cardinality = 3;
	//	m_Value = (float *)malloc(sizeof(float) * m_Cardinality);
	//} 
	//else if (0 == valueof.compare ("DIFFUSE")) {
	//	m_ValueOf = DIFFUSE;
	//	m_ValueType = Enums::VEC4;
	//	m_Cardinality = 4;
	//	m_Value = (float *)malloc(sizeof(float) * m_Cardinality);
	//} 
	//else if (0 == valueof.compare ("AMBIENT")) {
	//	m_ValueOf = AMBIENT;
	//	m_ValueType = Enums::VEC4;
	//	m_Cardinality = 4;
	//	m_Value = (float *)malloc(sizeof(float) * m_Cardinality);
	//} 
	//else if (0 == valueof.compare ("SPECULAR")) {
	//	m_ValueOf = SPECULAR;
	//	m_ValueType = Enums::VEC4;
	//	m_Cardinality = 4;
	//	m_Value = (float *)malloc(sizeof(float) * m_Cardinality);
	//} 
	//else if (0 == valueof.compare ("EMISSION")) {
	//	m_ValueOf = EMISSION;
	//	m_ValueType = Enums::VEC4;
	//	m_Cardinality = 4;
	//	m_Value = (float *)malloc(sizeof(float) * m_Cardinality);
	//} 
	//else if (0 == valueof.compare ("SHININESS")) {
	//	m_ValueOf = SHININESS;
	//	m_ValueType = Enums::FLOAT;
	//	m_Cardinality = 1;
	//	m_Value = (float *)malloc(sizeof(float) * m_Cardinality);
	//} 
	//else if (0 == valueof.compare ("NORMAL_MATRIX")) {
	//	m_ValueOf = NORMALMATRIX;
	//	m_ValueType = Enums::MAT3;
	//	m_Cardinality = 9;
	//	m_Value = (float *)malloc(sizeof(float) * m_Cardinality);
	//} 
	//else if (0 == valueof.compare ("PROJECTION_VIEW_MATRIX")) {
	//	m_ValueOf = PROJECTIONVIEWMATRIX;
	//	m_ValueType = Enums::MAT4;
	//	m_Cardinality = 16;
	//	m_Value = (float *)malloc(sizeof(float) * m_Cardinality);
	//} 
	//else if (0 == valueof.compare ("VIEW_MATRIX")) {
	//	m_ValueOf = VIEWMATRIX;
	//	m_ValueType = Enums::MAT4;
	//	m_Cardinality = 16;
	//	m_Value = (float *)malloc(sizeof(float) * m_Cardinality);
	//} 
	//else if (0 == valueof.compare ("MODEL_MATRIX")) {
	//	m_ValueOf = MODELMATRIX;
	//	m_ValueType = Enums::MAT4;
	//	m_Cardinality = 16;
	//	m_Value = (float *)malloc(sizeof(float) * m_Cardinality);
	//} 
	//else if (0 == valueof.compare ("PROJECTION_MATRIX")) {
	//	m_ValueOf = PROJECTIONMATRIX;
	//	m_ValueType = Enums::MAT4;
	//	m_Cardinality = 16;
	//	m_Value = (float *)malloc(sizeof(float) * m_Cardinality);
	//} 
	//else if (0 == valueof.compare ("VIEW_MODEL_MATRIX")) {
	//	m_ValueOf = VIEWMODELMATRIX;
	//	m_ValueType = Enums::MAT4;
	//	m_Cardinality = 16;
	//	m_Value = (float *)malloc(sizeof(float) * m_Cardinality);
	//} 
	//else if (0 == valueof.compare ("PROJECTION_VIEW_MODEL_MATRIX")) {
	//	m_ValueOf = PROJECTIONVIEWMODELMATRIX;
	//	m_ValueType = Enums::MAT4;
	//	m_Cardinality = 16;
	//	m_Value = (float *)malloc(sizeof(float) * m_Cardinality);
	//} 
	//else if (0 == valueof.compare ("TEXTURE_MATRIX")) {
	//	m_ValueOf = TEXTUREMATRIX;
	//	m_ValueType = Enums::MAT4;
	//	m_Cardinality = 16;
	//	m_Value = (float *)malloc(sizeof(float) * m_Cardinality);
	//} 
	//else if (0 == valueof.compare ("TS05_MVPMATRIX")) {
	//	m_ValueOf = TS05_MVPMATRIX;
	//	m_ValueType = Enums::MAT4;
	//	m_Cardinality = 16;
	//	m_Value = (float *)malloc(sizeof(float) * m_Cardinality);
	//}
	//else if (0 == valueof.compare ("MAT4")) {
	//	m_ValueOf = USERDATA;
	//	m_ValueType = Enums::MAT4;
	//	m_Cardinality = 16;
	//	m_Value = (float *)malloc(sizeof(float) * m_Cardinality);
	////  m_Value = textutil::ParseFloats (m_Context);
	//} 
	//else if (0 == valueof.compare ("MAT3")) {
	//	m_ValueOf = USERDATA;
	//	m_ValueType = Enums::MAT3;
	//	m_Cardinality = 9;
	//	m_Value = (float *)malloc(sizeof(float) * m_Cardinality);
	////  m_Value = textutil::ParseFloats (m_Context);
	//} 
	//else if (0 == valueof.compare ("MAT2")) {
	//	m_ValueOf = USERDATA;
	//	m_ValueType = Enums::MAT2;
	//	m_Cardinality = 4;
	//	m_Value = (float *)malloc(sizeof(float) * m_Cardinality);
	////  m_Value = textutil::ParseFloats (m_Context);
	//} 
	//else if (0 == valueof.compare ("FLOAT")) {
	//	m_ValueOf = USERDATA;
	//	m_ValueType = Enums::FLOAT;
	//	m_Cardinality = 1;
	//	m_Value = (float *)malloc(sizeof(float) * m_Cardinality);
	//	m_Value[0] = textutil::ParseFloat (m_Context);
	//}
	//else if (0 == valueof.compare ("VEC2")) {
	//	m_ValueOf = USERDATA;
	//	m_ValueType = Enums::VEC2;
	//	m_Cardinality = 2;
	//	m_Value = (float *)malloc(sizeof(float) * m_Cardinality);
	////  m_Value = textutil::ParseFloats (m_Context);
	//}
	//else if (0 == valueof.compare ("VEC3")) {
	//	m_ValueOf = USERDATA;
	//	m_ValueType = Enums::VEC3;
	//	m_Cardinality = 3;
	//	m_Value = (float *)malloc(sizeof(float) * m_Cardinality);
	////  m_Value = textutil::ParseFloats (m_Context);
	//}
	//else if (0 == valueof.compare ("VEC4")) {
	//	m_ValueOf = USERDATA;
	//	m_ValueType = Enums::VEC4;
	//	m_Cardinality = 4;
	//	m_Value = (float *)malloc(sizeof(float) * m_Cardinality);
	////  m_Value = textutil::ParseFloats (m_Context);
	//}
	//else if (0 == valueof.compare ("SAMPLER")) {
	//	m_ValueOf = USERDATA;
	//	m_ValueType = Enums::SAMPLER;
	//	m_Cardinality = 1;
	//	m_IntValue = (int *)malloc(sizeof(int) * m_Cardinality);
	//	m_IntValue[0] = textutil::ParseInt (m_Context);
	//}
	//else if (0 == valueof.compare ("INT")) {
	//	m_ValueOf = USERDATA;
	//	m_ValueType = Enums::INT;
	//	m_Cardinality = 1;
	//	m_IntValue = (int *)malloc(sizeof(int) * m_Cardinality);
	//	m_IntValue[0] = textutil::ParseInt (m_Context);
	//}
	//else if (0 == valueof.compare ("BOOL")) {
	//	m_ValueOf = USERDATA;
	//	m_ValueType = Enums::BOOL;
	//	m_Cardinality = 1;
	//	m_IntValue = (int *)malloc(sizeof(int) * m_Cardinality);
	//	m_IntValue[0] = textutil::ParseInt (m_Context);
	//}
	//else if (0 == valueof.compare ("INT_VEC2")) {
	//	m_ValueOf = USERDATA;
	//	m_ValueType = Enums::IVEC2;
	//	m_Cardinality = 2;
	//	m_IntValue = (int *)malloc(sizeof(int) * m_Cardinality);
	////  m_IntValue = textutil::ParseInts (m_Context);
	//}
	//else if (0 == valueof.compare ("BOOL_VEC2")) {
	//	m_ValueOf = USERDATA;
	//	m_ValueType = Enums::BVEC2;
	//	m_Cardinality = 2;
	//	m_IntValue = (int *)malloc(sizeof(int) * m_Cardinality);
	////  m_IntValue = textutil::ParseInts (m_Context);
	//}
	//else if (0 == valueof.compare ("INT_VEC3")) {
	//	m_ValueOf = USERDATA;
	//	m_ValueType = Enums::IVEC3;
	//	m_Cardinality = 3;
	//	m_IntValue = (int *)malloc(sizeof(int) * m_Cardinality);
	////  m_IntValue = textutil::ParseInts (m_Context);
	//}
	//else if (0 == valueof.compare ("BOOL_VEC3")) {
	//	m_ValueOf = USERDATA;
	//	m_ValueType = Enums::BVEC3;
	//	m_Cardinality = 3;
	//	m_IntValue = (int *)malloc(sizeof(int) * m_Cardinality);
	////  m_IntValue = textutil::ParseInts (m_Context);
	//}
	//else if (0 == valueof.compare ("INT_VEC4")) {
	//	m_ValueOf = USERDATA;
	//	m_ValueType = Enums::IVEC4;
	//	m_Cardinality = 4;
	//	m_IntValue = (int *)malloc(sizeof(int) * m_Cardinality);
	////  m_IntValue = textutil::ParseInts (m_Context);
	//}
	//else if (0 == valueof.compare ("BOOL_VEC4")) {
	//	m_ValueOf = USERDATA;
	//	m_ValueType = Enums::BVEC4;
	//	m_Cardinality = 4;
	//	m_IntValue = (int *)malloc(sizeof(int) * m_Cardinality);
	////  m_IntValue = textutil::ParseInts (m_Context);
	//}

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

	if (pv.m_Value) {
		for (int i = 0 ; i < m_Cardinality; i++)
			m_Value[i] = pv.m_Value[i];
	}
	if (pv.m_IntValue) {
		for (int i = 0 ; i < m_Cardinality; i++)
			m_IntValue[i] = pv.m_IntValue[i];
	}
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
ProgramValue::setValueType(nau::Enums::DataType s) {

	m_ValueType = s;
}

void 
ProgramValue::setValueOfUniform (int *values) { 

	for (int i = 0 ; i < m_Cardinality; i++) {
		m_IntValue[i] = (int)values[i];	
	}
}

// this version is more generic than the previous one
// this is on purpose!
void 
ProgramValue::setValueOfUniform (float *values) { 

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



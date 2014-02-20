#include <nau/material/colormaterial.h>

#include <nau.h>


using namespace nau::material;
using namespace nau::render;


const std::string ColorMaterial::ColorString[] = {"AMBIENT", "SPECULAR", "EMISSION", "DIFFUSE",
													"AMBIENT_AND_DIFFUSE", "SHININESS"};

bool 
ColorMaterial::validateComponent(std::string s)
{
	for (int i = 0; i < COUNT_COLORCOMPONENTS; i++) {

		if (s == ColorString[i]) {
		
			return true;
		}
	}
	return false;
	
}

void
ColorMaterial::getComponentTypeAndId(std::string s, nau::Enums::DataType *dt, ColorComponent *id) 
{
	if ("SHININESS" == s) {

		*id = SHININESS;
		*dt = Enums::FLOAT;
		return;
	}

	for (int i = 0; i < COUNT_COLORCOMPONENTS; i++) {

		if (s == ColorString[i]) {
		
			*id = (ColorComponent)i;
			*dt = Enums::VEC4;
			return;
		}
	}
	NAU_THROW("Invalid Color Component: %s", s);
}



ColorMaterial::ColorMaterial() 
{
  clear();
}

ColorMaterial::~ColorMaterial()
{
   //dtor
}


void
ColorMaterial::setColorComponent(ColorComponent c, float r, float g, float b, float a) 
{
	switch(c) {
	
		case AMBIENT: setAmbient(r,g,b,a); break;
		case SPECULAR: setSpecular(r,g,b,a); break;
		case EMISSION: setEmission(r,g,b,a); break;
		case DIFFUSE: setDiffuse(r,g,b,a); break;
		case SHININESS: setShininess(r); break;
		case AMBIENT_AND_DIFFUSE: 
			setAmbient(r,g,b,a);
			setDiffuse(r,g,b,a);
			break;
	}	
}



float *
ColorMaterial::getColorCompoment(ColorComponent c) {

	switch(c) {
	
		case AMBIENT: return m_Ambient; break;
		case SPECULAR: return m_Specular; break;
		case EMISSION: return m_Emission;  break;
		case DIFFUSE: return m_Diffuse; break;
		case SHININESS: return &m_Shininess; break;
		case AMBIENT_AND_DIFFUSE: return m_Diffuse; break;
	}	
	return NULL;
}


void
ColorMaterial::setColorComponent(ColorComponent c, float *f)
{
	if (c == SHININESS)
		setColorComponent(c, f[0]);
	else
		setColorComponent(c, f[0], f[1], f[2], f[3]);
}



void 
ColorMaterial::setAmbient (const float *values)
{
   this->setAmbient (values[0], values[1], values[2], values[3]);
}

void 
ColorMaterial::setAmbient (float r, float g, float b, float a)
{
   this->m_Ambient[0] = r;
   this->m_Ambient[1] = g;
   this->m_Ambient[2] = b;
   this->m_Ambient[3] = a;
}

const float* 
ColorMaterial::getAmbient (void) const
{
   return this->m_Ambient;
}

void 
ColorMaterial::setSpecular (const float *values)
{
   this->setSpecular (values[0], values[1], values[2], values[3]);
}

void 
ColorMaterial::setSpecular (float r, float g, float b, float a)
{
   this->m_Specular[0] = r;
   this->m_Specular[1] = g;
   this->m_Specular[2] = b;
   this->m_Specular[3] = a;
}

const float* 
ColorMaterial::getSpecular(void) const
{
   return this->m_Specular;
}

void 
ColorMaterial::setEmission (const float *values)
{
   this->setEmission (values[0], values[1], values[2], values[3]);
}

void 
ColorMaterial::setEmission (float r, float g, float b, float a)
{
   this->m_Emission[0] = r;
   this->m_Emission[1] = g;
   this->m_Emission[2] = b;
   this->m_Emission[3] = a;
}

const float* 
ColorMaterial::getEmission (void) const
{
   return this->m_Emission;
}

void 
ColorMaterial::setDiffuse (const float *values)
{
   this->setDiffuse (values[0], values[1], values[2], values[3]);
}

void 
ColorMaterial::setDiffuse (float r, float g, float b, float a)
{
   this->m_Diffuse[0] = r;
   this->m_Diffuse[1] = g;
   this->m_Diffuse[2] = b;
   this->m_Diffuse[3] = a;
}

const float* 
ColorMaterial::getDiffuse (void) const
{
   return this->m_Diffuse;
}


void 
ColorMaterial::setShininess (float shininess)
{
   this->m_Shininess = shininess;
}

const float 
ColorMaterial::getShininess () const
{
   return this->m_Shininess;
}

float *
ColorMaterial::getShininessPtr()
{
	return &(this->m_Shininess);
}

void 
ColorMaterial::prepare ()
{
	RENDERER->setMaterial(*this);
	//RENDERER->setMaterial (IRenderer::FRONT_AND_BACK, IRenderer::AMBIENT, m_Ambient);
	//RENDERER->setMaterial (IRenderer::FRONT_AND_BACK, IRenderer::DIFFUSE, m_Diffuse); 
	//RENDERER->setMaterial (IRenderer::FRONT_AND_BACK, IRenderer::SPECULAR, m_Specular);
	//RENDERER->setMaterial (IRenderer::FRONT_AND_BACK, IRenderer::EMISSION, m_Emissive);
	//RENDERER->setMaterial (IRenderer::FRONT_AND_BACK, IRenderer::SHININESS, m_Shininess);
}

void 
ColorMaterial::restore()
{
	float ambient[4] = {0.2f, 0.2f, 0.2f, 1.0f};

	float specular[4] = {0.0f, 0.0f, 0.0f, 1.0f};

	float diffuse[4] = {0.8f, 0.8f, 0.8f, 1.0f};

	float emission[4] = {0.0f, 0.0f, 0.0f, 1.0f};

	float shininess = 0.0f;

	RENDERER->setMaterial(diffuse, ambient, emission, specular, shininess);

	//RENDERER->setMaterial (IRenderer::FRONT_AND_BACK, IRenderer::AMBIENT, ambient);
	//RENDERER->setMaterial (IRenderer::FRONT_AND_BACK,IRenderer::DIFFUSE, diffuse); 
	//RENDERER->setMaterial (IRenderer::FRONT_AND_BACK,IRenderer::SPECULAR, specular);
	//RENDERER->setMaterial (IRenderer::FRONT_AND_BACK,IRenderer::EMISSION, emissive);
	//RENDERER->setMaterial (IRenderer::FRONT_AND_BACK,IRenderer::SHININESS, shininess);
}

void
ColorMaterial::clear()
{
   // Using default values from OpenGL //
   this->m_Ambient[0] = 0.2f;
   this->m_Ambient[1] = 0.2f;
   this->m_Ambient[2] = 0.2f;
   this->m_Ambient[3] = 1.0f;

   this->m_Specular[0] = 0.0f;
   this->m_Specular[1] = 0.0f;
   this->m_Specular[2] = 0.0f;
   this->m_Specular[3] = 1.0f;

   this->m_Diffuse[0] = 0.8f;
   this->m_Diffuse[1] = 0.8f;
   this->m_Diffuse[2] = 0.8f;
   this->m_Diffuse[3] = 1.0f;

   this->m_Emission[0] = 0.0f;
   this->m_Emission[1] = 0.0f;
   this->m_Emission[2] = 0.0f;
   this->m_Emission[3] = 1.0f;

   this->m_Shininess = 0.0f;
  
}


void 
ColorMaterial::clone(const ColorMaterial &mat)
{
	const float * c = mat.getAmbient();
	this->m_Ambient[0] = c[0];
	this->m_Ambient[1] = c[1];
	this->m_Ambient[2] = c[2];
	this->m_Ambient[3] = c[3];

	c = mat.getSpecular();
	this->m_Specular[0] = c[0];
	this->m_Specular[1] = c[1];
	this->m_Specular[2] = c[2];
	this->m_Specular[3] = c[3];

	c = mat.getDiffuse();
	this->m_Diffuse[0] = c[0];
	this->m_Diffuse[1] = c[1];
	this->m_Diffuse[2] = c[2];
	this->m_Diffuse[3] = c[3];


	c = mat.getEmission();
	this->m_Emission[0] = c[0];
	this->m_Emission[1] = c[1];
	this->m_Emission[2] = c[2];
	this->m_Emission[3] = c[3];

	const float s = mat.getShininess();
	this->m_Shininess = s;
}
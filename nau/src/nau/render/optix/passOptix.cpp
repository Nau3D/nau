#include "nau/config.h"

#if NAU_OPTIX == 1

#include "nau/render/optix/passOptix.h"

#include "nau.h"
#include "nau/slogger.h"
#include "nau/debug/profile.h"
#include "nau/geometry/axis.h"
#include "nau/geometry/frustum.h"
#include "nau/render/passFactory.h"


#include <glbinding/gl/gl.h>
using namespace gl;
//#include <GL/glew.h>

#include <sstream>
#include <algorithm>

using namespace nau::material;
using namespace nau::scene;
using namespace nau::render;
using namespace nau::render::optixRender;
using namespace nau::geometry;
	

bool PassOptix::Inited = PassOptix::Init();


bool
PassOptix::Init() {

	PASSFACTORY->registerClass("optix", Create);
	return true;
}


PassOptix::PassOptix(const std::string &passName) : 
	Pass(passName), o_OutputBuffer(0), o_OutputPBO(0)
{
	try {
		OptixRenderer::Init();
		o_Context = OptixRenderer::GetContext();
		o_EntryPoint = OptixRenderer::GetNextAvailableEntryPoint();

		o_Context->setStackSize(2048);
		o_Context->setExceptionEnabled(RT_EXCEPTION_ALL,false);
		o_BufferLib.setContext(o_Context);

		o_TexLib.setContext(o_Context);

		o_MatLib.setContext(o_Context);
		o_MatLib.setTextureLib(&o_TexLib);

		o_GeomLib.setContext(o_Context);
		o_GeomLib.setBufferLib(&o_BufferLib);
		o_GeomLib.setMaterialLib(&o_MatLib);

		o_OptixIsPrepared = false;
	}
	catch(optix::Exception& e) {	
		NAU_THROW("Optix Error: pass %s [%s]", m_Name.c_str(), e.getErrorString().c_str());
	}
}


std::shared_ptr<Pass>
PassOptix::Create(const std::string &passName) {

	return dynamic_pointer_cast<Pass>(std::shared_ptr<PassOptix>(new PassOptix(passName)));
}


void 
PassOptix::setGeometryIntersectProc(std::string file, std::string proc) {

	o_GeomLib.setGeometryIntersectProc(file, proc);
}


void 
PassOptix::setBoundingBoxProc(std::string file, std::string proc) {

	o_GeomLib.setBoundingBoxProc(file, proc);
}


void 
PassOptix::setOptixEntryPointProcedure(OptixRenderer::ProgramTypes type, std::string file, std::string proc) {

	OptixRenderer::SetProgram(type, o_EntryPoint, file, proc );
}


void 
PassOptix::setDefaultMaterialProc(OptixMaterialLib::MaterialPrograms type, std::string rayType, std::string file, std::string proc) {

	if (o_RayType.count(rayType) == 0)
		o_RayType[rayType] = OptixRenderer::GetNextAvailableRayType();
	o_Context[rayType]->setInt(o_RayType[rayType]);

	o_MatLib.setMaterialProgram(type, o_RayType[rayType],  file, proc );
}


void 
PassOptix::setMaterialProc(std::string name, OptixMaterialLib::MaterialPrograms type, std::string rayType, std::string file, std::string proc) {

	if (o_RayType.count(rayType) == 0)
		o_RayType[rayType] = OptixRenderer::GetNextAvailableRayType();
	o_Context[rayType]->setInt(o_RayType[rayType]);

	o_MatLib.setMaterialProgram(name, type, o_RayType[rayType],  file, proc );
}


void 
PassOptix::setInputBuffer(std::string optixVar, std::string buffer) {

	o_InputBuffers[optixVar] = buffer;
}


void
PassOptix::setOutputBuffer(std::string optixVar, std::string buffer) {

	databuffer db;
	db.pbo = 0;
	db.texName = buffer;
	o_OutputDataBuffer[optixVar] = db;
}


void 
PassOptix::addVertexAttribute(unsigned int attr) {

	o_GeomLib.addVertexAttribute(attr);
}


void 
PassOptix::addMaterialAttribute(std::string name, ProgramValue &p) {

	o_MatLib.addMaterialAttribute(name,p);
}


void 
PassOptix::addGlobalAttribute(std::string name, nau::material::ProgramValue &p) {

	o_GlobalAttribute[name] = p;
}


PassOptix::~PassOptix() {

	OptixRenderer::Terminate();
}


void 
PassOptix::setRenderTarget (nau::render::IRenderTarget* rt)
{
	glGetError();
	if (rt == NULL) {
		//if (m_RenderTarget != NULL) 
		//	delete m_Viewport;
		m_UseRT = true;
	}
	else {
		if (m_RenderTarget == NULL){
			std::string s = "__" + m_Name;
			m_Viewport = RENDERMANAGER->createViewport(s);
			m_UseRT = true;
		}
		setRTSize(rt->getPropui2(IRenderTarget::SIZE));
		m_Viewport->setPropf4(Viewport::CLEAR_COLOR, rt->getPropf4(IRenderTarget::CLEAR_VALUES));
	}
	m_RenderTarget = rt;

	unsigned int n =  rt->getNumberOfColorTargets();

	o_OutputPBO.resize(n);
	glGenBuffers(n, (unsigned int *)&o_OutputPBO[0]);
	nau::material::ITexture* texID;
	
	try {
		for (unsigned int i = 0; i < n; ++i) {

			texID = rt->getTexture(i);
			int format = texID->getPrope(ITexture::FORMAT);

			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, o_OutputPBO[i]);
			// need to allow different types
			nau::math::uivec2 vec2;
			vec2 = rt->getPropui2(IRenderTarget::SIZE);
			glBufferData(GL_PIXEL_UNPACK_BUFFER, vec2.x * vec2.y * rt->getTexture(i)->getPropi(ITexture::ELEMENT_SIZE)/8, 0, GL_STREAM_READ);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

			optix::Buffer b = o_Context->createBufferFromGLBO(RT_BUFFER_OUTPUT, o_OutputPBO[i]);
			b->setSize(vec2.x, vec2.y);
			// same here (types)
			b->setFormat(getOptixFormat(rt->getTexture(i)));

			// project loader must deal with named outputs
			std::ostringstream bufferName;
			bufferName << "output" << i;
			o_Context[bufferName.str()]->setBuffer(b);
			o_OutputBuffer.push_back(b);
			//
		}
	}
	catch(optix::Exception& e) {
	
		NAU_THROW("Optix Error: setting render target %s in pass %s [%s]", 
			rt->getName().c_str(), m_Name.c_str(), e.getErrorString().c_str());
	}
	// what about the depth target?

}


void
PassOptix::prepare (void)
{
	if (!o_OptixIsPrepared)
		optixInit();

	setupCamera();

	setupLights();

	std::map<std::string, nau::material::ProgramValue>::iterator iter;

	iter = o_GlobalAttribute.begin();
	for ( ; iter != o_GlobalAttribute.end(); ++iter) {
		void *k = iter->second.getValues();
		if (k != NULL) {

			Enums::DataType dt = iter->second.getValueType();
			switch (dt) {
			case Enums::INT:
			case Enums::BOOL:
			case Enums::FLOAT:
			case Enums::UINT:
			case Enums::SAMPLER:
			case Enums::DOUBLE:
				break;
			default:
				k = ((Data *)k)->getPtr();
			}


			unsigned int *j;

			switch (dt) {
					
				case Enums::UINT:
					j = (unsigned int *)iter->second.getValues();
					o_Context[iter->first]->set1uiv((unsigned int *)k);
					break;
				case Enums::INT:
				case Enums::BOOL:
				case Enums::ENUM:
					o_Context[iter->first]->set1iv((int *)k);
					break;
				case Enums::IVEC2:
				case Enums::BVEC2:
					o_Context[iter->first]->set2iv((int *)k);
					break;
				case Enums::IVEC3:
				case Enums::BVEC3:
					o_Context[iter->first]->set3iv((int *)k);
					break;
				case Enums::IVEC4:
				case Enums::BVEC4:
					o_Context[iter->first]->set4iv((int *)k);
					break;

				case Enums::FLOAT:
					o_Context[iter->first]->set1fv((float *)k);
					break;
				case Enums::VEC2:
					o_Context[iter->first]->set2fv((float *)k);
					break;
				case Enums::VEC3:
					o_Context[iter->first]->set3fv((float *)k);
					break;
				case Enums::VEC4:
					o_Context[iter->first]->set4fv((float *)k);
					break;
				case Enums::MAT2:
					o_Context[iter->first]->setMatrix2x2fv(false,(float *)k);
					break;
				case Enums::MAT3:
					o_Context[iter->first]->setMatrix3x3fv(false,(float *)k);
					break;
				case Enums::MAT4:
					o_Context[iter->first]->setMatrix4x4fv(false,(float *)k);
					break;
				default:
					continue;
			}
		}
	}
}


void
PassOptix::restore (void) {

	restoreCamera();
	RENDERER->removeLights();
}


void
PassOptix::doPass (void) {

	glGetError();
	glFinish();

	nau::math::uivec2 v2 = m_RenderTarget->getPropui2(IRenderTarget::SIZE);
	try {
		PROFILE("Optix");

		o_Context->validate();
		o_Context->launch(0, v2.x, v2.y);
	} 
	catch(optix::Exception& e) {
		NAU_THROW("Optix Error: Launching Kernel in pass %s [%s]", m_Name.c_str(), e.getErrorString().c_str());
	}

	//COPY OPTIX OUTPUT TO RENDER TARGET TEXTURES

	for (unsigned int i = 0; i < m_RenderTarget->getNumberOfColorTargets(); ++i) {
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, o_OutputPBO[i]);
		glBindTexture(GL_TEXTURE_2D, m_RenderTarget->getTexture(i)->getPropi(ITexture::ID));
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 
						v2.x, v2.y,
						(GLenum)m_RenderTarget->getTexture(i)->getPrope(ITexture::FORMAT),
						(GLenum)m_RenderTarget->getTexture(i)->getPrope(ITexture::TYPE), 0);
		glBindTexture(GL_TEXTURE_2D, 0);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	}

//	std::map<std::string, databuffer>::iterator iter;
//	iter = o_OutputDataBuffer.begin();
//	for ( ; iter != o_OutputDataBuffer.end(); ++iter) {
//	
//		ITexture *t = RESOURCEMANAGER->getTexture(iter->second.texName);
//		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, iter->second.pbo);
//		glBindTexture(GL_TEXTURE_2D, t->getPropui(ITexture::ID));
//		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
//		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 
//						t->getPropi(ITexture::WIDTH), t->getPropi(ITexture::HEIGHT),
//						t->getPrope(ITexture::FORMAT),
//						t->getPrope(ITexture::TYPE), 0);
//
//		//glGetTexImage(	GL_TEXTURE_2D, 0, t->getPrope(ITexture::FORMAT), t->getPrope(ITexture::TYPE), m);
// 
//		glBindTexture(GL_TEXTURE_2D, 0);
//#if NAU_OPENGL_VERSION >= 430
//		GLubyte zero = 0;
//		glClearBufferData(GL_PIXEL_UNPACK_BUFFER, GL_R8, GL_RED, GL_UNSIGNED_BYTE, &zero);
//#else
//		//SLOG("%f %f %f %f", m[0], m[1], m[2], m[3]);
//		void *m;
//		m = malloc(t->getPropi(ITexture::ELEMENT_SIZE) * t->getPropi(ITexture::WIDTH)*t->getPropi(ITexture::HEIGHT) );
//		memset(m, 0, t->getPropi(ITexture::ELEMENT_SIZE) * t->getPropi(ITexture::WIDTH)*t->getPropi(ITexture::HEIGHT) );
//		glBufferData(GL_PIXEL_UNPACK_BUFFER, t->getPropi(ITexture::WIDTH)*t->getPropi(ITexture::HEIGHT)*t->getPropi(ITexture::ELEMENT_SIZE), m, GL_STREAM_READ);
//#endif
//		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
//	}



}


void
PassOptix::setupCamera (void) {

	std::shared_ptr<Camera> &aCam = RENDERMANAGER->getCamera (m_StringProps[CAMERA]);
	
	if (m_ExplicitViewport) {
		m_RestoreViewport = aCam->getViewport();
		aCam->setViewport (m_Viewport);
	}
	
	RENDERER->setCamera (aCam);

	const vec4 &v1 = aCam->getPropf4(Camera::POSITION);
	o_Context["eye"]->setFloat(v1.x, v1.y, v1.z);
	const vec4 &v2 = aCam->getPropf4(Camera::NORMALIZED_UP_VEC);
	o_Context["V"]->setFloat(v2.x, v2.y, v2.z);
	const vec4 &v3 = aCam->getPropf4(Camera::NORMALIZED_RIGHT_VEC);
	o_Context["U"]->setFloat(v3.x, v3.y, v3.z);
	const vec4 &v4 = aCam->getPropf4(Camera::VIEW_VEC);
	o_Context["W"]->setFloat(v4.x, v4.y, v4.z);
	float fov = aCam->getPropf(Camera::FOV) * 0.5f;
	o_Context["fov"]->setFloat(tan(fov*3.14159f/180.0f));
}


void
PassOptix::optixInit() {

	glGetError();

	// Can these be removed due to texture count?
	o_TexLib.addTexture(1);
	o_Context["tex0"]->set(o_TexLib.getTexture(1));

	/*std::set<std::string> *materialNames = new std::set<std::string>;*/
	
	std::vector<std::string>::iterator scenesIter;
	scenesIter = m_SceneVector.begin();
	for ( ; scenesIter != m_SceneVector.end(); ++scenesIter) {

		std::shared_ptr<IScene> &aScene = RENDERMANAGER->getScene (*scenesIter);
		aScene->compile();

		// Adding Materials to optix lib
		const std::set<std::string> &materialNames = aScene->getMaterialNames();
		
		for( auto matIter : materialNames) {
			o_MatLib.addMaterial(m_MaterialMap[matIter]);
		}

		std::vector<std::shared_ptr<SceneObject>> objs;
		aScene->getAllObjects(&objs);

		for (auto &so: objs) {
			o_GeomLib.addSceneObject(so, m_MaterialMap);
//			o_GeomLib.addSceneObject((*objsIter)->getId(), m_MaterialMap);
		}
	}
	o_GeomLib.buildGeometryGroup();
	o_Context["top_object"]->set(o_GeomLib.getGeometryGroup());
	

	o_MatLib.applyMissPrograms();


	std::map<std::string, std::string>::iterator iter;
	iter = o_InputBuffers.begin();
	for ( ; iter != o_InputBuffers.end(); ++iter) {
		try {
			unsigned int id = RESOURCEMANAGER->getTexture(iter->second)->getPropi(ITexture::ID);
			if (RESOURCEMANAGER->getTexture(iter->second)->getPrope(ITexture::DIMENSION) == GL_TEXTURE_2D) {
				optix::TextureSampler rtWorldSpaceTexture = o_Context->createTextureSamplerFromGLImage(id, RT_TARGET_GL_TEXTURE_2D);
				rtWorldSpaceTexture->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
				rtWorldSpaceTexture->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
				rtWorldSpaceTexture->setIndexingMode(RT_TEXTURE_INDEX_ARRAY_INDEX);
				rtWorldSpaceTexture->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);
				rtWorldSpaceTexture->setMaxAnisotropy(1.0f);
				rtWorldSpaceTexture->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_LINEAR);
				o_Context[iter->first]->setTextureSampler(rtWorldSpaceTexture);
			}
		}
		catch(optix::Exception& e) {
			NAU_THROW("Optix Error: Input Buffer preparation in pass %s [%s]", m_Name.c_str(), e.getErrorString().c_str());
		}
	}

	std::map<std::string, databuffer>::iterator iter2;
	iter2 = o_OutputDataBuffer.begin();
	ITexture *texID;
	try {
		for (; iter2 != o_OutputDataBuffer.end(); ++iter2) {

			texID = RESOURCEMANAGER->getTexture(iter2->second.texName);
			//		int format = texID->getPrope(ITexture::FORMAT);
			int tex = texID->getPropi(ITexture::ID);

			unsigned int pbo;
			glGenBuffers(1, &pbo);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
			// need to allow different types
			glBufferData(GL_PIXEL_UNPACK_BUFFER, texID->getPropi(ITexture::WIDTH)*texID->getPropi(ITexture::HEIGHT)*texID->getPropi(ITexture::ELEMENT_SIZE)/8, 0, GL_STREAM_READ);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

			optix::Buffer ob = o_Context->createBufferFromGLBO(RT_BUFFER_OUTPUT, pbo);
			ob->setSize(texID->getPropi(ITexture::WIDTH), texID->getPropi(ITexture::HEIGHT));
			// same here (types)
			ob->setFormat(getOptixFormat(texID));

			o_Context[iter2->first]->setBuffer(ob);

			o_OutputDataBuffer[iter2->first].pbo = pbo;
			//
		}
	}
	catch (optix::Exception& e) {
		NAU_THROW("Optix Error: Output Buffer preparation in pass %s [%s]", m_Name.c_str(), e.getErrorString().c_str());
	}
	

#if (TEST == 2)
	try {
		unsigned int id = RESOURCEMANAGER->getTexture("Deferred Render Targets::pos")->getId();
		optix::ITextureSampler rtWorldSpaceTexture = o_Context->createTextureSamplerFromGLImage(id, RT_TARGET_GL_TEXTURE_2D);
		rtWorldSpaceTexture->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
		rtWorldSpaceTexture->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
		rtWorldSpaceTexture->setIndexingMode(RT_TEXTURE_INDEX_ARRAY_INDEX);
		rtWorldSpaceTexture->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);
		rtWorldSpaceTexture->setMaxAnisotropy(1.0f);
		rtWorldSpaceTexture->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_LINEAR);
		o_Context["pos_buffer"]->setTextureSampler(rtWorldSpaceTexture);
	}
	catch(optix::Exception& e) {
		NAU_THROW("Optix Error: Input Buffer preparation in pass %s [%s]", m_Name.c_str(), e.getErrorString().c_str());
	}
#endif
	o_OptixIsPrepared = true;
}


void 
PassOptix::addScene (const std::string &sceneName) {

	if (m_SceneVector.end() == std::find (m_SceneVector.begin(), m_SceneVector.end(), sceneName)) {
	
		m_SceneVector.push_back (sceneName);
	
		const std::set<std::string> &materialNames = 
			RENDERMANAGER->getScene(sceneName)->getMaterialNames();
		
		for (auto iter : materialNames) {
			
			if (m_MaterialMap.count(iter) == 0)
				m_MaterialMap[iter] = MaterialID(DEFAULTMATERIALLIBNAME, iter);
		}

		std::shared_ptr<IScene> &sc = RENDERMANAGER->getScene(sceneName);
		sc->compile();
	}
}

RTformat
PassOptix::getOptixFormat(ITexture *t) {

	int nComp = t->getPropi(ITexture::COMPONENT_COUNT);
	int type = t->getPrope(ITexture::TYPE);

	if (type == GL_FLOAT) {
		switch (nComp) {
			case 1: return RT_FORMAT_FLOAT;
			case 2: return RT_FORMAT_FLOAT2;
			case 3: return RT_FORMAT_FLOAT3;
			case 4: return RT_FORMAT_FLOAT4;
		}
	}
	else if (type == GL_UNSIGNED_BYTE) {
		switch (nComp) {
			case 1: return RT_FORMAT_UNSIGNED_BYTE;
			case 2: return RT_FORMAT_UNSIGNED_BYTE2;
			case 3: return RT_FORMAT_UNSIGNED_BYTE3;
			case 4: return RT_FORMAT_UNSIGNED_BYTE4;
		}
	}
	else if (type == GL_UNSIGNED_SHORT) {
		switch (nComp) {
			case 1: return RT_FORMAT_UNSIGNED_SHORT;
			case 2: return RT_FORMAT_UNSIGNED_SHORT2;
			case 3: return RT_FORMAT_UNSIGNED_SHORT3;
			case 4: return RT_FORMAT_UNSIGNED_SHORT4;
		}
	}
	else if (type == GL_UNSIGNED_INT) {
		switch (nComp) {
			case 1: return RT_FORMAT_UNSIGNED_INT;
			case 2: return RT_FORMAT_UNSIGNED_INT2;
			case 3: return RT_FORMAT_UNSIGNED_INT3;
			case 4: return RT_FORMAT_UNSIGNED_INT4;
		}
	}
	else if (type == GL_SHORT) {
		switch (nComp) {
			case 1: return RT_FORMAT_SHORT;
			case 2: return RT_FORMAT_SHORT2;
			case 3: return RT_FORMAT_SHORT3;
			case 4: return RT_FORMAT_SHORT4;
		}
	}
	else if (type == GL_BYTE) {
		switch (nComp) {
			case 1: return RT_FORMAT_BYTE;
			case 2: return RT_FORMAT_BYTE2;
			case 3: return RT_FORMAT_BYTE3;
			case 4: return RT_FORMAT_BYTE4;
		}
	}
	else if (type == GL_INT) {
		switch (nComp) {
			case 1: return RT_FORMAT_INT;
			case 2: return RT_FORMAT_INT2;
			case 3: return RT_FORMAT_INT3;
			case 4: return RT_FORMAT_INT4;
		}
	}
	return RT_FORMAT_FLOAT;

}

#endif
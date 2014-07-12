#ifdef NAU_OPTIX 

#include <nau/render/passOptix.h>

#include <nau/geometry/axis.h>

#include <sstream>
#include <algorithm>

#include <nau.h>
#include <nau/geometry/frustum.h>
#include <nau/debug/profile.h>
#include <nau/slogger.h>

#include <GL/glew.h>

using namespace nau::material;
using namespace nau::scene;
using namespace nau::render;
using namespace nau::render::optixRender;
using namespace nau::geometry;

#define TEST 1	


PassOptix::PassOptix (const std::string &passName) :
	Pass(passName)
{
	try {
		o_EntryPoint = OptixRenderer::getNextAvailableEntryPoint();

	#if (TEST == 1)
	//	OptixRenderer::setProgram(OptixRenderer::RAY_GEN, o_EntryPoint, "optix/common.ptx", "buffer_camera" );
	#elif (TEST == 2)
		OptixRenderer::setProgram(OptixRenderer::RAY_GEN, o_EntryPoint, "optix/common.ptx", "buffer_camera" );
	#endif
		//OptixRenderer::setProgram(OptixRenderer::EXCEPTION, o_EntryPoint, "optix/common.ptx", "exception" );
		o_Context = OptixRenderer::getContext();
		o_Context->setStackSize(2048);
		o_Context->setExceptionEnabled(RT_EXCEPTION_ALL,false);
		o_BufferLib.setContext(o_Context);

		o_TexLib.setContext(o_Context);

		o_MatLib.setContext(o_Context);
		o_MatLib.setTextureLib(&o_TexLib);
	#if (TEST == 1)

		//o_RayType["Phong"] = OptixRenderer::getNextAvailableRayType();
		//o_RayType["Shadow"] = OptixRenderer::getNextAvailableRayType();
		//o_MatLib.setMaterialProgram(o_MatLib.MISS, o_RayType["Phong"], "optix/common.ptx", "miss" );
		//o_MatLib.setMaterialProgram(o_MatLib.CLOSEST_HIT, o_RayType["Phong"],  "optix/common.ptx", "shade" );
		//o_MatLib.setMaterialProgram(o_MatLib.ANY_HIT, o_RayType["Phong"],  "optix/common.ptx", "any_hit" );
		//o_MatLib.setMaterialProgram(o_MatLib.ANY_HIT, o_RayType["Shadow"], "optix/common.ptx", "shadow" );
	#elif (TEST == 2)
		o_RayType["Shadow"] = OptixRenderer::getNextAvailableRayType();
		OptixRenderer::setProgram(OptixRenderer::MISS, o_RayType["Shadow"], "optix/common.ptx", "miss" );
		o_MatLib.setMaterialProgram(o_MatLib.ANY_HIT, o_RayType["Shadow"],  "optix/common.ptx", "any_hit_shadow" );
	#endif
		o_GeomLib.setContext(o_Context);
		o_GeomLib.setBufferLib(&o_BufferLib);
		o_GeomLib.setMaterialLib(&o_MatLib);

		o_OptixIsPrepared = false;
	}
	catch(optix::Exception& e) {	
		NAU_THROW("Optix Error: pass %s [%s]", m_Name.c_str(), e.getErrorString().c_str());
	}


	//try {
	//	o_Context = optix::Context::create();
	//	o_Context->setRayTypeCount(1);
	//	o_Context->setEntryPointCount(1);
	//	o_Context->setExceptionEnabled(RT_EXCEPTION_ALL, 1);

	//	optix::Program ray_gen_program = o_Context->createProgramFromPTXFile( "optix/common.ptx", "pinhole_camera" );
	//	o_Context->setRayGenerationProgram( 0, ray_gen_program );

	//	optix::Program miss_program = o_Context->createProgramFromPTXFile( "optix/common.ptx", "miss" );
	//	o_Context->setMissProgram(0, miss_program);

	//	optix::Program exception_program = o_Context->createProgramFromPTXFile( "optix/common.ptx", "exception" );
	//	o_Context->setExceptionProgram(0, exception_program);

	//	o_Material = o_Context->createMaterial();
	//	o_ClosestHitProgram = o_Context->createProgramFromPTXFile( "optix/common.ptx", "shade");
	//	o_Material->setClosestHitProgram(0, o_ClosestHitProgram);

	//	o_GeomGroup = o_Context->createGeometryGroup();
	//	o_Context["top_object"]->set(o_GeomGroup);

	//	o_GeometryIntersectionProgram = o_Context->createProgramFromPTXFile("optix/common.ptx","geometryintersection");
	//	o_BoundingBoxProgram = o_Context->createProgramFromPTXFile("optix/common.ptx","boundingbox");

	//	o_Context->setStackSize(2048);
	//}
	//catch(optix::Exception& e) {
	//
	//	NAU_THROW("Optix Error: Init pass %s [%s]", m_Name.c_str(), e.getErrorString().c_str());
	//}
}

void 
PassOptix::setGeometryIntersectProc(std::string file, std::string proc){

	o_GeomLib.setGeometryIntersectProc(file, proc);
}


void 
PassOptix::setBoundingBoxProc(std::string file, std::string proc){

	o_GeomLib.setBoundingBoxProc(file, proc);
}


void 
PassOptix::setOptixEntryPointProcedure(OptixRenderer::ProgramTypes type, std::string file, std::string proc) {

	OptixRenderer::setProgram(type, o_EntryPoint, file, proc );
}


void 
PassOptix::setDefaultMaterialProc(OptixMaterialLib::MaterialPrograms type, std::string rayType, std::string file, std::string proc) {

	if (o_RayType.count(rayType) == 0)
		o_RayType[rayType] = OptixRenderer::getNextAvailableRayType();
	o_Context[rayType]->setInt(o_RayType[rayType]);

	o_MatLib.setMaterialProgram(type, o_RayType[rayType],  file, proc );
}


void 
PassOptix::setMaterialProc(std::string name, OptixMaterialLib::MaterialPrograms type, std::string rayType, std::string file, std::string proc) {

	if (o_RayType.count(rayType) == 0)
		o_RayType[rayType] = OptixRenderer::getNextAvailableRayType();
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


PassOptix::~PassOptix()
{
	/***MARK***/ //Delete resources ?
}


void 
PassOptix::setRenderTarget (nau::render::RenderTarget* rt)
{
	glGetError();
	if (rt == NULL) {
		if (m_RenderTarget != NULL) 
			delete m_pViewport;
		m_UseRT = true;
	}
	else {
		if (m_RenderTarget == NULL){
			m_pViewport = new Viewport();
			m_UseRT = true;
		}
		setRTSize(rt->getWidth(), rt->getHeight());
		m_pViewport->setProp(Viewport::CLEAR_COLOR, rt->getClearValues());
	}
	m_RenderTarget = rt;

	unsigned int n =  rt->getNumberOfColorTargets();

	glGenBuffers(n, o_OutputPBO);
	nau::render::Texture* texID;
	
	try {
		for (unsigned int i = 0; i < n; ++i) {

			texID = rt->getTexture(i);
			int format = texID->getPrope(Texture::FORMAT);

			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, o_OutputPBO[i]);
			// need to allow different types
			glBufferData(GL_PIXEL_UNPACK_BUFFER, rt->getWidth()*rt->getHeight()*rt->getTexture(i)->getPropi(Texture::ELEMENT_SIZE), 0, GL_STREAM_READ);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

			o_OutputBuffer[i] = o_Context->createBufferFromGLBO(RT_BUFFER_OUTPUT, o_OutputPBO[i]);
			o_OutputBuffer[i]->setSize(rt->getWidth(), rt->getHeight());
			// same here (types)
			o_OutputBuffer[i]->setFormat(getOptixFormat(rt->getTexture(i)));

			// project loader must deal with named outputs
			std::ostringstream bufferName;
			bufferName << "output" << i;
			o_Context[bufferName.str()]->setBuffer(o_OutputBuffer[i]);
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
		if (iter->second.getValues() != NULL) {
			switch (iter->second.getValueType()) {
							
				case Enums::INT:
				case Enums::BOOL:
				case Enums::ENUM:
					o_Context[iter->first]->set1iv((int *)iter->second.getValues());
					break;
				case Enums::IVEC2:
				case Enums::BVEC2:
					o_Context[iter->first]->set2iv((int *)iter->second.getValues());
					break;
				case Enums::IVEC3:
				case Enums::BVEC3:
					o_Context[iter->first]->set3iv((int *)iter->second.getValues());
					break;
				case Enums::IVEC4:
				case Enums::BVEC4:
					o_Context[iter->first]->set4iv((int *)iter->second.getValues());
					break;

				case Enums::FLOAT:
					o_Context[iter->first]->set1fv((float *)iter->second.getValues());
					break;
				case Enums::VEC2:
					o_Context[iter->first]->set2fv((float *)iter->second.getValues());
					break;
				case Enums::VEC3:
					o_Context[iter->first]->set3fv((float *)iter->second.getValues());
					break;
				case Enums::VEC4:
					o_Context[iter->first]->set4fv((float *)iter->second.getValues());
					break;
				case Enums::MAT2:
					o_Context[iter->first]->setMatrix2x2fv(false,(float *)iter->second.getValues());
					break;
				case Enums::MAT3:
					o_Context[iter->first]->setMatrix3x3fv(false,(float *)iter->second.getValues());
					break;
				case Enums::MAT4:
					o_Context[iter->first]->setMatrix4x4fv(false,(float *)iter->second.getValues());
					break;
				default:
					continue;
			}
		}
	}
}


void
PassOptix::restore (void)
{
	restoreCamera();
	RENDERER->removeLights();
}


bool 
PassOptix::renderTest (void)
{
	return true;
}


void
PassOptix::doPass (void)
{
	glGetError();
	glFinish();

	try {
		PROFILE("Optix");

		o_Context->validate();
		o_Context->launch(0, m_RenderTarget->getWidth(), m_RenderTarget->getHeight());
	} 
	catch(optix::Exception& e) {
		NAU_THROW("Optix Error: Launching Kernel in pass %s [%s]", m_Name.c_str(), e.getErrorString().c_str());
	}

	//COPY OPTIX OUTPUT TO RENDER TARGET TEXTURES

	for (unsigned int i = 0; i < m_RenderTarget->getNumberOfColorTargets(); ++i) {
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, o_OutputPBO[i]);
		glBindTexture(GL_TEXTURE_2D, m_RenderTarget->getTexture(i)->getPropui(Texture::ID));
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 
						m_RenderTarget->getWidth(), m_RenderTarget->getHeight(),
						m_RenderTarget->getTexture(i)->getPrope(Texture::FORMAT),
						m_RenderTarget->getTexture(i)->getPrope(Texture::TYPE), 0);
		glBindTexture(GL_TEXTURE_2D, 0);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	}

//	std::map<std::string, databuffer>::iterator iter;
//	iter = o_OutputDataBuffer.begin();
//	for ( ; iter != o_OutputDataBuffer.end(); ++iter) {
//	
//		Texture *t = RESOURCEMANAGER->getTexture(iter->second.texName);
//		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, iter->second.pbo);
//		glBindTexture(GL_TEXTURE_2D, t->getPropui(Texture::ID));
//		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
//		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 
//						t->getPropi(Texture::WIDTH), t->getPropi(Texture::HEIGHT),
//						t->getPrope(Texture::FORMAT),
//						t->getPrope(Texture::TYPE), 0);
//
//		//glGetTexImage(	GL_TEXTURE_2D, 0, t->getPrope(Texture::FORMAT), t->getPrope(Texture::TYPE), m);
// 
//		glBindTexture(GL_TEXTURE_2D, 0);
//#if NAU_OPENGL_VERSION >= 430
//		GLubyte zero = 0;
//		glClearBufferData(GL_PIXEL_UNPACK_BUFFER, GL_R8, GL_RED, GL_UNSIGNED_BYTE, &zero);
//#else
//		//SLOG("%f %f %f %f", m[0], m[1], m[2], m[3]);
//		void *m;
//		m = malloc(t->getPropi(Texture::ELEMENT_SIZE) * t->getPropi(Texture::WIDTH)*t->getPropi(Texture::HEIGHT) );
//		memset(m, 0, t->getPropi(Texture::ELEMENT_SIZE) * t->getPropi(Texture::WIDTH)*t->getPropi(Texture::HEIGHT) );
//		glBufferData(GL_PIXEL_UNPACK_BUFFER, t->getPropi(Texture::WIDTH)*t->getPropi(Texture::HEIGHT)*t->getPropi(Texture::ELEMENT_SIZE), m, GL_STREAM_READ);
//#endif
//		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
//	}



}


void
PassOptix::setupCamera (void)
{
	Camera *aCam = 0;

	aCam = RENDERMANAGER->getCamera (m_CameraName);
	
	if (0 == aCam) {
		return; 
	}

	if (0 != m_pViewport) {
		m_pRestoreViewport = aCam->getViewport();
		aCam->setViewport (m_pViewport);
	}
	
	RENDERER->setCamera (aCam);

	const vec4 &v1 = aCam->getPropf4(Camera::POSITION);
	o_Context["eye"]->setFloat(v1.x, v1.y, v1.z);
	const vec4 &v2 = aCam->getPropf4(Camera::NORMALIZED_UP_VEC);
	o_Context["V"]->setFloat(v2.x, v2.y, v2.z);
	const vec4 &v3 = aCam->getPropf4(Camera::NORMALIZED_RIGHT_VEC);
	o_Context["U"]->setFloat(v3.x, v3.y, v3.z);
	const vec4 &v4 = aCam->getPropf4(Camera::NORMALIZED_VIEW_VEC);
	o_Context["W"]->setFloat(v4.x, v4.y, v4.z);
	float fov = aCam->getPropf(Camera::FOV) * 0.5;
	o_Context["fov"]->setFloat(tan(fov*3.14159/180.0));
}


void
PassOptix::optixInit() {

	glGetError();

	// Can these be removed due to texture count?
	o_TexLib.addTexture(1);
	o_Context["tex0"]->set(o_TexLib.getTexture(1));

	std::set<std::string> *materialNames = new std::set<std::string>;
	
	std::vector<std::string>::iterator scenesIter;
	scenesIter = m_SceneVector.begin();
	for ( ; scenesIter != m_SceneVector.end(); ++scenesIter) {

		IScene *aScene = RENDERMANAGER->getScene (*scenesIter);
		aScene->compile();

		// Adding Materials to optix lib
		aScene->getMaterialNames(materialNames);
		std::set<std::string>::iterator matIter;
		matIter = materialNames->begin();
		for( ; matIter != materialNames->end(); ++matIter) {
		
			o_MatLib.addMaterial(m_MaterialMap[*matIter]);
		}

		materialNames->clear();


		std::vector<SceneObject *> objs = aScene->getAllObjects();
		std::vector<nau::scene::SceneObject*>::iterator objsIter;

		objsIter = objs.begin();
		for ( ; objsIter != objs.end(); ++objsIter) {
			o_GeomLib.addSceneObject((*objsIter)->getId(), m_MaterialMap);
		}
	}
	o_GeomLib.buildGeometryGroup();
	o_Context["top_object"]->set(o_GeomLib.getGeometryGroup());

	o_MatLib.applyMissPrograms();


	std::map<std::string, std::string>::iterator iter;
	iter = o_InputBuffers.begin();
	for ( ; iter != o_InputBuffers.end(); ++iter) {
		try {
			unsigned int id = RESOURCEMANAGER->getTexture(iter->second)->getPropui(Texture::ID);
			if (RESOURCEMANAGER->getTexture(iter->second)->getPrope(Texture::DIMENSION) == GL_TEXTURE_2D) {
				optix::TextureSampler rtWorldSpaceTexture = o_Context->createTextureSamplerFromGLImage(id, RT_TARGET_GL_TEXTURE_2D);
				rtWorldSpaceTexture->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
				rtWorldSpaceTexture->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
				rtWorldSpaceTexture->setIndexingMode(RT_TEXTURE_INDEX_ARRAY_INDEX);
				rtWorldSpaceTexture->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);
				rtWorldSpaceTexture->setMaxAnisotropy(1.0f);
				rtWorldSpaceTexture->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE);
				o_Context[iter->first]->setTextureSampler(rtWorldSpaceTexture);
			}
		}
		catch(optix::Exception& e) {
			NAU_THROW("Optix Error: Input Buffer preparation in pass %s [%s]", m_Name.c_str(), e.getErrorString().c_str());
		}
	}

	std::map<std::string, databuffer>::iterator iter2;
	iter2 = o_OutputDataBuffer.begin();
	Texture *texID;

	for ( ; iter2 != o_OutputDataBuffer.end() ; ++iter2) {

		texID = RESOURCEMANAGER->getTexture(iter2->second.texName);
//		int format = texID->getPrope(Texture::FORMAT);
		int tex = texID->getPropui(Texture::ID);

		unsigned int pbo;
		glGenBuffers(1, &pbo);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
		// need to allow different types
		glBufferData(GL_PIXEL_UNPACK_BUFFER, texID->getPropi(Texture::WIDTH)*texID->getPropi(Texture::HEIGHT)*texID->getPropi(Texture::ELEMENT_SIZE), 0, GL_STREAM_READ);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		optix::Buffer ob = o_Context->createBufferFromGLBO(RT_BUFFER_OUTPUT, pbo);
		ob->setSize(texID->getPropi(Texture::WIDTH), texID->getPropi(Texture::HEIGHT));
		// same here (types)
		ob->setFormat(getOptixFormat(texID));

		o_Context[iter2->first]->setBuffer(ob);

		o_OutputDataBuffer[iter2->first].pbo = pbo;
		//
	}

#if (TEST == 2)
	try {
		unsigned int id = RESOURCEMANAGER->getTexture("Deferred Render Targets::pos")->getId();
		optix::TextureSampler rtWorldSpaceTexture = o_Context->createTextureSamplerFromGLImage(id, RT_TARGET_GL_TEXTURE_2D);
		rtWorldSpaceTexture->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
		rtWorldSpaceTexture->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
		rtWorldSpaceTexture->setIndexingMode(RT_TEXTURE_INDEX_ARRAY_INDEX);
		rtWorldSpaceTexture->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);
		rtWorldSpaceTexture->setMaxAnisotropy(1.0f);
		rtWorldSpaceTexture->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE);
		o_Context["pos_buffer"]->setTextureSampler(rtWorldSpaceTexture);
	}
	catch(optix::Exception& e) {
		NAU_THROW("Optix Error: Input Buffer preparation in pass %s [%s]", m_Name.c_str(), e.getErrorString().c_str());
	}
#endif
	o_OptixIsPrepared = true;
}


void 
PassOptix::addScene (const std::string &sceneName)
{
	if (m_SceneVector.end() == std::find (m_SceneVector.begin(), m_SceneVector.end(), sceneName)) {
	
		m_SceneVector.push_back (sceneName);
	
		std::set<std::string> *materialNames = new std::set<std::string>;
		RENDERMANAGER->getScene(sceneName)->getMaterialNames(materialNames);
		
		std::set<std::string>::iterator iter;
		iter = materialNames->begin();
		for ( ; iter != materialNames->end(); ++iter) {
			
			if (m_MaterialMap.count((*iter)) == 0)
				m_MaterialMap[(*iter)] = MaterialID(DEFAULTMATERIALLIBNAME, (*iter));
		}
		delete materialNames;


		IScene *sc = RENDERMANAGER->getScene(sceneName);
		sc->compile();

/*		std::vector<SceneObject *> objs = sc->getAllObjects();
		for (unsigned int i = 0; i < objs.size(); ++i) {

			IRenderable &r = objs[i]->getRenderable();
			VertexData &v = r.getVertexData();
			// just to make sure we get this data compiled
			//r.getVertexData().compile();

			// which attrs should we send to optix?
			unsigned int id = v.getBufferID(0);
			// clear OpenGL Errors so far; otherwise Optix won't work
			int e = glGetError();

			optix::Buffer buffers[VertexData::MaxAttribs];
			try {
				for (unsigned int b = 0; b < VertexData::MaxAttribs; ++b) { 
					if (v.getBufferID(b)) {
						buffers[b] = o_Context->createBufferFromGLBO(RT_BUFFER_INPUT,v.getBufferID(b));
						buffers[b]->setFormat(RT_FORMAT_FLOAT4);
						buffers[b]->setSize(v.getDataOf(0).size());
					}
				}
			}
			catch ( optix::Exception& e ) {
				NAU_THROW("Optix Error: Adding scene %s to pass %s (creating buffers from VBOs) [%s]",
										sc->getName().c_str(), m_Name.c_str(),e.getErrorString().c_str()); 
			}

			std::vector<IMaterialGroup *> mg = r.getMaterialGroups();
			for (unsigned int g = 0; g < mg.size(); ++g) {

				if (mg[g]->getNumberOfPrimitives() > 0) {
					try {
						optix::Geometry geom = o_Context->createGeometry();
						geom->setPrimitiveCount(mg[g]->getNumberOfPrimitives());
						geom->setBoundingBoxProgram(o_BoundingBoxProgram);
						geom->setIntersectionProgram(o_GeometryIntersectionProgram);

						geom["vertex_buffer"]->setBuffer(buffers[0]);
						for (unsigned int b = 1; b < VertexData::MaxAttribs; ++b) {
							if (v.getBufferID(b))
								geom[VertexData::Syntax[b]]->setBuffer(buffers[b]);
						}
						id = mg[g]->getIndexData().getBufferID();
						optix::Buffer indices = o_Context->createBufferFromGLBO(RT_BUFFER_INPUT, id);
						indices->setFormat(RT_FORMAT_UNSIGNED_INT);
						indices->setSize(mg[g]->getIndexData().getIndexSize());

						geom["index_buffer"]->setBuffer(indices);
						o_Material = o_Context->createMaterial();
						o_Material->setClosestHitProgram(0, o_ClosestHitProgram);
						
						o_Material["diffuse"]->set4fv(MATERIALLIBMANAGER->getMaterial(this->m_MaterialMap[mg[g]->getMaterialName()])->getColor().getDiffuse());

						o_GeomInstances.push_back(o_Context->createGeometryInstance());
						o_GeomInstances[o_GeomInstances.size()-1]->setMaterialCount(1);
						o_GeomInstances[o_GeomInstances.size()-1]->setMaterial(0, o_Material);
						o_GeomInstances[o_GeomInstances.size()-1]->setGeometry(geom);
					}
					catch ( optix::Exception& e ) {
						NAU_THROW("Optix Error: Adding scene %s to pass %s (adding material groups) [%s]",
												sc->getName().c_str(), m_Name.c_str(),e.getErrorString().c_str()); 
					}
				}
			}
		}
		try {
			o_GeomGroup->setChildCount(o_GeomInstances.size());
			for (unsigned int i = 0; i < o_GeomInstances.size(); ++i) {
				o_GeomGroup->setChild(i,o_GeomInstances[i]);
			}

			o_GeomGroup->setAcceleration(o_Context->createAcceleration("Bvh","Bvh"));
		}
		catch ( optix::Exception& e ) {
			NAU_THROW("Optix Error: Adding scene %s to pass %s (adding instances to geometry group) [%s]",
												sc->getName().c_str(), m_Name.c_str(),e.getErrorString().c_str()); 
		}

		//o_Context["geometry"]->set(o_GeomGroup);






*/		

	}
}

RTformat
PassOptix::getOptixFormat(Texture *t) {

	int nComp = t->getPropi(Texture::COMPONENT_COUNT);
	int type = t->getPrope(Texture::TYPE);

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
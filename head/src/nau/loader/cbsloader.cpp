#include <nau/loader/cbsloader.h>

#include <nau.h>

#include <nau/scene/isceneobject.h>
#include <nau/scene/sceneobjectfactory.h>
#include <nau/geometry/iboundingvolume.h>
#include <nau/geometry/boundingvolumefactory.h>
#include <nau/math/vec3.h>
#include <nau/math/mat4.h>
#include <nau/math/transformfactory.h>
#include <nau/render/vertexdata.h>
#include <nau/render/irenderable.h>
#include <nau/render/renderablefactory.h>
#include <nau/material/imaterialgroup.h>
#include <nau/material/materialgroup.h>
#include <nau/clogger.h>
#include <nau/material/material.h>
#include <nau/system/fileutil.h>

#include <assert.h>
#include <fstream>
#include <map>

using namespace nau::loader;
using namespace nau::scene;
using namespace nau::math;
using namespace nau::geometry;
using namespace nau::render;
using namespace nau::material;

void
_writeVertexData (VertexData& aVertexData, std::fstream &f) 
{
	size_t size;

	for (int i = VertexData::VERTEX_ARRAY; i <= VertexData::CUSTOM_ATTRIBUTE_ARRAY7; i++) {
		std::vector<vec3> &aVec = aVertexData.getDataOf ((VertexData::VertexDataType)i);
		
		size = aVec.size();
		f.write (reinterpret_cast<char *> (&size), sizeof (size));
		if (aVec.size() > 0) {
			f.write (reinterpret_cast<char *> (&(aVec[0])), 
					 static_cast<std::streamsize>(size) * sizeof(vec3));
		}
	}

	std::vector<unsigned int> &aVec = aVertexData.getIndexData();
	size = aVec.size();
	f.write (reinterpret_cast<char *> (&size), sizeof (size));

	if (aVec.size() > 0) {
		f.write (reinterpret_cast<char *> (&(aVec[0])), 
				 static_cast<std::streamsize> (size) * sizeof(unsigned int));
	}
}

void
_readVertexData (VertexData& aVertexData, std::fstream &f)
{
	unsigned int size;

	for (int i = VertexData::VERTEX_ARRAY; i <= VertexData::CUSTOM_ATTRIBUTE_ARRAY7; i++) {
		f.read (reinterpret_cast<char *> (&size), sizeof (size));

		if (size > 0) {
			std::vector<vec3> *aNewVector = new std::vector<vec3>(size);

			f.read (reinterpret_cast<char *> (&(*aNewVector)[0]), size * sizeof (vec3));
			aVertexData.setDataFor ((VertexData::VertexDataType)i, aNewVector);
		}
	}
	
	f.read (reinterpret_cast<char *> (&size), sizeof (size));
	if (size > 0) {
		std::vector<unsigned int> *aNewVector = new std::vector<unsigned int>(size);

		f.read (reinterpret_cast<char *> (&(*aNewVector)[0]), size * sizeof (unsigned int));

		aVertexData.setIndexData (aNewVector);
	}
}

void
_ignoreVertexData (std::fstream &f)
{
	unsigned int size;

	for (int i = VertexData::VERTEX_ARRAY; i <= VertexData::CUSTOM_ATTRIBUTE_ARRAY7; i++) {
		f.read (reinterpret_cast<char *> (&size), sizeof (size));

		if (size > 0) {
			f.ignore (size * sizeof (vec3));
		}
	}
	
	f.read (reinterpret_cast<char *> (&size), sizeof (size));
	if (size > 0) {
		f.ignore (size * sizeof (unsigned int));
	}

}

void
_writeString (const std::string& aString, std::fstream &f)
{
	unsigned int size = aString.size();

	f.write (reinterpret_cast<char *> (&size), sizeof (size));
	f.write (aString.c_str(), size + 1);
}
/*
void
_writeString (const std::string& aString, std::fstream &f)
{
	size_t size = aString.size();

	f.write (reinterpret_cast<char *> (&size), sizeof (size));
	f.write (aString.c_str(), 
		static_cast<std::streamsize>(size) + 1);
}
*/
void
_readString ( char *buffer, std::fstream &f)
{
	unsigned int size;

	memset (buffer, 0, 1024);
	f.read (reinterpret_cast<char *> (&size), sizeof (size));
	f.read (buffer, size + 1);
}

void
_ignoreString (std::fstream &f)
{
	unsigned int size;
	f.read (reinterpret_cast<char *> (&size), sizeof (size));
	f.ignore (size + 1);
}

void 
CBSLoader::loadScene (nau::scene::IScene *aScene, std::string &aFilename)
{
	std::string path = FileUtil::GetPath(aFilename);

	std::fstream f (aFilename.c_str(), std::fstream::in | std::fstream::binary);

	std::map<std::string, IRenderable*> renderables; /***MARK***/ //PROTO Renderables Manager
	//std::map<std::pair<std::string, std::string>, int> materialTrack;

	if (!f.is_open()) {
		LOG_ERROR ("Cannot open file: %s", aFilename.c_str());
		return;
	}

	char buffer[1024];
	unsigned int nObjects;
	unsigned int nMatGroups;

	_readString (buffer, f);

	LOG_INFO ("[Reading] Scene type: [%s]", buffer);

	f.read (reinterpret_cast<char *> (&nObjects), sizeof(nObjects));

	LOG_INFO ("[Reading] Number of Objects: [%d]", nObjects);

	for (unsigned int i = 0; i < nObjects; i++) {
		_readString (buffer, f);

	//	std::map<std::string, IRenderable*> renderables; /***MARK***/ //PROTO Renderables Manager
		LOG_INFO ("[Reading] Type of object: [%s]", buffer);

		ISceneObject *aObject = SceneObjectFactory::create (buffer);

		_readString (buffer, f);

		aObject->setName (buffer);

		aObject->readSpecificData (f);

		_readString (buffer, f);

		LOG_INFO ("[Reading] Type of BoundingVolume: [%s]", buffer);

		IBoundingVolume *aBoundingVolume = BoundingVolumeFactory::create (buffer);

		aObject->setBoundingVolume (aBoundingVolume);

		std::vector<vec3>& points = aBoundingVolume->getPoints();

		unsigned int nPoints;

		f.read (reinterpret_cast<char *> (&nPoints), sizeof (nPoints));

		LOG_INFO ("[Reading] Number of points [%d]", nPoints);

		f.read (reinterpret_cast<char *> (&(points[0])), nPoints * sizeof (vec3));

		_readString (buffer, f);

		LOG_INFO ("[Reading] Type of Transform: [%s]", buffer);

		ITransform *aTransform = TransformFactory::create (buffer);

		mat4 *mat = new mat4;


		f.read (reinterpret_cast<char *> (const_cast<float*>(mat->getMatrix())), sizeof(mat4));

		for (int i = 0; i < 16; i++) {
			LOG_INFO ("Matrix(%d): [%f]", i, mat->getMatrix()[i]);
		}

		aTransform->setMat44 (mat);

		aObject->setTransform (aTransform);

		_readString (buffer, f);

		LOG_INFO ("[Reading] Renderable's name: [%s]", buffer);

		if (!strcmp(buffer,"polySurface1507"))
			int x = 1;

		std::string renderableName(buffer);

		if (0 == renderableName.compare ("NULLOBJECT")) {
			continue;
		}

		IRenderable *aRenderable = 0;

		if (0 == renderables.count (renderableName)) {
			/*Create the new renderable */

			_readString (buffer, f);
			aRenderable = RenderableFactory::create (buffer);

			aRenderable->setName (renderableName);

			//assert (0 == aRenderable);

			renderables[renderableName] = aRenderable;
			
			VertexData &vertexData = aRenderable->getVertexData();

			_readVertexData (vertexData, f);

			LOG_INFO ("[Reading] Renderable type: [%s]", buffer);

			f.read (reinterpret_cast<char *> (&nMatGroups), sizeof (nMatGroups));

			for (unsigned int i = 0; i < nMatGroups; i++) {
				MaterialGroup *aMatGroup = new MaterialGroup;

				aMatGroup->setParent (aRenderable);

				_readString (buffer, f);

				LOG_INFO ("[Reading] Material Groups name: [%s]", buffer);

				aMatGroup->setMaterialName (buffer);				

				VertexData &indexData = aMatGroup->getVertexData();

				_readVertexData (indexData, f);
				
				_readString (buffer, f);

				LOG_INFO ("[Reading] Material name: [%s]", buffer);

//				int pos = aScene->findMaterialByName (buffer);

				bool matExists = MATERIALLIBMANAGER->hasMaterial (DEFAULTMATERIALLIBNAME, buffer);

//				int pos = 0;
				if (false == matExists) {
					//materialTrack[buffer] = 1;

					Material* aMaterial = new Material;

					aMaterial->setName (buffer);

					float values[4];
					float value;

					f.read (reinterpret_cast<char*> (values), sizeof (float) * 4);
					aMaterial->getColor().setAmbient (values);

					f.read (reinterpret_cast<char*> (values), sizeof (float) * 4);
					aMaterial->getColor().setSpecular (values);

					f.read (reinterpret_cast<char*> (values), sizeof (float) * 4);
					aMaterial->getColor().setDiffuse (values);

					f.read (reinterpret_cast<char*> (values), sizeof (float) * 4);
					aMaterial->getColor().setEmissive (values);

					f.read (reinterpret_cast<char*> (&value), sizeof (float));
					aMaterial->getColor().setShininess (value);

					//aMaterial->getColor().setAmbient (0.8f, 0.8f, 0.8f, 0.8f);

					for (int i = 0; i < 8; i++) { /***MARK***/ //8!? Is it a magic number!?
						_readString (buffer, f);
						if (0 != std::string(buffer).compare("<no texture>")) {
							aMaterial->createTexture (i, FileUtil::GetFullPath(path,buffer));
						}
					}

					f.read (reinterpret_cast<char*> (&aMaterial->m_Transparent), sizeof (int));
					f.read (reinterpret_cast<char*> (&aMaterial->m_Priority), sizeof (int));
					f.read (reinterpret_cast<char*> (&aMaterial->m_InOrder), sizeof (int));
					f.read (reinterpret_cast<char*> (&aMaterial->m_ShaderID), sizeof (int));
					f.read (reinterpret_cast<char*> (&aMaterial->OGLstate), sizeof (aMaterial->OGLstate));

					
					MATERIALLIBMANAGER->addMaterial (DEFAULTMATERIALLIBNAME, aMaterial);
				}
				// else {
				//	if (0 == materialTrack[buffer]) {
				//		materialTrack[buffer] = 1;
				//	}
				//		f.ignore (sizeof (float) * 4 * 4);
				//		f.ignore (sizeof (float));

				//		for (int i = 0; i < 8; i++)  /***MARK***/ //8!? Is it a magic number!?
				//			_readString (buffer, f);
				//		

				//		f.ignore (sizeof (int) * 4);
				//		f.ignore (sizeof (nau::render::GlState));


				//	

				//} 
				
				

				//assert (pos != -1);
				//aMatGroup->setMaterialId (pos);

				//aRenderable->addMaterialGroup (aMatGroup);


				/***MARK***/ //Another ugly hack to prevent memory copy
				aRenderable->getMaterialGroups().push_back (aMatGroup);
			}

		} else {
			/*Skip*/

			//_ignoreString (f);
		
			//_ignoreVertexData (f);

			//f.read (reinterpret_cast<char *> (&nMatGroups), sizeof (nMatGroups));

			//for (int i = 0; i < nMatGroups; i++) {
			//	_readString (buffer, f);

			//	_ignoreVertexData (f);
			//	
			//	_ignoreString (f);

			//	f.ignore (sizeof (float) * 4 * 4);
			//	f.ignore (sizeof (float));

			//	for (int i = 0; i < 8; i++) { /***MARK***/ //8!? Is it a magic number!?
			//		_readString (buffer, f);
			//	}

			//	f.ignore (sizeof (int) * 4);
			//	f.ignore (sizeof (COGLState));

			//}

			/*Shared geometry*/
			aRenderable = renderables[renderableName];
		}

		//assert (0 == aRenderable);
		aObject->setRenderable (aRenderable);

		aScene->add (aObject);
	}
}

void 
CBSLoader::writeScene (nau::scene::IScene *aScene, std::string &aFilename)
{

	std::string path = FileUtil::GetPath(aFilename);
		
	std::map<std::string, IRenderable*> renderables;

	std::map<std::string, int> materials;

	//std::fstream f (aFilename.c_str(), std::fstream::out);
	std::fstream f (aFilename.c_str(), std::fstream::out | std::fstream::binary);


	if (!f.is_open()) {
		LOG_ERROR ("Cannot open file: %s", aFilename.c_str());
		return;
	}

	unsigned int size;
	//size_t size; ARF

	_writeString (aScene->getType(), f);
	LOG_INFO ("[Writing] scene type: %s", aScene->getType().c_str()); 

	std::vector<ISceneObject*> &sceneObjects = aScene->getAllObjects();
	std::vector<ISceneObject*>::iterator objIter;

	/* Number of objects */

	size = sceneObjects.size();
	f.write (reinterpret_cast<char *> (&size), sizeof(size));

	objIter = sceneObjects.begin();

	/* For each object in the scene */
	for ( ; objIter != sceneObjects.end(); objIter++) {
		/* Write the object type */


		_writeString ((*objIter)->getType(), f);

		LOG_INFO ("[Writing] object type: [%s]", (*objIter)->getType().c_str()); 


		/* Write the object's name */
		_writeString ((*objIter)->getName(), f); //Misses getId()

		LOG_INFO ("[Writing] object name: [%s]", (*objIter)->getName().c_str()); 

		/* Write the specific data */		
		(*objIter)->writeSpecificData (f);
		
		/* Write the bounding volume */
		const IBoundingVolume *aBoundingVolume = (*objIter)->getBoundingVolume();

		/* Bounding volume type */

		_writeString (aBoundingVolume->getType(), f);

		/* Bounding volume points */
		std::vector<vec3> &points = aBoundingVolume->getPoints();

		size = points.size();
		f.write (reinterpret_cast<char*> (&size), sizeof(size));

		f.write(reinterpret_cast<char*> (&(points[0])), 
			static_cast<std::streamsize> (size) * sizeof (vec3));

		/* Write the transform */
		const ITransform &aTransform = (*objIter)->getTransform();
		
		/* Transform type */
		_writeString (aTransform.getType(), f);

		/* The transform's matrix */
		f.write(reinterpret_cast<char*> (const_cast<float *>(aTransform.getMat44().getMatrix())), sizeof(mat4));

		IRenderable *aRenderablePtr = (*objIter)->_getRenderablePtr();

		if (0 == aRenderablePtr) {
			_writeString ("NULLOBJECT", f);
			continue;
		}

		/* Write the object's renderable */
		IRenderable &aRenderable = (*objIter)->getRenderable();


		/* The renderable name, for later lookup */
		_writeString (aRenderable.getName(), f);

		LOG_INFO ("[Writing] Renderable's name: [%s]", aRenderable.getName().c_str()); 

		if (0 == renderables.count (aRenderable.getName())) {

			renderables[aRenderable.getName()] = &aRenderable;

			_writeString (aRenderable.getType(), f);

			LOG_INFO ("[Writing] Renderable's type: [%s]", aRenderable.getType().c_str()); 

			/* Vertices data */
			VertexData &aVertexData = aRenderable.getVertexData();

			_writeVertexData (aVertexData, f);

			/* Material groups */
			std::vector<nau::material::IMaterialGroup*>& materialGroups = aRenderable.getMaterialGroups();
			std::vector<nau::material::IMaterialGroup*>::iterator mgIter;

			size = materialGroups.size();

			f.write (reinterpret_cast<char*> (&size), sizeof (size));

			mgIter = materialGroups.begin();

			for ( ; mgIter != materialGroups.end(); mgIter++) {
				IMaterialGroup *aMaterialGroup = (*mgIter);
				
				/*Write material's name */

				_writeString (aMaterialGroup->getMaterialName(), f);

				LOG_INFO ("[Writing] MaterialGroup's name: %s", aMaterialGroup->getMaterialName().c_str()); 
							
				/* Indices Data */
				VertexData &mgVertexData = aMaterialGroup->getVertexData();

				_writeVertexData (mgVertexData, f);

				/* Material */

				//Material *aMaterial = aScene->getMaterial (aMaterialGroup->getMaterialId()); /***MARK***/ //This should come from the material manager
				
				Material *aMaterial = MATERIALLIBMANAGER->getDefaultMaterial (aMaterialGroup->getMaterialName()); 

				_writeString (aMaterial->getName(), f);

				LOG_INFO ("[Writing] Material's name: %s", aMaterial->getName().c_str()); 

				if (0 == materials[aMaterial->getName()]) {
					materials[aMaterial->getName()] = 1;

					f.write (reinterpret_cast<char*> (const_cast<float*>(aMaterial->getColor().getAmbient())), sizeof (float) * 4);
					f.write (reinterpret_cast<char*> (const_cast<float*>(aMaterial->getColor().getSpecular())), sizeof (float) * 4);
					f.write (reinterpret_cast<char*> (const_cast<float*>(aMaterial->getColor().getDiffuse())), sizeof (float) * 4);
					f.write (reinterpret_cast<char*> (const_cast<float*>(aMaterial->getColor().getEmissive())), sizeof (float) * 4);

					float value = aMaterial->getColor().getShininess();
					f.write (reinterpret_cast<char*> (&value), sizeof (float));

					for (int i = 0; i < 8; i++) { /***MARK***/ //8!? Is it a magic number!?
						if (0 != aMaterial->getTextures() && 0 != aMaterial->getTextures()->getTexture(i)) {
							std::string label = aMaterial->getTextures()->getTexture(i)->getLabel();
							_writeString (FileUtil::GetRelativePathTo(path,label), f);
						} else {
							_writeString ("<no texture>", f);
						}
					}

					f.write (reinterpret_cast<char*> (&aMaterial->m_Transparent), sizeof (int));
					f.write (reinterpret_cast<char*> (&aMaterial->m_Priority), sizeof (int));
					f.write (reinterpret_cast<char*> (&aMaterial->m_InOrder), sizeof (int));
					f.write (reinterpret_cast<char*> (&aMaterial->m_ShaderID), sizeof (int));
					f.write (reinterpret_cast<char*> (&aMaterial->OGLstate), sizeof (aMaterial->OGLstate));
				}
			}
		}
	}

	f.close();
}

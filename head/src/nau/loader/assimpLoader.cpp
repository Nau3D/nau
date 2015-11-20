#include "nau/loader/assimpLoader.h"

#include "nau.h"
#include "nau/config.h"
#include "nau/slogger.h"
#include "nau/geometry/mesh.h"
#include "nau/material/material.h"
#include "nau/material/materialGroup.h"
#include "nau/render/iRenderable.h"
#include "nau/geometry/vertexData.h"
#include "nau/system/file.h"

#include <fstream>

using namespace nau::loader;
using namespace nau::geometry;
using namespace nau::system;

Assimp::Importer AssimpLoader::importer;

void
AssimpLoader::loadScene(nau::scene::IScene *aScene, std::string &aFilename, std::string &params) {

	std::string path = File::GetPath(aFilename);
	const aiScene *sc;
	//check if file exists
	std::ifstream fin(aFilename.c_str());
	if(!fin.fail()) {
		fin.close();
	}

	else{
		SLOG ("Cannot open file: %s", aFilename.c_str());
		return;
	}

	sc = importer.ReadFile( aFilename, aiProcessPreset_TargetRealtime_Quality);//aiProcess_CalcTangentSpace|aiProcess_Triangulate);//aiProcessPreset_TargetRealtime_Quality);

	// If the import failed, report it
	if( !sc)
	{
		SLOG ("Error reading file: %s\n%s\n", aFilename.c_str(),importer.GetErrorString());
		return;
	}

	unsigned int order = XYZ;
	if (params.find("SWAP_YZ") != std::string::npos)
		order = XZ_Y;

	unsigned int primitive;
	if (params.find("USE_ADJACENCY") != std::string::npos) 
		primitive = IRenderable::TRIANGLES_ADJACENCY;
	else 
		primitive = IRenderable::TRIANGLES;

	std::map<unsigned int, std::string> meshNameMap;

		// For each mesh
	for (unsigned int n = 0; n < sc->mNumMeshes; ++n)
	{
		const  aiMesh* mesh = sc->mMeshes[n];

		if (mesh->mPrimitiveTypes != 4)
			continue;

		Mesh *renderable =  (Mesh *)RESOURCEMANAGER->createRenderable("Mesh", mesh->mName.data, aFilename);
		meshNameMap[n] = renderable->getName();
		renderable->setDrawingPrimitive(primitive);

		std::shared_ptr<std::vector<unsigned int>> indices = 
			std::shared_ptr<std::vector<unsigned int>>(new std::vector<unsigned int>);

		for (unsigned int t = 0; t < mesh->mNumFaces; ++t) {
			const  aiFace* face = &mesh->mFaces[t];
			indices->push_back(face->mIndices[0]);
			indices->push_back(face->mIndices[1]);
			indices->push_back(face->mIndices[2]);
		}

		aiMaterial *mtl = sc->mMaterials[mesh->mMaterialIndex];
		aiString name;
		mtl->Get(AI_MATKEY_NAME,name);
		std::string matName = name.data;
		if (matName == "")
			matName = "Default";
		std::shared_ptr<MaterialGroup> aMaterialGroup = MaterialGroup::Create(renderable, matName);
		aMaterialGroup->setIndexList(indices);
		if (primitive == IRenderable::TRIANGLES_ADJACENCY)
			aMaterialGroup->getIndexData()->useAdjacency(true);

		renderable->addMaterialGroup(aMaterialGroup);
		
		VertexData &vertexData = renderable->getVertexData();

		if (mesh->HasPositions()) {
			std::vector<VertexData::Attr>* vertex = readGL3FArray((float *)mesh->mVertices,mesh->mNumVertices, order, 1.0f);
			vertexData.setDataFor(VertexData::GetAttribIndex(std::string("position")), vertex);
		}

		if (mesh->HasNormals()) {
			std::vector<VertexData::Attr>* normal = readGL3FArray((float *)mesh->mNormals,mesh->mNumVertices, order, 0.0f);
			vertexData.setDataFor(VertexData::GetAttribIndex(std::string("normal")), normal);
		}

		// buffer for vertex texture coordinates
		if (mesh->HasTextureCoords(0)) {
			std::vector<VertexData::Attr>* texCoord = new std::vector<VertexData::Attr>(mesh->mNumVertices);
			for (unsigned int k = 0; k < mesh->mNumVertices; ++k) {

				texCoord->at(k).x   = mesh->mTextureCoords[0][k].x;
				texCoord->at(k).y   = mesh->mTextureCoords[0][k].y; 
				texCoord->at(k).z  = mesh->mTextureCoords[0][k].z; 
				//texCoord->at(k).z = 0.0;
				texCoord->at(k).w = 1.0;
			}
			vertexData.setDataFor(VertexData::GetAttribIndex(std::string("texCoord0")), texCoord);
		}

		if (!MATERIALLIBMANAGER->hasMaterial (DEFAULTMATERIALLIBNAME, matName)) {

			Material *m = MATERIALLIBMANAGER->createMaterial(matName);
			aiString texPath;	
			if(AI_SUCCESS == mtl->GetTexture(aiTextureType_DIFFUSE, 0, &texPath)){
				m->createTexture(0,File::GetFullPath(path,texPath.data));
			}
			if (AI_SUCCESS == mtl->GetTexture(aiTextureType_HEIGHT, 0, &texPath)){
				m->createTexture(1, File::GetFullPath(path, texPath.data));
			}
			if (AI_SUCCESS == mtl->GetTexture(aiTextureType_NORMALS, 0, &texPath)){
				m->createTexture(2, File::GetFullPath(path, texPath.data));
			}

			aiColor4D color;
			ColorMaterial *cm = &(m->getColor());

			if(AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_DIFFUSE, &color)) {
				cm->setPropf4(ColorMaterial::DIFFUSE,color.r, color.g, color.b, color.a);
			}

			if(AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_AMBIENT, &color)) {
				cm->setPropf4(ColorMaterial::AMBIENT, color.r, color.g, color.b, color.a);
			}

			if(AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_SPECULAR, &color)) {
				cm->setPropf4(ColorMaterial::SPECULAR, color.r, color.g, color.b, color.a);
			}

			if(AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_EMISSIVE, &color)) {
				cm->setPropf4(ColorMaterial::EMISSION, color.r, color.g, color.b, color.a);
			}
			float shininess = 0.0;
			unsigned int max;
			if (AI_SUCCESS == aiGetMaterialFloatArray(mtl, AI_MATKEY_SHININESS, &shininess, &max))
				cm->setPropf(ColorMaterial::SHININESS, shininess);
		}
	}
	mat4 m;
	m.setIdentity();
	recursiveWalk(aScene, aFilename, sc, sc->mRootNode, m, meshNameMap);
}


void
AssimpLoader::writeScene(nau::scene::IScene *aScene, std::string &aFilename) {

}


void 
AssimpLoader::recursiveWalk (nau::scene::IScene *aScene, std::string &aFilename,
									const  aiScene *sc, const  aiNode* nd, mat4 &m,
									std::map<unsigned int, std::string> meshNameMap) {
	mat4 original, aux, m4;
	original.copy(m);
	
	aiMatrix4x4 mA = nd->mTransformation;
	mA.Transpose();
	float f[16];
	memcpy(f,&mA, sizeof(float)*16);

	m4.setMatrix(f);
	aux.copy(m4);
	m *= aux;

	for (unsigned int n=0; n < nd->mNumMeshes; ++n) {
	
		if (sc->mMeshes[nd->mMeshes[n]]->mPrimitiveTypes == 4) {
		//sc->mMeshes[nd->mMeshes[n]]->mName.data;
		SceneObject *so = SceneObjectFactory::Create("SimpleObject");
		so->setRenderable(RESOURCEMANAGER->getRenderable(meshNameMap[nd->mMeshes[n]],""));
		so->setTransform(m);
		aScene->add(so);
		}
	}

	for (unsigned int n=0; n < nd->mNumChildren; ++n){
		recursiveWalk(aScene, aFilename, sc, nd->mChildren[n], m, meshNameMap);
	}

	m.copy(original);
}


std::vector<VertexData::Attr>*
AssimpLoader::readGL3FArray(float* a, unsigned int arraysize, unsigned int order, float w) {

	std::vector<VertexData::Attr> *v = new std::vector<VertexData::Attr>(arraysize);

	for(unsigned int i=0;i<arraysize;i++)
		if (order == XYZ)
			(*v)[i] = VertexData::Attr((float)a[(i)*3], (float)a[((i)*3)+1], (float)a[((i)*3)+2], w);
		else 
			(*v)[i] = VertexData::Attr((float)a[(i)*3], -(float)a[((i)*3)+2], (float)a[((i)*3)+1], w);

	return v;
}


std::vector<VertexData::Attr>*
AssimpLoader::readGL2FArray(float* a, unsigned int arraysize) {

	std::vector<VertexData::Attr> *v = new std::vector<VertexData::Attr>(arraysize);

	for(unsigned int i=0;i<arraysize;i++)
		(*v)[i] = VertexData::Attr((float)a[(i)*2], (float)a[((i)*2)+1], 0.0f, 1.0f);

	return v;
}


void 
AssimpLoader::color4_to_float4(const  aiColor4D *c, float f[4]) {

	f[0] = c->r;
	f[1] = c->g;
	f[2] = c->b;
	f[3] = c->a;
}


void 
AssimpLoader::set_float4(float f[4], float a, float b, float c, float d) {

	f[0] = a;
	f[1] = b;
	f[2] = c;
	f[3] = d;
}

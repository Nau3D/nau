#include "nau/loader/objLoader.h"

#include "nau.h"
#include "nau/config.h"
#include "nau/slogger.h"
#include "nau/clogger.h"
#include "nau/debug/profile.h"
#include "nau/geometry/iBoundingVolume.h"
#include "nau/geometry/boundingvolumefactory.h"
#include "nau/geometry/vertexData.h"
#include "nau/material/material.h"
#include "nau/material/materialGroup.h"
#include "nau/math/matrix.h"
#include "nau/math/vec4.h"
#include "nau/render/iRenderable.h"
#include "nau/scene/sceneObject.h"
#include "nau/scene/sceneObjectFactory.h"
#include "nau/system/file.h"

// Assert and other basics
#include <assert.h>
//#include <fstream>
#include <math.h>
#include <map>
#include <cstring>
#include <set>
#include <vector>

#ifdef WIN32
#define PATH_SEPARATOR "\\"
#define PATH_SEPARATOR_C '\\'
#define strdup _strdup
#else
#define PATH_SEPARATOR "/"
#define PATH_SEPARATOR_C '/'
#endif

using namespace nau::system;
using namespace nau::loader;
using namespace nau::scene;
using namespace nau::math;
using namespace nau::geometry;
using namespace nau::render;
using namespace nau::material;


#ifndef M_PI
#define M_PI 3.14159265
#endif

#undef SINGLE_STRING_GROUP_NAMES

float
OBJLoader::dot(float* u, float* v) {

	assert(u); assert(v);
	
	return u[0]*v[0] + u[1]*v[1] + u[2]*v[2];
}


void
OBJLoader::cross(float* u, float* v, float* res) {

	assert(u); assert(v); assert(res);
	
	res[0] = u[1]*v[2] - u[2]*v[1];
	res[1] = u[2]*v[0] - u[0]*v[2];
	res[2] = u[0]*v[1] - u[1]*v[0];
}


void
OBJLoader::normalize(float* v) {

	float l;
	
	assert(v);
	
	l = (float)sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
	v[0] /= l;
	v[1] /= l;
	v[2] /= l;
}


void 
OBJLoader::initGroup(std::string &s) {

	Group group;

	if (m_Groups.count(s) == 0) {
		group.name = s;
		group.material = "DefaultOBJMaterial";
		group.numTriangles = 0;
		m_Groups[s] = group;
		m_NumGroups++;
	}
}


/* readMTL: read a wavefront material library file
 *
 * name  - name of the material library
 */
void
OBJLoader::readMTL(std::string &name)
{
	FILE* file;
	char  buf[128];
	char buf2[1024];

	file = fopen(File::BuildFullFileName(m_Dir, name).c_str(), "r");
	if (!file) {
		NAU_THROW("Failed to open OBJ material file: %s", name.c_str());
	}
	
	m_NumMaterials = 0;
	std::shared_ptr<Material> mat;
	float val; vec4 v4;
	while(fscanf(file, "%s", buf) != EOF) {
		switch(buf[0]) {
		case '#':				/* comment */
			/* eat up rest of line */
			fgets(buf, sizeof(buf), file);
			break;
		case 'n':				/* newmtl */
			m_NumMaterials++;
			fgets(buf, sizeof(buf), file);
			sscanf(buf, "%s %s", buf, buf);
			mat = MATERIALLIBMANAGER->createMaterial(strdup(buf));
			break;
		case 'm' :		
			fgetc(file);
			fgets(buf2, sizeof(buf2), file);
			sscanf(buf2, "%s", buf2);
			if (buf[4] == 'K') { // map_K?
				switch (buf[5]) {
				case 'd':	// map_Kd
					mat->createTexture(0, File::BuildFullFileName(m_Dir, buf2));
					break;
				case 'a':	// map_Ka
					mat->createTexture(3, File::BuildFullFileName(m_Dir, buf2));
					break;
				case 's':	// map_Ks
					mat->createTexture(4, File::BuildFullFileName(m_Dir, buf2));
					break;
				}
			}
			else if (buf[4] = 'N') { // map_Ns
				mat->createTexture(5, File::BuildFullFileName(m_Dir, buf2));
			}
			else if (buf[4] == 'b') { // map_bump
				mat->createTexture(1, File::BuildFullFileName(m_Dir, buf2));
			}
			break;
		case 'b':
			fgetc(file);
			fgets(buf2, sizeof(buf2), file);
			sscanf(buf2, "%s", buf2);
			mat->createTexture(1, File::BuildFullFileName(m_Dir, buf2));
			break;
		case 'N':
			fscanf(file, "%f", &val);
			/* wavefront shininess is from [0, 1000], so scale for OpenGL */
			//model->materials[nummaterials].shininess = val * 128.0 / 1000.0;
			mat->getColor().setPropf(ColorMaterial::SHININESS, val * 128.0f / 1000.0f);
			break;
		case 'K':
			switch(buf[1]) {
			case 'd':
				fscanf(file, "%f %f %f", &v4.x, &v4.y, &v4.z);
				v4.w = 1.0;
				mat->getColor().setPropf4(ColorMaterial::DIFFUSE, v4);
				break;
			case 's':
				fscanf(file, "%f %f %f", &v4.x, &v4.y, &v4.z);
				v4.w = 1.0;
				mat->getColor().setPropf4(ColorMaterial::SPECULAR, v4);
				break;
			case 'a':
				fscanf(file, "%f %f %f", &v4.x, &v4.y, &v4.z);
				v4.w = 1.0;
				mat->getColor().setPropf4(ColorMaterial::AMBIENT, v4);
				break;
			default:
				/* eat up rest of line */
				fgets(buf, sizeof(buf), file);
				break;
			}
			break;
		default:
			/* eat up rest of line */
			fgets(buf, sizeof(buf), file);
			break;
		}
	}
	fclose(file);
}



/* glmFirstPass: first pass at a Wavefront OBJ file that gets all the
 * statistics of the model (such as #vertices, #normals, etc)
 *
 * model - properly initialized GLMmodel structure
 * file  - (fopen'd) file descriptor 
 */

void
OBJLoader::firstPass(FILE* file) 
{
	unsigned int v, n, t;
	char      buf[128];
	std::string groupName, materialName;

	Group *group;
	/* make a default group */
	std::string s = "default";
	initGroup(s);
	group = &m_Groups["default"];
	groupName = "default";
	materialName = "DefaultOBJMaterial";
	
	m_NumVertices = m_NumNormals = m_NumTexCoords = m_NumTriangles = 0;
	m_NumMaterials = 0;
	while(fscanf(file, "%s", buf) != EOF) {

		switch (buf[0]) {
		case '#':				/* comment */
			/* eat up rest of line */
			fgets(buf, sizeof(buf), file);
			break;
		case 'v':				/* v, vn, vt */
			switch (buf[1]) {
			case '\0':			/* vertex */
				/* eat up rest of line */
				fgets(buf, sizeof(buf), file);
				m_NumVertices++;
				break;
			case 'n':				/* normal */
				/* eat up rest of line */
				fgets(buf, sizeof(buf), file);
				m_NumNormals++;
				break;
			case 't':				/* texcoord */
				/* eat up rest of line */
				fgets(buf, sizeof(buf), file);
				m_NumTexCoords++;
				break;
			default:
				printf("glmFirstPass(): Unknown token \"%s\".\n", buf);
				exit(1);
				break;
			}
			break;

		case 'm':
			fgets(buf, sizeof(buf), file);
			sscanf(buf, "%s %s", buf, buf);
			s = buf;
			m_MtlLibName = s;
			readMTL(s);
			break;
		case 'u':
			/* eat up rest of line */
			fgets(buf, sizeof(buf), file);
			////
			sscanf(buf, "%s %s", buf, buf);
			s = groupName + "-" + buf;
			initGroup(s);
			group = &(m_Groups[s]);
			group->material = buf;
			materialName = buf;
			break;
		case 'g':				/* group */
			/* eat up rest of line */
			fgets(buf, sizeof(buf), file);
#ifdef SINGLE_STRING_GROUP_NAMES
			sscanf(buf, "%s", buf);
#else
			buf[strlen(buf) - 1] = '\0';	/* nuke '\n' */
#endif
			s = std::string(buf);
			initGroup(s);
			group = &(m_Groups[s]);
			group->material = materialName;
			groupName = s;
			break;

		case 'f':				/* face */
			v = n = t = 0;
			fscanf(file, "%s", buf);
			/* can be one of %d, %d//%d, %d/%d, %d/%d/%d %d//%d */
			if (strstr(buf, "//")) {
				/* v//n */
				sscanf(buf, "%d//%d", &v, &n);
				fscanf(file, "%d//%d", &v, &n);
				fscanf(file, "%d//%d", &v, &n);
				m_NumTriangles++;
				group->numTriangles++;
				while (fscanf(file, "%d//%d", &v, &n) > 0) {
					m_NumTriangles++;
					group->numTriangles++;
				}
			}
			else if (sscanf(buf, "%d/%d/%d", &v, &t, &n) == 3) {
				/* v/t/n */
				fscanf(file, "%d/%d/%d", &v, &t, &n);
				fscanf(file, "%d/%d/%d", &v, &t, &n);
				m_NumTriangles++;
				group->numTriangles++;
				while (fscanf(file, "%d/%d/%d", &v, &t, &n) > 0) {
					m_NumTriangles++;
					group->numTriangles++;
				}
			}
			else if (sscanf(buf, "%d/%d", &v, &t) == 2) {
				/* v/t */
				fscanf(file, "%d/%d", &v, &t);
				fscanf(file, "%d/%d", &v, &t);
				m_NumTriangles++;
				group->numTriangles++;
				while (fscanf(file, "%d/%d", &v, &t) > 0) {
					m_NumTriangles++;
					group->numTriangles++;
				}
			}
			else {
				/* v */
				fscanf(file, "%d", &v);
				fscanf(file, "%d", &v);
				m_NumTriangles++;
				group->numTriangles++;
				while (fscanf(file, "%d", &v) > 0) {
					m_NumTriangles++;
					group->numTriangles++;
				}
			}
			break;
		default:
			/* eat up rest of line */
			fgets(buf, sizeof(buf), file);
			break;
		}
	}

	/* allocate memory for the triangles in each group */
	for (auto g : m_Groups) {
		m_Groups[g.first].indices.resize(m_Groups[g.first].numTriangles);
		m_Groups[g.first].numTriangles = 0;
	}
	m_NumGroups = (unsigned int)m_Groups.size();

	m_Indices.resize(m_NumTriangles * 3);
}


/* glmSecondPass: second pass at a Wavefront OBJ file that gets all
 * the data.
 *
 * model - properly initialized GLMmodel structure
 * file  - (fopen'd) file descriptor 
 */

void 
OBJLoader::addIndex(unsigned int modelIndex, unsigned int v, unsigned int t, unsigned int n) {

	m_Indices[modelIndex].v = v;
	m_Indices[modelIndex].t = t;
	m_Indices[modelIndex].n = n;
}


void
OBJLoader::secondPass(FILE* file) 
{
	unsigned int    numvertices;		/* number of vertices in model */
	unsigned int    numnormals;			/* number of normals in model */
	unsigned int    numtexcoords;		/* number of texcoords in model */
	unsigned int    numtriangles;		/* number of triangles in model */
	Group* group;			/* current group pointer */
	std::string    material;			/* current material */
	unsigned int    v, n, t;
	char      buf[128];
	std::string s, groupName;

	group        = &m_Groups["default"];
	groupName = "default";
	//group->material = "DefaultOBJMaterial";

	/* on the second pass through the file, read all the data into the
     allocated arrays */
	numvertices = numnormals = numtexcoords = 1;
	numtriangles = 0;
	material = "DefaultOBJMaterial";
	while(fscanf(file, "%s", buf) != EOF) {
		switch(buf[0]) {
		case '#':				/* comment */
			/* eat up rest of line */
			fgets(buf, sizeof(buf), file);
			break;
		case 'v':				/* v, vn, vt */
			switch(buf[1]) {
			case '\0':			/* vertex */
				fscanf(file, "%f %f %f",
					&m_Vertices[numvertices].x,
					&m_Vertices[numvertices].y,
					&m_Vertices[numvertices].z);
				m_Vertices[numvertices].w = 1.0f; 
				numvertices++;
				break;
			case 'n':				/* normal */
				fscanf(file, "%f %f %f", 
					  &m_Normals[numnormals].x,
					  &m_Normals[numnormals].y,
					  &m_Normals[numnormals].z);
				m_Normals[numnormals].w = 0.0f;
				numnormals++;
				break;
			case 't':				/* texcoord */
				fscanf(file, "%f %f", 
			        &m_TexCoords[numtexcoords].x,
					&m_TexCoords[numtexcoords].y);
				m_TexCoords[numtexcoords].z = 0.0f;
				m_TexCoords[numtexcoords].w = 0.0f;
				numtexcoords++;
				break;
			}
			break;
		case 'u':
			fgets(buf, sizeof(buf), file);
			sscanf(buf, "%s %s", buf, buf);
			material = buf;
			s = groupName + "-" + material;
			group = &m_Groups[s];
			//group->material = buf;
			break;
		case 'g':				/* group */
			/* eat up rest of line */
			fgets(buf, sizeof(buf), file);
#ifdef SINGLE_STRING_GROUP_NAMES
			sscanf(buf, "%s", buf);
#else
			buf[strlen(buf)-1] = '\0';	/* nuke '\n' */
#endif
			s = std::string(buf);
			group = &m_Groups[s];
			groupName = s;
//			group = &m_Groups[buf];
//			group->material = material;
			break;
		case 'f':				/* face */
			v = n = t = 0;
			fscanf(file, "%s", buf);
			/* can be one of %d, %d//%d, %d/%d, %d/%d/%d %d//%d */
			if (strstr(buf, "//")) {
			/* v//n */
				sscanf(buf, "%d//%d", &v, &n);
				addIndex(numtriangles * 3, v,0,n);
				fscanf(file, "%d//%d", &v, &n);
				addIndex(numtriangles * 3+1, v,0,n);
				fscanf(file, "%d//%d", &v, &n);
				addIndex(numtriangles * 3+2, v,0,n);
				group->indices[group->numTriangles] = numtriangles;
				group->numTriangles++;
				numtriangles++;
				while(fscanf(file, "%d//%d", &v, &n) > 0) {
					addIndex(numtriangles * 3, 
						m_Indices[(numtriangles - 1) * 3].v,0,
						m_Indices[(numtriangles - 1) * 3].n);
					addIndex(numtriangles * 3 + 1, 
						m_Indices[(numtriangles - 1) * 3 + 2].v,0,
						m_Indices[(numtriangles - 1) * 3 + 2].n);
					addIndex(numtriangles * 3 + 2, v,0,n);
					group->indices[group->numTriangles] = numtriangles;
					group->numTriangles++;
					numtriangles++;
				}
			} 
			else if (sscanf(buf, "%d/%d/%d", &v, &t, &n) == 3) {
				/* v/t/n */
				addIndex(numtriangles * 3, v,t,n);
				fscanf(file, "%d/%d/%d", &v, &t, &n);
				addIndex(numtriangles * 3+1, v,t,n);
				fscanf(file, "%d/%d/%d", &v, &t, &n);
				addIndex(numtriangles * 3+2, v,t,n);
				group->indices[group->numTriangles] = numtriangles;
				group->numTriangles++;
				numtriangles++;
				while(fscanf(file, "%d/%d/%d", &v, &t, &n) > 0) {
					addIndex(numtriangles * 3, 
						m_Indices[(numtriangles - 1) * 3].v,
						m_Indices[(numtriangles - 1) * 3].t,
						m_Indices[(numtriangles - 1) * 3].n);
					addIndex(numtriangles * 3 + 1, 
						m_Indices[(numtriangles - 1) * 3 + 2].v,
						m_Indices[(numtriangles - 1) * 3 + 2].t,
						m_Indices[(numtriangles - 1) * 3 + 2].n);
					addIndex(numtriangles * 3 + 2, v,t,n);
					group->indices[group->numTriangles] = numtriangles;
					group->numTriangles++;
					numtriangles++;
			
				}
			} 
			else if (sscanf(buf, "%d/%d", &v, &t) == 2) {
				/* v/t */
				addIndex(numtriangles * 3,v,t,0);
				fscanf(file, "%d/%d", &v, &t);
				addIndex(numtriangles * 3+1, v,t,0);
				fscanf(file, "%d/%d", &v, &t);
				addIndex(numtriangles * 3+2, v,t,0);
				group->indices[group->numTriangles] = numtriangles;
				group->numTriangles++;
				numtriangles++;
				while(fscanf(file, "%d/%d", &v, &t) > 0) {
					addIndex(numtriangles * 3,
						m_Indices[(numtriangles - 1) * 3].v,
						m_Indices[(numtriangles - 1) * 3].t,0);
					addIndex(numtriangles * 3 + 1, 
						m_Indices[(numtriangles - 1) * 3 + 2].v,
						m_Indices[(numtriangles - 1) * 3 + 2].t,0);
					addIndex(numtriangles * 3 + 2, v,t,0);
					group->indices[group->numTriangles] = numtriangles;
					group->numTriangles++;
					numtriangles++;
			
				}
			} 
			else {
				/* v */
				addIndex(numtriangles * 3, v,0,0);
				fscanf(file, "%d", &v);
				addIndex(numtriangles * 3+1, v,0,0);
				fscanf(file, "%d", &v);
				addIndex(numtriangles * 3+2, v,0,0);
				group->indices[group->numTriangles] = numtriangles;
				group->numTriangles++;
				numtriangles++;
				while(fscanf(file, "%d", &v) > 0) {
					addIndex(numtriangles * 3, 
						m_Indices[(numtriangles - 1) * 3].v,0,0);
					addIndex(numtriangles * 3 + 1,
						m_Indices[(numtriangles - 1) * 3 + 2].v,0,0);
					addIndex(numtriangles * 3 + 2, v,0,0);
					group->indices[group->numTriangles] = numtriangles;
					group->numTriangles++;
					numtriangles++;
			
				}
			}
			break;

		default:
			/* eat up rest of line */
			fgets(buf, sizeof(buf), file);
			break;
		}
	}
}







/* glmFacetNormals: Generates facet normals for a model (by taking the
 * cross product of the two vectors derived from the sides of each
 * triangle).  Assumes a counter-clockwise winding.
 *
 * model - initialized GLMmodel structure
 */

void
OBJLoader::faceNormals()
{
	unsigned int  i;
	float u[3];
	float v[3];
  
	/* allocate memory for the new facet normals */
	m_NumFacetNorms = m_NumTriangles;
	m_FacetNorms.resize(m_NumFacetNorms + 1);
	
	for (i = 0; i < m_NumTriangles; i++) {
	
		u[0] = m_Vertices[m_Indices[i*3+1].v].x -
		       m_Vertices[m_Indices[i*3].v].x;
		u[1] = m_Vertices[m_Indices[i*3+1].v].y -
		       m_Vertices[m_Indices[i*3].v].y;
		u[2] = m_Vertices[m_Indices[i*3+1].v].z -
		       m_Vertices[m_Indices[i*3].v].z;
			   			 
		v[0] = m_Vertices[m_Indices[i*3+2].v].x -
		       m_Vertices[m_Indices[i*3].v].x;
		v[1] = m_Vertices[m_Indices[i*3+2].v].y -
		       m_Vertices[m_Indices[i*3].v].y;
		v[2] = m_Vertices[m_Indices[i*3+2].v].z -
		       m_Vertices[m_Indices[i*3].v].z;
			
		float res[3];
		cross(u, v, res);
		normalize(res);
		m_FacetNorms[(i + 1)].set(res);
	}
}



/* glmDelete: Deletes a GLMmodel structure.
 *
 * model - initialized GLMmodel structure
 */

void
OBJLoader::deleteModel()
{
	m_Indices.clear();
	m_Normals.clear();
	m_Vertices.clear();
	m_TexCoords.clear();
	m_Groups.clear();
	m_FacetNorms.clear();
}

/* glmReadOBJ: Reads a model description from a Wavefront .OBJ file.
 * Returns a pointer to the created object which should be free'd with
 * glmDelete().
 *
 * filename - name of the file containing the Wavefront .OBJ format data.  
 */

void
OBJLoader::readOBJ(std::string &filename)
{
	FILE*     file;

	/* open the file */
	file = fopen(filename.c_str(), "r");
	if (!file) {
		fprintf(stderr, "ReadOBJ() failed: can't open data file \"%s\".\n",
			filename.c_str());
		exit(1);
	}

	/* allocate a new model */
	m_Pathname = filename;
	m_MtlLibName = "";
	m_Dir = File::GetPath(filename);
	m_Position[0] = 0.0;
	m_Position[1] = 0.0;
	m_Position[2] = 0.0;

	/* make a first pass through the file to get a count of the number
	   of vertices, normals, texcoords & triangles */
	firstPass(file);

	/* allocate memory */
	m_Vertices.resize(m_NumVertices + 1);

	if (m_NumNormals) {
		m_Normals.resize(m_NumNormals + 1);
	}
	else
		m_Normals.resize(1);


	if (m_NumTexCoords) {
		m_TexCoords.resize(m_NumTexCoords + 1);
	}
	else
		m_TexCoords.resize(1);

	/* rewind to beginning of file and read in the data this pass */
	rewind(file);

	secondPass(file);

	/* close the file */
	fclose(file);
}


bool
OBJLoader::found(std::vector<std::vector<unsigned int>> &vecMap, Index *index, unsigned int *foundIndex) {

	if (vecMap[index->v].size() != 0) {

		for (auto k : vecMap[index->v]) {
			Index *ii = &m_Indices[k];
			if (ii->v == index->v && ii->n == index->n && ii->t == index->t) {
				*foundIndex = ii->realIndex;
				return true;
			}
		}
		return false;
	}
	else
		return false;
}


void 
OBJLoader::loadScene (nau::scene::IScene *aScene, std::string &aFilename, std::string &params)
{
	OBJLoader obj;
	// Read OBJ file
	obj.readOBJ(aFilename);

	unsigned int verts;

	if (!obj.m_NumNormals) {
		obj.faceNormals();

		// build structure with vertex index -> list of triangle indices
		std::vector<std::vector<unsigned int> > triIndices(obj.m_NumVertices + 1);
		std::vector<std::vector<std::set<unsigned int>>> triNormalIndexes(obj.m_NumVertices + 1);

		//for (unsigned int i = 0; i < obj.m_NumVertices + 1; ++i) {
		//	triIndices[i] = NULL;
		//}

		for (unsigned int i = 0; i < obj.m_NumTriangles * 3; ++i) {

			int vert = obj.m_Indices[i].v;
			int triangle = i;
			//if (triIndices[vert] == NULL)
			//	triIndices[vert] = new std::vector<unsigned int>;
			triIndices[vert].push_back(triangle);
		}


		unsigned int count, k;
		float dots, average[3];
		std::set<unsigned int> done, partial;
		done.clear(); partial.clear();
		std::vector<unsigned int> work;
		unsigned int numNormals = 0;
		for (unsigned int i = 1; i < obj.m_NumVertices + 1; ++i) {
			count = (unsigned int)triIndices[i].size();
			work = (triIndices[i]);
			for (k = 0; k < work.size(); ++k) {
				if (done.find(k) == done.end()) {
					done.insert(k);
					partial.insert(work[k]);
					for (unsigned int j = k + 1; j < work.size(); ++j)  {
						if (done.find(j) == done.end()) {
							dots = obj.dot(&obj.m_FacetNorms[(1 + work[k] / 3) ].x, &obj.m_FacetNorms[(1 + work[j] / 3) ].x);
							if (dots > 0.01f) {
								done.insert(j);
								partial.insert(work[j]);
							}
						}
					}
					numNormals++;
					triNormalIndexes[i].push_back(partial);
					partial.clear();
				}
			}
			done.clear();
		}
		triIndices.clear();

		obj.m_Normals.resize(numNormals + 1);
		obj.m_NumNormals = numNormals;
		std::set<unsigned int>::iterator iter;
		std::set<unsigned int> aSet;
		bool avg;
		unsigned int normalIndex = 0, triangle;
		for (unsigned int i = 1; i < triNormalIndexes.size(); ++i) {

			for (unsigned int j = 0; j < triNormalIndexes[i].size(); ++j) {
				aSet = triNormalIndexes[i][j];
				iter = aSet.begin();
				triangle = *iter / 3 + 1;
				obj.m_Indices[*iter].n = normalIndex;
				average[0] = obj.m_FacetNorms[triangle].x;
				average[1] = obj.m_FacetNorms[triangle].y;
				average[2] = obj.m_FacetNorms[triangle].z;
				iter++;
				avg = false;
				for (; iter != aSet.end(); iter++) {
					avg = true;
					triangle = *iter / 3 + 1;
					obj.m_Indices[*iter].n = normalIndex;
					average[0] += obj.m_FacetNorms[triangle].x;
					average[1] += obj.m_FacetNorms[triangle].y;
					average[2] += obj.m_FacetNorms[triangle].z;
				}
				if (avg)
					obj.normalize(average);
				obj.m_Normals[normalIndex] = VertexData::Attr(average[0], average[1], average[2], 0.0f);
				normalIndex++;
			}
		}
		triNormalIndexes.clear();
	}

	std::vector<std::vector<unsigned int>> vecMap;
	vecMap.resize(obj.m_NumVertices + 1);

	obj.m_Indices[0].realIndex = 0;
	obj.m_Indices[1].realIndex = 1;
	obj.m_Indices[2].realIndex = 2;
	vecMap[obj.m_Indices[0].v].push_back(0);
	vecMap[obj.m_Indices[1].v].push_back(1);
	vecMap[obj.m_Indices[2].v].push_back(2);

	verts = 3;
	bool k; unsigned int foundIndex;
	for (unsigned int i = 3; i < obj.m_NumTriangles*3; ++i) {
		
		k = obj.found(vecMap, &(obj.m_Indices[i]), &foundIndex);
		if (k) {
				obj.m_Indices[i].realIndex = foundIndex;
		}
		else {
			obj.m_Indices[i].realIndex = verts;
			vecMap[obj.m_Indices[i].v].push_back(i);
			verts ++;
		}

	}
	std::shared_ptr<std::vector<VertexData::Attr>> t;

	std::shared_ptr<std::vector<VertexData::Attr>> v = 
		std::shared_ptr<std::vector<VertexData::Attr>>(new std::vector<VertexData::Attr>); 
	v->reserve(verts);
	std::shared_ptr<std::vector<VertexData::Attr>> n = 
		std::shared_ptr<std::vector<VertexData::Attr>>(new std::vector<VertexData::Attr>);
	n->reserve(verts);
	if (obj.m_NumTexCoords) {
		t = std::shared_ptr<std::vector<VertexData::Attr>>(new std::vector<VertexData::Attr>);
		t->reserve(verts);
	}	
	verts = 0;
	for (unsigned int i = 0; i < obj.m_NumTriangles*3; ++i) {
		
		if (! (obj.m_Indices[i].realIndex < verts) ) {
			v->push_back(obj.m_Vertices[obj.m_Indices[i].v]); 
			n->push_back(obj.m_Normals[obj.m_Indices[i].n]);
			if (obj.m_NumTexCoords)
				t->push_back(obj.m_TexCoords[obj.m_Indices[i].t]);
			verts++;
		}
	}


	unsigned int primitive;
	if (params.find("USE_ADJACENCY") != std::string::npos)
		primitive = IRenderable::TRIANGLES_ADJACENCY;
	else
		primitive = IRenderable::TRIANGLES;



	// Create an object for the Model.  
	SceneObject *aObject = SceneObjectFactory::Create("SimpleObject");

	// Get and set name as the model path 
	aObject->setName(obj.m_Pathname);

	// Create standard Bounding Box
	IBoundingVolume *aBoundingVolume = BoundingVolumeFactory::create ("BoundingBox");
	// Bind it to the object
	aObject->setBoundingVolume (aBoundingVolume);
	// Use the vertices list to calculate bounds and center.
	aBoundingVolume->calculate(v);

	// Transform
	mat4 aTransform;
	// Set Transform
	aObject->setTransform(aTransform);

	// Renderable
	// Set Renderable Factory 
	IRenderable *aRenderable = 0;
	aRenderable = RESOURCEMANAGER->createRenderable ("Mesh","unnamed", aObject->getName());
	aRenderable->setDrawingPrimitive(primitive);

	// Import VERTEX/NORMAL/TEXTURE data into Renderable
	std::shared_ptr<VertexData> vdata = aRenderable->getVertexData(); 

	vdata->setDataFor(VertexData::GetAttribIndex(std::string("position")), v);
	vdata->setDataFor(VertexData::GetAttribIndex(std::string("normal")), n);
	if (obj.m_NumTexCoords)
		vdata->setDataFor(VertexData::GetAttribIndex(std::string("texCoord0")), t);


	// ARE THERE ANY ACTUAL MATERIALS DEFINED?
	//if (obj.m_NumMaterials==0) {

		// There HAS to be a material
		// Create material
		if (!MATERIALLIBMANAGER->hasMaterial(DEFAULTMATERIALLIBNAME,"DefaultOBJMaterial"))
			MATERIALLIBMANAGER->createMaterial("DefaultOBJMaterial");
	//}



	Group *currG;

	for (auto g: obj.m_Groups) {

		currG = &g.second;
		// ARE THERE ANY ACTUAL MATERIALS DEFINED?
			
		std::string s;
		if (obj.m_NumMaterials==0)
			// NONE! Use default
			s = "DefaultOBJMaterial";
		else {
			// Set material group name
			s = currG->material;
		//	if (s == "")
		//		s = "DefaultOBJMat";
		}

		std::shared_ptr<MaterialGroup> aMatGroup = MaterialGroup::Create(aRenderable, s);

		// Set up the index array

		// Create array
		std::shared_ptr<std::vector<unsigned int>> iArr = 
			std::shared_ptr<std::vector<unsigned int>>(new std::vector<unsigned int>);
		iArr->resize(currG->numTriangles*3);

		for (unsigned int i = 0; i < currG->numTriangles; ++i) {
			(*iArr)[i*3]   = obj.m_Indices[currG->indices[i] * 3].realIndex;
			(*iArr)[i*3+1] = obj.m_Indices[currG->indices[i] * 3 + 1].realIndex;
			(*iArr)[i*3+2] = obj.m_Indices[currG->indices[i] * 3 + 2].realIndex;
		}
		// Assign it to Material Group
		aMatGroup->setIndexList(iArr);
		if (primitive == IRenderable::TRIANGLES_ADJACENCY)
			aMatGroup->getIndexData()->useAdjacency(true);

		// Add it to the Renderable
		if (currG->numTriangles > 0)
			aRenderable->addMaterialGroup(aMatGroup);

	}

	// Set the Object's Renderable
	aObject->setRenderable (aRenderable);

	// Delete temporary data
	obj.deleteModel();

	// Add Object to the scene 
	aScene->add (aObject);
	

}

void OBJLoader::writeScene (nau::scene::IScene *aScene, std::string &aFilename)
{

}
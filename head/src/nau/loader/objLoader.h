#ifndef OBJLOADER_H
#define OBJLOADER_H

#include "nau/scene/iScene.h"

namespace nau 
{

	namespace loader 
	{
		// Wavefront OBJ format loader
		// Uses Nate Robin's GLM implementation as the core
		class OBJLoader
		{
		public:
			// Load Scene
			static void loadScene (nau::scene::IScene *aScene, std::string &aFilename, std::string &params);
			// Write Scene
			static void writeScene (nau::scene::IScene *aScene, std::string &aFilename);

		private:
			// Constructor
			OBJLoader(void) {};
			// Destructor
			~OBJLoader(void) {};

			typedef struct _Index {
				unsigned int v;			/* array of triangle vertex indices */
				unsigned int n;			/* array of triangle normal indices */
				unsigned int t;			/* array of triangle texcoord indices*/
				unsigned int realIndex;
			} Index;

			typedef struct _Group {
				std::string					name;		/* name of this group */
				unsigned int				numTriangles;	/* number of triangles in this group */
				std::vector<unsigned int>	indices;		/* array of triangle indices */
				std::string					material;           /* index to material for group */
			} Group;

			std::string    m_Pathname;			/* path to this model */
			std::string	   m_Dir;
			std::string    m_MtlLibName;			/* name of the material library */
			
			unsigned int   m_NumVertices;			/* number of vertices in model */
			std::vector<VertexData::Attr> m_Vertices;			/* array of vertices  */
			
			unsigned int   m_NumNormals;			/* number of normals in model */
			std::vector<VertexData::Attr> m_Normals;			/* array of normals */
			
			unsigned int   m_NumTexCoords;		/* number of texcoords in model */
			std::vector<VertexData::Attr> m_TexCoords;			/* array of texture coordinates */
			
			unsigned int   m_NumFacetNorms;		/* number of facetnorms in model */
			std::vector<vec3> m_FacetNorms;			/* array of facetnorms */
			
			unsigned int       m_NumTriangles;		/* number of triangles in model */
			std::vector<Index> m_Indices;		/* array of indices */
			
			unsigned int       m_NumMaterials;		/* number of materials in model */
			
			unsigned int       m_NumGroups;		/* number of groups in model */
			std::map<std::string, Group>    m_Groups;			/* linked list of groups */
			
			float m_Position[3];			/* position of the model */

			void readOBJ(std::string &filename);
			void firstPass(FILE* file);
			void secondPass(FILE* file);
			void readMTL(std::string &name);
			
			void faceNormals();
			void deleteModel();
			void initGroup(std::string &s);

			void addIndex(unsigned int index, unsigned int v, unsigned int t, unsigned int n);
			bool found(std::vector<std::vector<unsigned int>> &vecMap, Index *index, unsigned int *foundIndex);
			float dot(float* u, float * v);
			void cross(float* u, float* v, float* n);
			void normalize(float *v);

		};
	};
};

#endif //OBJLOADER_H
///////////NOVO//////////////

#ifndef DISTANCEMAP_H
#define DISTANCEMAP_H

#include "nau/math/ray.h"

namespace nau
{
	namespace geometry
	{
		class DistanceMap
		{
		private:
			std::vector<std::vector<vec4>>* distmap;
			unsigned int rows;
			unsigned int columns;
			std::vector<std::vector<int>>* meshId;
			std::vector<std::vector<int>>* triangleId;
			std::vector<std::vector<float>>* diffuse;
			//std::vector<std::vector<IScene*>>* scene;
			std::vector<std::vector<std::vector<int>>>* inShadow;
			std::vector<int>* lightId;
			std::vector<std::vector<std::vector<std::vector<vec2>>>>* shadingIDs;
			std::vector<std::vector<std::vector<std::vector<vec3>>>>* shadingVertices;
			std::vector<std::vector<std::vector<std::vector<vec4>>>>* shadingIntersections;
			std::vector<std::vector<std::vector<std::vector<int>>>>* pixelPosition;
			std::vector<std::vector<std::vector<int>>>* intersectionCount;
			std::vector<std::vector<float>>* normalAngle;

		public:
			DistanceMap();
			DistanceMap(std::vector<std::vector<vec4>>* dm);
			DistanceMap(std::vector<std::vector<vec4>>* dm, std::vector<std::vector<int>>* mid,
				std::vector<std::vector<int>>* tid);
			~DistanceMap();

			std::vector<std::vector<vec4>>& getDistMap();
			std::vector<std::vector<int>>& getMeshId();
			std::vector<std::vector<int>>& getTriangleId();
			std::vector<std::vector<float>>& getDiffuse();
			//std::vector<std::vector<IScene*>>& getScene();
			std::vector<std::vector<std::vector<int>>>& getInShadow();
			std::vector<std::vector<std::vector<std::vector<vec2>>>>& getShadingIDs();
			std::vector<std::vector<std::vector<std::vector<vec3>>>>& getShadingVertices();
			std::vector<std::vector<std::vector<std::vector<vec4>>>>& getShadingIntersections();
			std::vector<std::vector<std::vector<std::vector<int>>>>& getPixelPosition();
			std::vector<std::vector<std::vector<int>>>& getIntersectionCount();
			std::vector<std::vector<float>>& getNormalAngle();
			std::vector<std::vector<vec4>>* getDistMapPointer();
			std::vector<std::vector<int>>* getMeshIdPointer();
			std::vector<std::vector<int>>* getTriangleIdPointer();
			std::vector<std::vector<float>>* getDiffusePointer();
			//std::vector<std::vector<IScene*>>& getScenePointer();
			std::vector<std::vector<std::vector<int>>>* getInShadowPointer();
			std::vector<std::vector<std::vector<std::vector<vec2>>>>* getShadingIDsPointer();
			std::vector<std::vector<std::vector<std::vector<vec3>>>>* getShadingVerticesPointer();
			std::vector<std::vector<std::vector<std::vector<vec4>>>>* getShadingIntersectionsPointer();
			std::vector<std::vector<std::vector<std::vector<int>>>>* getPixelPositionPointer();
			std::vector<std::vector<std::vector<int>>>* getIntersectionCountPointer();
			std::vector<std::vector<float>>* getNormalAnglePointer();
			unsigned int getRows();
			unsigned int getColumns();

			void closerIntersections(DistanceMap * prm, vec3 point);

			void toArray(vec3 origin, float maxDistance, unsigned char * dados, unsigned int width, unsigned int height);
			void shadowToArray(unsigned char ** dados, unsigned int width, unsigned int height);
			void shadowToArrayWithNormals(unsigned char ** dados, unsigned int width, unsigned int height);

			void setShadowMap(std::vector<std::vector<int>>* sm, int lightId);
			void setShadowMapWithShaders(std::vector<std::vector<int>>* sm, int lid,
				std::vector<std::vector<std::vector<vec2>>>* sids, std::vector<std::vector<std::vector<vec3>>>* sv,
				std::vector<std::vector<std::vector<vec4>>>* si);
			void setShadowMapWithShadersAndNeighbourStuff(std::vector<std::vector<int>>* sm, int lid,
				std::vector<std::vector<std::vector<vec2>>>* sids, std::vector<std::vector<std::vector<vec3>>>* sv,
				std::vector<std::vector<std::vector<vec4>>>* si, std::vector<std::vector<std::vector<int>>>* spp,
				std::vector<std::vector<int>>* sic);

			//void setRows(unsigned int r);
			//void setColumns(unsigned int c);

			//void makeDistanceMap(PrimaryRayMatrix prm, nau::scene::Octree o);

			void makeRasterDistanceMap(unsigned int pr, unsigned int pc);
			void makeRasterDistanceMapWithNormals(unsigned int pr, unsigned int pc);
		};
	};
};

#endif
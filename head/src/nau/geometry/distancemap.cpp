///////////NOVO//////////////

#include "nau/geometry/distancemap.h"

#include "nau.h"

using namespace nau::geometry;

DistanceMap::DistanceMap(void) :
	distmap (0),
	rows (0),
	columns (0),
	meshId (0),
	triangleId (0),
	diffuse (0),
	//scene (0),
	inShadow (0),
	lightId (0),
	shadingIDs (0),
	shadingVertices (0),
	shadingIntersections (0),
	pixelPosition (0),
	intersectionCount (0),
	normalAngle (0)
{
	distmap = new std::vector<std::vector<vec4>>;
	meshId = new std::vector<std::vector<int>>;
	triangleId = new std::vector<std::vector<int>>;
	diffuse = new std::vector<std::vector<float>>;
	//scene = new std::vector<std::vector<IScene*>>;
	inShadow = new std::vector<std::vector<std::vector<int>>>;
	lightId = new std::vector<int>;
	shadingIDs = new std::vector<std::vector<std::vector<std::vector<vec2>>>>;
	shadingVertices = new std::vector<std::vector<std::vector<std::vector<vec3>>>>;
	shadingIntersections = new std::vector<std::vector<std::vector<std::vector<vec4>>>>;
	pixelPosition = new std::vector<std::vector<std::vector<std::vector<int>>>>;
	intersectionCount = new std::vector<std::vector<std::vector<int>>>;
	normalAngle = new std::vector<std::vector<float>>;
}

DistanceMap::DistanceMap(std::vector<std::vector<vec4>>* dm) :
	distmap (0),
	rows (0),
	columns (0),
	meshId (0),
	triangleId (0),
	diffuse (0),
	//scene (0),
	inShadow (0),
	lightId (0),
	shadingIDs (0),
	shadingVertices (0),
	shadingIntersections (0),
	pixelPosition (0),
	intersectionCount (0),
	normalAngle (0)
{
	distmap = dm;
	rows = dm->size();
	if(rows>0)
	{
		columns = (dm->at(0)).size();
	}
	else
	{
		columns = 0;
	}
	meshId = new std::vector<std::vector<int>>;
	triangleId = new std::vector<std::vector<int>>;
	diffuse = new std::vector<std::vector<float>>;
	//scene = new std::vector<std::vector<IScene*>>;
	inShadow = new std::vector<std::vector<std::vector<int>>>;
	lightId = new std::vector<int>;
	shadingIDs = new std::vector<std::vector<std::vector<std::vector<vec2>>>>;
	shadingVertices = new std::vector<std::vector<std::vector<std::vector<vec3>>>>;
	shadingIntersections = new std::vector<std::vector<std::vector<std::vector<vec4>>>>;
	pixelPosition = new std::vector<std::vector<std::vector<std::vector<int>>>>;
	intersectionCount = new std::vector<std::vector<std::vector<int>>>;
	normalAngle = new std::vector<std::vector<float>>;
}

DistanceMap::DistanceMap(std::vector<std::vector<vec4>>* dm, std::vector<std::vector<int>>* mid,
	std::vector<std::vector<int>>* tid) :
	distmap (0),
	rows (0),
	columns (0),
	meshId (0),
	triangleId (0),
	diffuse (0),
	//scene (0),
	inShadow (0),
	lightId (0),
	shadingIDs (0),
	shadingVertices (0),
	shadingIntersections (0),
	pixelPosition (0),
	intersectionCount (0),
	normalAngle (0)
{
	distmap = dm;
	rows = dm->size();
	if(rows>0)
	{
		columns = (dm->at(0)).size();
	}
	else
	{
		columns = 0;
	}
	meshId = mid;
	triangleId = tid;
	diffuse = new std::vector<std::vector<float>>;
	//scene = is;
	inShadow = new std::vector<std::vector<std::vector<int>>>;
	lightId = new std::vector<int>;
	shadingIDs = new std::vector<std::vector<std::vector<std::vector<vec2>>>>;
	shadingVertices = new std::vector<std::vector<std::vector<std::vector<vec3>>>>;
	shadingIntersections = new std::vector<std::vector<std::vector<std::vector<vec4>>>>;
	pixelPosition = new std::vector<std::vector<std::vector<std::vector<int>>>>;
	intersectionCount = new std::vector<std::vector<std::vector<int>>>;
	normalAngle = new std::vector<std::vector<float>>;
}
			
DistanceMap::~DistanceMap(void)
{
	if (0 != distmap)
	{
		delete distmap;
	}
	if (0 != meshId)
	{
		delete meshId;
	}
	if (0 != triangleId)
	{
		delete triangleId;
	}
	if (0 != diffuse)
	{
		delete diffuse;
	}
	//if (0 != scene)
	//{
	//	delete scene;
	//}
	if (0 != inShadow)
	{
		delete inShadow;
	}
	if (0 != lightId)
	{
		delete lightId;
	}
	if(0 != shadingIDs)
	{
		delete shadingIDs;
	}
	if(0 != shadingVertices)
	{
		delete shadingVertices;
	}
	if(0 != shadingIntersections)
	{
		delete shadingVertices;
	}
	if(0 != pixelPosition)
	{
		delete pixelPosition;
	}
	if(0 != intersectionCount)
	{
		delete intersectionCount;
	}
	if(0 != normalAngle)
	{
		delete normalAngle;
	}
}

std::vector<std::vector<vec4>>&
DistanceMap::getDistMap()
{
	return (*distmap);
}

std::vector<std::vector<int>>&
DistanceMap::getMeshId()
{
	return (*meshId);
}

std::vector<std::vector<int>>&
DistanceMap::getTriangleId()
{
	return (*triangleId);
}

std::vector<std::vector<float>>&
DistanceMap::getDiffuse()
{
	return (*diffuse);
}

//std::vector<std::vector<IScene*>>&
//DistanceMap::getScene()
//{
//	return (*scene);
//}

std::vector<std::vector<std::vector<int>>>&
DistanceMap::getInShadow()
{
	return (*inShadow);
}

std::vector<std::vector<std::vector<std::vector<vec2>>>>&
DistanceMap::getShadingIDs()
{
	return (*shadingIDs);
}

std::vector<std::vector<std::vector<std::vector<vec3>>>>&
DistanceMap::getShadingVertices()
{
	return (*shadingVertices);
}

std::vector<std::vector<std::vector<std::vector<vec4>>>>&
DistanceMap::getShadingIntersections()
{
	return (*shadingIntersections);
}

std::vector<std::vector<std::vector<std::vector<int>>>>&
DistanceMap::getPixelPosition()
{
	return (*pixelPosition);
}

std::vector<std::vector<std::vector<int>>>&
DistanceMap::getIntersectionCount()
{
	return (*intersectionCount);
}

std::vector<std::vector<float>>&
DistanceMap::getNormalAngle()
{
	return (*normalAngle);
}

std::vector<std::vector<vec4>>*
DistanceMap::getDistMapPointer()
{
	return distmap;
}

std::vector<std::vector<int>>*
DistanceMap::getMeshIdPointer()
{
	return meshId;
}

std::vector<std::vector<int>>*
DistanceMap::getTriangleIdPointer()
{
	return triangleId;
}

std::vector<std::vector<float>>*
DistanceMap::getDiffusePointer()
{
	return diffuse;
}

//std::vector<std::vector<IScene*>>*
//DistanceMap::getScenePointer()
//{
//	return scene;
//}

std::vector<std::vector<std::vector<int>>>*
DistanceMap::getInShadowPointer()
{
	return inShadow;
}

std::vector<std::vector<std::vector<std::vector<vec2>>>>*
DistanceMap::getShadingIDsPointer()
{
	return shadingIDs;
}

std::vector<std::vector<std::vector<std::vector<vec3>>>>*
DistanceMap::getShadingVerticesPointer()
{
	return shadingVertices;
}

std::vector<std::vector<std::vector<std::vector<vec4>>>>*
DistanceMap::getShadingIntersectionsPointer()
{
	return shadingIntersections;
}

std::vector<std::vector<std::vector<std::vector<int>>>>*
DistanceMap::getPixelPositionPointer()
{
	return pixelPosition;
}

std::vector<std::vector<std::vector<int>>>*
DistanceMap::getIntersectionCountPointer()
{
	return intersectionCount;
}

std::vector<std::vector<float>>*
DistanceMap::getNormalAnglePointer()
{
	return normalAngle;
}

unsigned int
DistanceMap::getRows()
{
	return rows;
}

unsigned int
DistanceMap::getColumns()
{
	return columns;
}

void
DistanceMap::closerIntersections(DistanceMap * dm, vec3 point)
{
	std::vector<std::vector<vec4>> dmAux;
	std::vector<std::vector<int>> tid, mid;
	//std::vector<std::vector<IScene*>> is;
	dmAux = dm->getDistMap();
	tid = dm->getTriangleId();
	mid = dm->getMeshId();
	//is = dm->getScene();
	unsigned int cAux = dm->getColumns();
	unsigned int rAux = dm->getRows();

	if(cAux < columns)
	{
		columns = cAux;
	}
	if(rAux < rows)
	{
		rows = rAux;
	}

	std::vector<std::vector<vec4>>::iterator thisRIt = distmap->begin();
	std::vector<std::vector<vec4>>::iterator auxRIt = dmAux.begin();
	std::vector<vec4>::iterator thisCIt;
	std::vector<vec4>::iterator auxCIt;
	vec3 thisVec;
	vec3 auxVec;

	std::vector<std::vector<int>>::iterator thisTidIter = triangleId->begin();
	std::vector<std::vector<int>>::iterator auxTidIter = tid.begin();
	std::vector<int>::iterator thisCTidIt;
	std::vector<int>::iterator auxCTidIt;

	std::vector<std::vector<int>>::iterator thisMidIter = meshId->begin();
	std::vector<std::vector<int>>::iterator auxMidIter = mid.begin();
	std::vector<int>::iterator thisCMidIt;
	std::vector<int>::iterator auxCMidIt;

	//std::vector<std::vector<IScene*>>::iterator thisSceneIter = scene->begin();
	//std::vector<std::vector<IScene*>>::iterator auxSceneIter = is.begin();
	//std::vector<IScene*>::iterator thisCSceneIt;
	//std::vector<IScene*>::iterator auxCSceneIt;

	unsigned int linhas, colunas;
	distmap->resize(rows);
	triangleId->resize(rows);
	meshId->resize(rows);
	//scene->resize(rows);
	for(linhas = 0; linhas < rows; linhas ++)
	{
		(*thisRIt).resize(columns);
		(*thisTidIter).resize(columns);
		(*thisMidIter).resize(columns);
		//(*thisSceneIter).resize(columns);
		thisCIt = (*thisRIt).begin();
		auxCIt = (*auxRIt).begin();
		thisCTidIt = (*thisTidIter).begin();
		auxCTidIt = (*auxTidIter).begin();
		thisCMidIt = (*thisMidIter).begin();
		auxCMidIt = (*auxMidIter).begin();
		//thisCSceneIt = (*thisSceneIter).begin();
		//auxCSceneIt = (*auxSceneIter).begin();
		for(colunas = 0; colunas < columns; colunas ++)
		{
			if((*auxCIt).w == 1)
			{
				if((*thisCIt).w == 0)
				{
					(*thisCIt).x = (*auxCIt).x;
					(*thisCIt).y = (*auxCIt).y;
					(*thisCIt).z = (*auxCIt).z;
					(*thisCIt).w = (*auxCIt).w;
					(*thisCTidIt) = (*auxCTidIt);
					(*thisCMidIt) = (*auxCMidIt);
					//(*thisCSceneIt) = (*auxCSceneIt);
				}
				else
				{
					thisVec.x = (*thisCIt).x;
					thisVec.y = (*thisCIt).y;
					thisVec.z = (*thisCIt).z;
					auxVec.x = (*auxCIt).x;
					auxVec.y = (*auxCIt).y;
					auxVec.z = (*auxCIt).z;
					if(point.distance(thisVec) > point.distance(auxVec))
					{
						(*thisCIt).x = (*auxCIt).x;
						(*thisCIt).y = (*auxCIt).y;
						(*thisCIt).z = (*auxCIt).z;
						(*thisCTidIt) = (*auxCTidIt);
						(*thisCMidIt) = (*auxCMidIt);
						//(*thisCSceneIt) = (*auxCSceneIt);
					}
				}
			}
			thisCIt++; auxCIt++;
			thisCTidIt++; auxCTidIt++; thisCMidIt++; auxCMidIt++;
			//thisCSceneIt++; auxCSceneIt++;
		}
		thisRIt++; auxRIt++;
		thisTidIter++; auxTidIter++; thisMidIter++;	auxMidIter++;
		//thisSceneIter++; auxSceneIter++;
	}
}

void
DistanceMap::toArray(vec3 origin, float maxDistance, unsigned char * dados, unsigned int width, unsigned int height)
{
	vec3 aux;
	std::vector<std::vector<vec4>>::iterator rowIter = distmap->begin();
	std::vector<vec4>::iterator colIter;
	float res;
	int i = 0;
	unsigned int heightAux = 0, widthAux = 0;

	for(; rowIter != distmap->end(); rowIter++)
	{
		widthAux = 0;
		colIter = (*rowIter).begin();
		for(; colIter != (*rowIter).end(); colIter++)
		{
			if((*colIter).w == 1)
			{
				aux.x = (*colIter).x;
				aux.y = (*colIter).y;
				aux.z = (*colIter).z;
				res = 1-((aux.distance(origin))/maxDistance);
				dados[i] = (char)(res*255);
			}
			else
			{
				dados[i] = 0;
			}
			i++;
			widthAux++;
		}
		for(; widthAux<width; widthAux++)
		{
			dados[i] = 0;
			i++;
		}
		heightAux++;
	}
	for(; heightAux<height; heightAux++)
	{
		for(widthAux = 0; widthAux<width; widthAux++)
		{
			dados[i] = 0;
			i++;
		}
	}
}

void
DistanceMap::shadowToArray(unsigned char ** dados, unsigned int width, unsigned int height)
{
	std::vector<std::vector<std::vector<int>>>::iterator smlIter = inShadow->begin();
	std::vector<std::vector<int>>::iterator smLineIter;
	std::vector<int>::iterator smColIter;
	int smlPos = 0, i = 0, widthAux, heightAux;
	for(; smlIter != inShadow->end(); smlIter++)
	{
		heightAux=0;
		smLineIter = (*smlIter).begin();
		for(; smLineIter != (*smlIter).end(); smLineIter++)
		{
			widthAux=0;
			smColIter = (*smLineIter).begin();
			for(; smColIter != (*smLineIter).end(); smColIter++)
			{
				switch(*smColIter)
				{
					case 0:
						dados[smlPos][i] = 255; i++;
						dados[smlPos][i] = 255; i++;
						dados[smlPos][i] = 255; i++;
						break;
					case 1:
						dados[smlPos][i] = 0; i++;
						dados[smlPos][i] = 0; i++;
						dados[smlPos][i] = 0; i++;
						break;
					case 2:
						dados[smlPos][i] = 255; i++;
						dados[smlPos][i] = 0; i++;
						dados[smlPos][i] = 0; i++;
						break;
					default:
						dados[smlPos][i] = 0; i++;
						dados[smlPos][i] = 255; i++;
						dados[smlPos][i] = 0; i++;
						break;
				}
				widthAux++;
			}
			for(; widthAux < width; widthAux++)
			{
				dados[smlPos][i] = 0; i++;
				dados[smlPos][i] = 255; i++;
				dados[smlPos][i] = 0; i++;
			}
			heightAux++;
		}
		for(; heightAux < height; heightAux++)
		{
			dados[smlPos][i] = 0; i++;
			dados[smlPos][i] = 255; i++;
			dados[smlPos][i] = 0; i++;
		}
		smlPos++;
	}
}

void
DistanceMap::shadowToArrayWithNormals(unsigned char ** dados, unsigned int width, unsigned int height)
{
	std::vector<std::vector<std::vector<int>>>::iterator smlIter = inShadow->begin();
	std::vector<std::vector<int>>::iterator smLineIter;
	std::vector<int>::iterator smColIter;
	std::vector<std::vector<float>>::iterator normlIter;
	std::vector<float>::iterator normColIter;
	int smlPos = 0, i = 0, widthAux, heightAux;
	for(; smlIter != inShadow->end(); smlIter++)
	{
		heightAux=0;
		smLineIter = (*smlIter).begin();
		normlIter = normalAngle->begin();
		for(; smLineIter != (*smlIter).end(); smLineIter++)
		{
			widthAux=0;
			smColIter = (*smLineIter).begin();
			normColIter = (*normlIter).begin();
			for(; smColIter != (*smLineIter).end(); smColIter++)
			{
				switch(*smColIter)
				{
					case 0:
						if((*normColIter) <= 0)
						{
							dados[smlPos][i] = 0; i++;
							dados[smlPos][i] = 0; i++;
							dados[smlPos][i] = 0; i++;
						}
						else
						{
							dados[smlPos][i] = 255; i++;
							dados[smlPos][i] = 255; i++;
							dados[smlPos][i] = 255; i++;
						}
						break;
					case 1:
						dados[smlPos][i] = 0; i++;
						dados[smlPos][i] = 0; i++;
						dados[smlPos][i] = 0; i++;
						break;
					case 2:
						dados[smlPos][i] = 255; i++;
						dados[smlPos][i] = 0; i++;
						dados[smlPos][i] = 0; i++;
						break;
					default:
						dados[smlPos][i] = 0; i++;
						dados[smlPos][i] = 255; i++;
						dados[smlPos][i] = 0; i++;
						break;
				}
				widthAux++;
				normColIter++;
			}
			for(; widthAux < width; widthAux++)
			{
				dados[smlPos][i] = 0; i++;
				dados[smlPos][i] = 255; i++;
				dados[smlPos][i] = 0; i++;
			}
			heightAux++;
			normlIter++;
		}
		for(; heightAux < height; heightAux++)
		{
			dados[smlPos][i] = 0; i++;
			dados[smlPos][i] = 255; i++;
			dados[smlPos][i] = 0; i++;
		}
		smlPos++;
	}
}

void
DistanceMap::setShadowMap(std::vector<std::vector<int>>* sm, int lid)
{
	std::vector<int>::iterator idIter = lightId->begin();
	std::vector<std::vector<std::vector<int>>>::iterator smlIter = inShadow->begin();
	for(; idIter != lightId->end() && (*idIter) != lid; idIter++)
	{
		smlIter++;
	}
	if(idIter == lightId->end())
	{
		inShadow->push_back((*sm));
		lightId->push_back(lid);
	}
	else
	{
		std::vector<std::vector<int>>::iterator smthisIter = (*smlIter).begin(), smotherIter = sm->begin();
		std::vector<int>::iterator smthisColIter, smotherColIter;
		for(; smthisIter != (*smlIter).end(); smthisIter++)
		{
			smthisColIter = (*smthisIter).begin();
			smotherColIter = (*smotherIter).begin();
			for(; smthisColIter != (*smthisIter).end(); smthisColIter++)
			{
				if((*smotherColIter) != 2 && (*smthisColIter) != 1)
				{
					(*smthisColIter)=(*smotherColIter);
				}
				smotherColIter++;
			}
			smotherIter++;
		}
	}
}

void
DistanceMap::setShadowMapWithShaders(std::vector<std::vector<int>>* sm, int lid,
	std::vector<std::vector<std::vector<vec2>>>* sids, std::vector<std::vector<std::vector<vec3>>>* sv,
	std::vector<std::vector<std::vector<vec4>>>* si)
{
	std::vector<int>::iterator idIter = lightId->begin();
	std::vector<std::vector<std::vector<int>>>::iterator smlIter = inShadow->begin();
	std::vector<std::vector<std::vector<std::vector<vec2>>>>::iterator sidsIter = shadingIDs->begin();
	std::vector<std::vector<std::vector<std::vector<vec3>>>>::iterator svIter = shadingVertices->begin();
	std::vector<std::vector<std::vector<std::vector<vec4>>>>::iterator siIter = shadingIntersections->begin();
	for(; idIter != lightId->end() && (*idIter) != lid; idIter++)
	{
		smlIter++;
		sidsIter++;
		svIter++;
		siIter++;
	}
	if(idIter == lightId->end())
	{
		inShadow->push_back((*sm));
		sm->clear();
		lightId->push_back(lid);
		shadingIDs->push_back(*sids);
		sids->clear();
		shadingVertices->push_back(*sv);
		sv->clear();
		shadingIntersections->push_back(*si);
		si->clear();
	}
	else
	{
		(*smlIter) = (*sm);
		sm->clear();
		(*sidsIter) = (*sids);
		sids->clear();
		(*svIter) = (*sv);
		sv->clear();
		(*siIter) = (*si);
		si->clear();
		/*std::vector<int>::iterator smthisIter = (*smlIter).begin(), smotherIter = sm->begin();
		std::vector<std::vector<vec2>>::iterator sidsthisIter = (*sidsIter).begin(), sidsotherIter = sids->begin();
		std::vector<std::vector<vec3>>::iterator svthisIter = (*svIter).begin(), svotherIter = sv->begin();
		for(; smthisIter != (*smlIter).end(); smthisIter++)
		{
			if((*smotherIter) != 2 && (*smthisIter) != 1)
			{
				(*smthisIter)=(*smotherIter);
				(*sidsthisIter)=(*sidsotherIter);
				(*svthisIter)=(*svotherIter);
			}
			smotherIter++;
			sidsthisIter++; sidsotherIter++;
			svthisIter++; svotherIter++;
		}*/
	}
}

void
DistanceMap::setShadowMapWithShadersAndNeighbourStuff(std::vector<std::vector<int>>* sm, int lid,
	std::vector<std::vector<std::vector<vec2>>>* sids, std::vector<std::vector<std::vector<vec3>>>* sv,
	std::vector<std::vector<std::vector<vec4>>>* si, std::vector<std::vector<std::vector<int>>>* spp,
	std::vector<std::vector<int>>* sic)
{
	std::vector<int>::iterator idIter = lightId->begin();
	std::vector<std::vector<std::vector<int>>>::iterator smlIter = inShadow->begin();
	std::vector<std::vector<std::vector<std::vector<vec2>>>>::iterator sidsIter = shadingIDs->begin();
	std::vector<std::vector<std::vector<std::vector<vec3>>>>::iterator svIter = shadingVertices->begin();
	std::vector<std::vector<std::vector<std::vector<vec4>>>>::iterator siIter = shadingIntersections->begin();
	std::vector<std::vector<std::vector<std::vector<int>>>>::iterator ppIter = pixelPosition->begin();
	std::vector<std::vector<std::vector<int>>>::iterator icIter = intersectionCount->begin();
	for(; idIter != lightId->end() && (*idIter) != lid; idIter++)
	{
		smlIter++;
		sidsIter++;
		svIter++;
		siIter++;
		ppIter++;
		icIter++;
	}
	if(idIter == lightId->end())
	{
		inShadow->push_back((*sm));
		sm->clear();
		lightId->push_back(lid);
		shadingIDs->push_back(*sids);
		sids->clear();
		shadingVertices->push_back(*sv);
		sv->clear();
		shadingIntersections->push_back(*si);
		si->clear();
		pixelPosition->push_back(*spp);
		spp->clear();
		intersectionCount->push_back(*sic);
		sic->clear();
	}
	else
	{
		(*smlIter) = (*sm);
		sm->clear();
		(*sidsIter) = (*sids);
		sids->clear();
		(*svIter) = (*sv);
		sv->clear();
		(*siIter) = (*si);
		si->clear();
		(*ppIter) = (*spp);
		spp->clear();
		(*icIter) = (*sic);
		sic->clear();
	}
}

/*void
DistanceMap::setRows(unsigned int r)
{
	rows=r;
}

void
DistanceMap::setColumns(unsigned int c)
{
	columns=c;
}*/

void
DistanceMap::makeRasterDistanceMap(unsigned int pr, unsigned int pc)
{
	Texture * pixelCoords = RESOURCEMANAGER->getTexture("VisionMaterials::pixelCoords");
	TexImage * texPixelCoords = nau::material::TexImage::create(pixelCoords);
	float * pixelCoordsData = (float*) (texPixelCoords->getData());

	Texture * triangleIDs = RESOURCEMANAGER->getTexture("VisionMaterials::triangleIDs");
	TexImage * texTriangleIDs = nau::material::TexImage::create(triangleIDs);
	float * triangleIDsData = (float*) (texTriangleIDs->getData());

	int pos = 0;
	vec4 dim;

	std::vector<vec4> line;
	std::vector<int> mline;
	std::vector<int> tline;

	rows = pr;
	columns = pc;

	for(unsigned int i = 0; i<pr; i++)
	{
		for(unsigned int j = 0; j< pc; j++)
		{
			dim.x = pixelCoordsData[pos]; dim.y = pixelCoordsData[pos+1]; dim.z = pixelCoordsData[pos+2];
			if(pixelCoordsData[pos+3]<0.5)
			{
				dim.w = 0;
			}
			else
			{
				dim.w = 1;
			}
			line.push_back(dim);
			//distmap->push_back(dim);
			if(triangleIDsData[pos] - ((int) triangleIDsData[pos]) < 0.5)
			{
				//meshId->push_back((int) (triangleIDsData[pos]));
				mline.push_back((int) (triangleIDsData[pos]));
			}
			else
			{
				//meshId->push_back((int) (triangleIDsData[pos] + 1));
				mline.push_back((int) (triangleIDsData[pos] + 1));
			}
			if(triangleIDsData[pos+1] - ((int) triangleIDsData[pos+1]) < 0.5)
			{
				//triangleId->push_back((int) (triangleIDsData[pos+1]));
				tline.push_back((int) (triangleIDsData[pos+1]));
			}
			else
			{
				//triangleId->push_back((int) (triangleIDsData[pos+1] + 1));
				tline.push_back((int) (triangleIDsData[pos+1] + 1));
			}
			pos+=4;
		}
		distmap->push_back(line);
		line.clear();
		meshId->push_back(mline);
		mline.clear();
		triangleId->push_back(tline);
		tline.clear();
	}

	if(0 != texPixelCoords)
	{
		delete texPixelCoords;
	}
	if(0 != texTriangleIDs)
	{
		delete texTriangleIDs;
	}
}

void
DistanceMap::makeRasterDistanceMapWithNormals(unsigned int pr, unsigned int pc)
{
	Texture * pixelCoords = RESOURCEMANAGER->getTexture("VisionMaterials::pixelCoords");
	TexImage * texPixelCoords = nau::material::TexImage::create(pixelCoords);
	float * pixelCoordsData = (float*) (texPixelCoords->getData());

	Texture * triangleIDs = RESOURCEMANAGER->getTexture("VisionMaterials::triangleIDs");
	TexImage * texTriangleIDs = nau::material::TexImage::create(triangleIDs);
	float * triangleIDsData = (float*) (texTriangleIDs->getData());

	int pos = 0;
	vec4 dim;

	std::vector<vec4> line;
	std::vector<int> mline;
	std::vector<int> tline;
	std::vector<float> normline;

	rows = pr;
	columns = pc;

	for(unsigned int i = 0; i<pr; i++)
	{
		for(unsigned int j = 0; j< pc; j++)
		{
			dim.x = pixelCoordsData[pos]; dim.y = pixelCoordsData[pos+1]; dim.z = pixelCoordsData[pos+2];
			if(pixelCoordsData[pos+3]<0.5)
			{
				dim.w = 0;
			}
			else
			{
				dim.w = 1;
			}
			line.push_back(dim);
			//distmap->push_back(dim);
			if(triangleIDsData[pos] - ((int) triangleIDsData[pos]) < 0.5)
			{
				//meshId->push_back((int) (triangleIDsData[pos]));
				mline.push_back((int) (triangleIDsData[pos]));
			}
			else
			{
				//meshId->push_back((int) (triangleIDsData[pos] + 1));
				mline.push_back((int) (triangleIDsData[pos] + 1));
			}
			if(triangleIDsData[pos+1] - ((int) triangleIDsData[pos+1]) < 0.5)
			{
				//triangleId->push_back((int) (triangleIDsData[pos+1]));
				tline.push_back((int) (triangleIDsData[pos+1]));
			}
			else
			{
				//triangleId->push_back((int) (triangleIDsData[pos+1] + 1));
				tline.push_back((int) (triangleIDsData[pos+1] + 1));
			}
			normline.push_back(triangleIDsData[pos+2]);
			pos+=4;
		}
		distmap->push_back(line);
		line.clear();
		meshId->push_back(mline);
		mline.clear();
		triangleId->push_back(tline);
		tline.clear();
		normalAngle->push_back(normline);
		normline.clear();
	}

	if(0 != texPixelCoords)
	{
		delete texPixelCoords;
	}
	if(0 != texTriangleIDs)
	{
		delete texTriangleIDs;
	}
}
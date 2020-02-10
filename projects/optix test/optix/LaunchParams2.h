// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once
#include <stdint.h>
#include "vec_math.h"

struct vertexData {
		float4* position;
		float4* normal;
		float4* texCoord0;
		float4* tangent;
		float4* bitangent;
};

struct TriangleMeshSBTData {
    uint3 *index;
    vertexData vertexD;
		int hasTexture;
		cudaTextureObject_t texture;
    float3  color;
};

struct LaunchParams
{
  struct {
    int frame;
    uint32_t *colorBuffer;
    int2     size;
  } frame;
  
  struct {
    float3 position;
    float3 direction;
    float3 horizontal;
    float3 vertical;
  } camera;
  OptixTraversableHandle traversable;
};


struct RayGenData {
  int3 color;
};


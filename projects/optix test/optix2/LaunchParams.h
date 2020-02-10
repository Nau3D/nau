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


struct TriangleMeshSBTData {
    float3  color;
    float3 *vertex;
    int3 *index;
};

struct LaunchParams
{
  struct {
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



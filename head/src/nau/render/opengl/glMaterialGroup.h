#ifndef GLMATERIALGROUP_H
#define GLMATERIALGROUP_H

#include "nau/material/materialGroup.h"
#include "nau/render/iRenderable.h"

using namespace nau::material;

namespace nau {

	namespace render {
	
		namespace opengl {

			class GLMaterialGroup : public MaterialGroup {

			friend class MaterialGroup;

			protected:
				unsigned int m_VAO;

				GLMaterialGroup(nau::render::IRenderable *parent, std::string materialName);

			public:
				void compile();
				void resetCompilationFlag();
				bool isCompiled();

				void bind();
				void unbind();

				unsigned int getVAO();

				~GLMaterialGroup();
			};


		};
	};

};

#endif
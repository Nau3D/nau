#ifndef TEXTUREMAT_H
#define TEXTUREMAT_H

#include <vector>

#include "nau/system/file.h"
#include "nau/render/texture.h"
#include "nau/render/istate.h"
#include "nau/material/texturesampler.h"

namespace nau
{
	namespace material
	{
		class TextureMat {

		private:
			nau::render::Texture* m_Textures[8];
			nau::material::TextureSampler *m_Samplers[8];

		public:
			TextureMat();
			~TextureMat();

			TextureMat *clone();

			void setTexture(int unit, nau::render::Texture *t);
			void unset(int unit);

			std::vector<std::string> *getTextureNames();
			std::vector<int> *getTextureUnits();
			nau::render::Texture* getTexture(int unit);
			nau::material::TextureSampler* getTextureSampler(int unit);

			void prepare(nau::render::IState *state);
			void restore(nau::render::IState *state);
			void clear();
		};
	};
};

#endif //TEXTUREMAT_H

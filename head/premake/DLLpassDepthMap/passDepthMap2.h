#ifndef DEPTHMAPPASS_H
#define DEPTHMAPPASS_H


#include "iNau.h"
#include "nau/render/pass.h"

class PassDepthMap2 : public Pass
		{
		protected:
			nau::scene::Camera *m_LightCamera;

		public:

			static Pass *Create(const std::string &passName);
			PassDepthMap2(const std::string &name);
			~PassDepthMap2(void);

			virtual void prepare (void);
			virtual void doPass (void);
			virtual void restore (void);
			virtual void addLight (const std::string &light);
};

extern "C" {

	__declspec(dllexport) void *createPass(const char *s);
	__declspec(dllexport) void init(void *inau);
	__declspec(dllexport) char *getClassName();
}

#endif //DEPTHMAPPASS_H

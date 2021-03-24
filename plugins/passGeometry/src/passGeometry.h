#ifndef PASSGEOMETRY_H
#define PASSGEOMETRY_H


#include "iNau.h"
#include "nau/render/pass.h"
#include "nau/scene/scene.h"

class PassGeometry : public Pass
		{
		protected:

			bool m_Inited;

			void prepareGeometry();

		public:

			static Pass *Create(const std::string &passName);
			PassGeometry(const std::string &name);
			~PassGeometry(void);

			virtual void prepare (void);
			virtual void doPass (void);
			virtual void restore (void);


};

extern "C" {
#ifdef WIN32
	__declspec(dllexport) void *createPass(const char *s);
	__declspec(dllexport) void init(void *inau);
	__declspec(dllexport) char *getClassName();
#else
	void *createPass(const char *s);
	void init(void *inau);
	char *getClassName();
#endif	
}

#endif //PASSGEOMETRY_H

#ifndef PASSFACTORY_H
#define PASSFACTORY_H

#include "nau/render/pass.h"

#include <map>
#include <string>
#include <vector>

namespace nau
{

#define PASSFACTORY nau::render::PassFactory::GetInstance()

	namespace render
	{
		class PassFactory
		{
		public:
			static PassFactory* GetInstance (void);

			Pass* create (const std::string &type, const std::string &name);
			bool isClass(const std::string &name);
			std::vector<std::string> *getClassNames();
			void registerClass(const std::string &type, Pass * (*callback)(const std::string &));

		protected:
			PassFactory(void) ;
			~PassFactory(void);

			std::map<std::string, void *> m_Creator;
			//std::vector<void *> CreatorV;
			//Pass * (*callback)(const std::string &);

			static PassFactory *Instance;
		};
	};
};
#endif //PASSFACTORY_H

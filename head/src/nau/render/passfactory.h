#ifndef PASSFACTORY_H
#define PASSFACTORY_H

#include <nau/render/pass.h>

namespace nau
{
	namespace render
	{
		class PassFactory
		{
		public:
			static Pass* create (const std::string &type, const std::string &name);
			static bool isClass(const std::string &name);
			static std::vector<std::string> *getClassNames();

		private:
			PassFactory(void);
			~PassFactory(void);
		};
	};
};
#endif //PASSFACTORY_H

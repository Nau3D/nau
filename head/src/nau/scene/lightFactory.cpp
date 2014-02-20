// Marta
#include <nau/scene/lightFactory.h>
#include <nau/scene/lightWithSwitch.h>


using namespace nau::scene;


Light*
LightFactory::create( std::string lName, std::string lType)
{
	if ("default"== lType){
		return new Light(lName);
	}

	if ("Light" == lType) {
		return new Light(lName);
	}
	
	if ("LightWithSwitch" == lType) {
		return new LightWithSwitch(lName);
	}
	return 0;
}
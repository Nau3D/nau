#include "nau/event/objectAnimation.h"
#include "nau/event/eventFactory.h"
#include "nau.h"

ObjectAnimation::ObjectAnimation(std::string name, SceneObject *object)
{
	this->name=name;
	this->object=object;
	object->_getTransformPtr()->setIdentity();
	this->addAnimationListener();
}

ObjectAnimation::ObjectAnimation(void)
{
	name="";
	object=0;
}

ObjectAnimation::ObjectAnimation(const ObjectAnimation &c)
{
	name=c.name;
	object=c.object;
}

ObjectAnimation::~ObjectAnimation(void)
{
}

std::string &ObjectAnimation::getName(void){
	return name;
}

void ObjectAnimation::removeAnimationListener(void){
	nau::event_::IListener *lst;
	lst = this;
	EVENTMANAGER->removeListener("OBJECT_FRACTION",lst);
}

void ObjectAnimation::addAnimationListener(void){
	nau::event_::IListener *lst;
	lst = this;
	EVENTMANAGER->addListener("OBJECT_FRACTION",lst);
}

SceneObject *ObjectAnimation::getObject(void){
	return object;
}

void ObjectAnimation::eventReceived(const std::string &sender, 
	const std::string &eventType, const std::shared_ptr<IEventData> &evt){
		
	vec3 *f=(vec3 *)evt->getData();
	vec3 fc=*f;

	if(object!=0){
		object->_getTransformPtr()->translate (fc);
	}	
	std::shared_ptr<IEventData> e = EventFactory::Create("Vec3");
	e->setData(&fc);
	EVENTMANAGER->notifyEvent("OBJECT_POSITION", &name[0], "", e);
}


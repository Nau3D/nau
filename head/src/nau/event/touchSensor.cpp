#include "nau/event/touchSensor.h"

#include "nau.h"
#include "nau/event/eventFactory.h"

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <time.h>


using namespace nau::event_;
using namespace nau::render;
using namespace nau;

TouchSensor::TouchSensor(std::string name, ISceneObject *object, bool enabled, vec3 colorSensitive){
	this->name=name;
	this->object=object;
	this->enabled=enabled;
	this->colorSensitive=colorSensitive;
	this->addTouchListener();

}

TouchSensor::TouchSensor(const TouchSensor &c){
	name=c.name;
	object=c.object;
	enabled=c.enabled;
	colorSensitive=c.colorSensitive;
}

TouchSensor::TouchSensor(void){
	name="";
	object=0;
	enabled=false;
	colorSensitive.set(0,0,0);
}

TouchSensor::~TouchSensor(void){
	this->removeTouchListener();
}

void TouchSensor::setTouchSensor(std::string name, ISceneObject *object, bool enabled, vec3 colorSensitive){
	this->name=name;
	this->object=object;
	this->enabled=enabled;
	this->colorSensitive=colorSensitive;
	this->addTouchListener();
}
std::string &TouchSensor::getName(void){
	return name;
}

ISceneObject *TouchSensor::getObject(void){
	return object;
}

bool TouchSensor::getEnabled(void){
	return enabled;
}

vec3 TouchSensor::getColorSensitive(void){
	return colorSensitive;
}

bool TouchSensor::isSelected(int x, int y)
{

	RENDERMANAGER->getRenderer()->setMatrixMode (IRenderer::PROJECTION);
	RENDERMANAGER->getRenderer()->push();
	RENDERMANAGER->getRenderer()->loadIdentity();

	RENDERMANAGER->getRenderer()->setMatrixMode (IRenderer::MODELVIEW);
	RENDERMANAGER->getRenderer()->push();
	RENDERMANAGER->getRenderer()->loadIdentity();

	//RENDERMANAGER->setCamera (aCamera);
	RENDERMANAGER->getRenderer()->clear (IRenderer::COLORBUFFER | IRenderer::DEPTHBUFFER);

	RENDERMANAGER->getRenderer()->disableSurfaceShaders();
	RENDERMANAGER->getRenderer()->deactivateLighting();
	RENDERMANAGER->getRenderer()->disableTexturing();

	//std::vector<nau::scene::ISceneObject*>::iterator objIter;

	//objIter = objects.begin();

	//int i, j;

	//for (i = 10; objIter != objects.end(); objIter++, i+=10) { // não percebi !!!
		RENDERMANAGER->getRenderer()->setColor (10 / 255.0f, 0.0f, 0.0f, 1.0f);
		//RENDERMANAGER->getRenderer()->renderObject (object, VertexData::DRAW_VERTICES); // *(*objects) está comentada a função
	//}

	RENDERMANAGER->getRenderer()->setColor (1.0f, 1.0f, 1.0f, 1.0f);
	
	RENDERMANAGER->getRenderer()->enableSurfaceShaders();

	RENDERMANAGER->getRenderer()->setMatrixMode (IRenderer::PROJECTION);
	RENDERMANAGER->getRenderer()->pop();

	RENDERMANAGER->getRenderer()->setMatrixMode (IRenderer::MODELVIEW);
	RENDERMANAGER->getRenderer()->pop();

	vec3 result = RENDERMANAGER->getRenderer()->readpixel (x, RENDERMANAGER->getCamera ("MainCamera")->getViewport().getSize().y - y);
	
	// não percebi !!!
	//for (i = 10, j = 0; j < objects.size(); i+=10, j++){
	//	if (result.x == i) {
	//		return j;
	//	}
	//}

	if(result.x==10)
		return true;
	else
		return false;		
}

void TouchSensor::addTouchListener(void){
	nau::event_::Listener *lst;
	lst = this;
	EVENTMANAGER->addListener("TOUCH",lst);
}

void TouchSensor::removeTouchListener(void){
	nau::event_::Listener *lst;
	lst = this;
	EVENTMANAGER->removeListener("TOUCH",lst);
}

void TouchSensor::eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt){

	//if(strcmp(evt->getReceiver(), &name[0])!=0 && strcmp(evt->getReceiver(), "")!=0) return;

	vec3 *ve=(vec3 *)evt->getData();
	vec3 v=*ve;
	
	if(isSelected(v.x, v.y)){
		if (enabled==false){
			char s[30];
			time_t t;
			t=time(NULL);
			itoa(t,s,10);
			nau::event_::IEventData *e= nau::event_::EventFactory::create("String");
			e->setData(s);
			EVENTMANAGER->notifyEvent("PICKED_OBJECT", &name[0], "", e);
			enabled=true;
		}
	}else{
		if (enabled==true){
			enabled=false;
		}
	}
}

void TouchSensor::init(std::string name, bool enabled, std::string str){
	this->name=name;
	this->enabled=enabled;

	std::string sub;
	float v[3];
	char *c;	
	int pos=0;
	int pos1,pos2;

	pos1=str.find("colorSensitive:",pos);
	if(pos1!=-1){
		pos1=pos1+15;
		for(int i=0; i<3;i++){
			pos2=str.find(",",pos1);
			sub=str.substr(pos1,pos2-pos1);
			c=&sub[0];
			v[i]=atof(c);
			pos1=pos2+1;
		}
		this->colorSensitive.set(v[0],v[1],v[2]);
	}
	
	std::string sc;

	pos1=str.find("scene:",pos);
	if(pos1!=-1){
		pos1=pos1+6;
		pos2=str.find(";",pos1);
		sub=str.substr(pos1,pos2-pos1);
		sc=sub;
	}

	pos1=str.find("object:",pos);
	if(pos1!=-1){
		pos1=pos1+7;
		pos2=str.find(";",pos1);
		sub=str.substr(pos1,pos2-pos1);	
		this->object=RENDERMANAGER->getScene(sc)->getSceneObject(sub); // substituir o MainScene, pode ter mais do q uma Cena?
	}

	this->addTouchListener();
}
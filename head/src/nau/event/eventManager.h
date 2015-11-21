#ifndef EVENTMANAGER_H
#define EVENTMANAGER_H

#include "nau/event/listenerType.h"
#include "nau/event/Ilistener.h"
#include "nau/event/sensor.h"
#include "nau/event/iInterpolator.h"
#include "nau/event/route.h"

#include <memory>
#include <map>
#include <vector>

using namespace std;
using namespace nau::event_;

namespace nau
{
	namespace event_
	{
		class EventManager
		{
		private:
			std::map<std::string, nau::event_::Sensor*> m_Sensors;
			std::map<std::string, nau::event_::Interpolator*> m_Interpolators;
			std::map<std::string, nau::event_::Route*> m_Routes;

			void notify(std::string eventType, std::string sender, std::string receiver, 
				const std::shared_ptr<IEventData> &evt, vector<IListener *> lsts);

		public:

			// shouldn't this be private?
			vector<ListenerType *> listeners;
			

			EventManager(vector<ListenerType *> *listeners);
			EventManager(const EventManager &e);
			EventManager(void);
			~EventManager(void);


			void setListeners(vector<ListenerType *> *listeners);
			vector<ListenerType *> *getAllListeners(void);
			vector<IListener*> *getListeners(std::string eventType);

			
			void addListener(std::string eventType, IListener *lst);
			void removeListener(std::string eventType, IListener *lst);
			void eraseAllListenersType(std::string eventType);
			void eraseAllListeners(void);
			void clear(void);

			void notifyEvent(std::string eventType, std::string sender, std::string receiver, 
				const std::shared_ptr<IEventData> &evt);

			bool hasSensor (std::string sensorName);
			Sensor* getSensor (std::string sensorName, std::string sClass);

			bool hasInterpolator (std::string interpolatorName);
			Interpolator* getInterpolator (std::string interpolatorName, std::string sClass);

			bool hasRoute (std::string routeName);
			Route* getRoute (std::string routeName);

		};
	};
};

#endif
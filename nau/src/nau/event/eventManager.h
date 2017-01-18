#ifndef EVENTMANAGER_H
#define EVENTMANAGER_H

#include "nau/event/listenerType.h"
#include "nau/event/iListener.h"
#include "nau/event/sensor.h"
#include "nau/event/iInterpolator.h"
#include "nau/event/route.h"

#include <memory>
#include <map>
#include <vector>

using namespace std;
using namespace nau::event_;

#define EVENTMANAGER EventManager::GetInstance()

#ifdef _WINDLL
#ifdef nau_EXPORTS
#define nau_API __declspec(dllexport)   
#else  
#define nau_API __declspec(dllimport)   
#endif 
#else
#define nau_API
#endif


namespace nau
{
	namespace event_
	{
		class EventManager
		{
		protected:
			std::map<std::string, nau::event_::Sensor*> m_Sensors;
			std::map<std::string, nau::event_::Interpolator*> m_Interpolators;
			std::map<std::string, nau::event_::Route*> m_Routes;

			void notify(std::string eventType, std::string sender, std::string receiver, 
				const std::shared_ptr<IEventData> &evt, vector<IListener *> lsts);

			static EventManager *Instance;

			EventManager(vector<ListenerType *> *listeners);
			EventManager(const EventManager &e);
			EventManager(void);

		public:

			~EventManager(void);

			static nau_API EventManager *GetInstance();

			// shouldn't this be private?
			vector<ListenerType *> listeners;
			



			void setListeners(vector<ListenerType *> *listeners);
			nau_API vector<ListenerType *> *getAllListeners(void);
			nau_API vector<IListener*> *getListeners(std::string eventType);

			
			nau_API void addListener(std::string eventType, IListener *lst);
			nau_API void removeListener(std::string eventType, IListener *lst);
			nau_API void eraseAllListenersType(std::string eventType);
			nau_API void eraseAllListeners(void);
			nau_API void clear(void);

			nau_API void notifyEvent(std::string eventType, std::string sender, std::string receiver,
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

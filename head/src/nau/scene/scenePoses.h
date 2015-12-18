#ifndef SCENEPOSES_H
#define SCENEPOSES_H

#include "nau/scene/scene.h"
#include "nau/geometry/meshPoseAnim.h"

using namespace nau::geometry;


namespace nau {

	namespace scene {

		class ScenePoses : public Scene
		{
			friend class SceneFactory;

			protected:
				ScenePoses(void);

			public:
				~ScenePoses(void);

				virtual void compile();
			
				virtual void eventReceived(const std::string &sender, const std::string &eventType, 
					const std::shared_ptr<IEventData> &evt);

				void setActiveAnim(std::string aName);
				std::string getActiveAnim();
				void addAnim(std::string aName, float aLength);
				MeshPoseAnim &getAnim(std::string aName);
				void addAnimTrack(std::string aName, unsigned int aTrack);

			private:
				void setPose(int index);
				void setPose(std::string name);
				void setRelativeTime(std::string aAnim, float time);


				std::map<std::string, MeshPoseAnim> m_Anims;
				std::string m_ActiveAnim;

		};
	};
};



#endif

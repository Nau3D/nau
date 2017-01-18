#ifndef SCENESKELETON_H
#define SCENESKELETON_H

#include "nau/scene/scene.h"
#include "nau/geometry/skeletonAnim.h"
#include "nau/geometry/skeletonBone.h"

using namespace nau::geometry;

namespace nau {

	namespace scene {

		class SceneSkeleton : public Scene
		{
			friend class SceneFactory;

			protected:
				SceneSkeleton(void);

			public:
				~SceneSkeleton(void);

				virtual void compile();
			
				virtual void eventReceived(const std::string &sender, const std::string &eventType, 
					const std::shared_ptr<IEventData> &evt);

				void addAnim(std::string aName, float aLength);
				SkeletonAnim &getAnim(std::string aName);
				void addAnimTrack(std::string aName, unsigned int aTrack);

				void addBone(std::string name, unsigned int id, vec3 pos, float angle, vec3 axis);
				void setBoneRelation(std::string child, std::string parent);

				int seekBoneID(std::string);

			private:

				void setRelativeTime(std::string aAnim, float time);

				std::map<std::string, SkeletonAnim> m_Anims;

				std::map<unsigned int, SkeletonBone> m_Bones;

				// map that relates a bone (id) with its parent (id)
				// if a bone is not present in the map, then it has no parents
				std::map<unsigned int, int> m_BoneHierarchy;
		};
	};
};


#endif

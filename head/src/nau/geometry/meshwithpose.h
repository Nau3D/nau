#ifndef MESHWITHPOSE_H
#define MESHWITHPOSE_H

#include <nau/geometry/mesh.h>
#include <nau/geometry/poseoffset.h>

using namespace nau::math;
using namespace nau::geometry;

namespace nau
{
	namespace geometry 
	{
		class MeshPose : public Mesh 
		{
			protected:
				std::vector<PoseOffset *> m_vOffsets;
				std::vector<std::string> m_vNames;

				std::vector<vec3> m_ReferencePose;

				unsigned int m_ActivePose;

				MeshPose(void);

			public:

				friend class nau::resource::ResourceManager;
				~MeshPose(void);

				void addPose(std::string aName, PoseOffset *aPose);
				void setPose(unsigned int index);
				void setPose(std::string aPoseName);
				unsigned int getActivePose();

				unsigned int getNumberOfPoses();

				void setPose(std::map<unsigned int , float > *influences);

				void setReferencePose(std::vector<vec4> vertexData);
				void setReferencePose();

				virtual std::string getType (void);
				void eventReceived(const std::string &sender, const std::string &eventType, 
					nau::event_::IEventData *evt);
		};
	};

};

#endif


#ifndef CAMERA_H
#define CAMERA_H

#include <string>

#include <nau/scene/sceneobject.h>
#include <nau/math/simpletransform.h>
#include <nau/math/vec4.h>
#include <nau/math/spherical.h>
#include <nau/render/viewport.h>
#include <nau/enums.h>


#include <nau/event/EventVec3.h>
#include <nau/attribute.h>

using namespace nau::math;
using namespace nau::scene;


namespace nau
{
	namespace scene
	{

		class Camera : public SceneObject
		{
		public:

			typedef enum {
				ORTHO,
				PERSPECTIVE
			} CameraType;

			typedef enum { POSITION, VIEW_VEC, NORMALIZED_VIEW_VEC, 
				UP_VEC, NORMALIZED_UP_VEC, 
				NORMALIZED_RIGHT_VEC, LOOK_AT_POINT,
				COUNT_FLOAT4PROPERTY} Float4Property;

			typedef enum { FOV, NEARP, FARP, LEFT, RIGHT, TOP, BOTTOM, 
				ELEVATION_ANGLE, ZX_ANGLE, 
				COUNT_FLOATPROPERTY} FloatProperty;

			typedef enum { VIEW_MATRIX, PROJECTION_MATRIX, VIEW_INVERSE_MATRIX,
				PROJECTION_VIEW_MATRIX, TS05_PVM_MATRIX, COUNT_MAT4PROPERTY } Mat4Property;

			typedef enum { COUNT_INTPROPERTY} IntProperty;

			typedef enum { PROJECTION_TYPE, COUNT_ENUMPROPERTY} EnumProperty;


			static AttribSet Attribs;

			Camera (const std::string &name);
			virtual ~Camera (void);


			void setProp(Float4Property prop, float r, float g, float b, float a);
			void setProp(FloatProperty prop, float value);
			void setProp(EnumProperty prop, int value);
			// Note: no validation is performed!
			void setProp(int prop, Enums::DataType type, void *value);

			float getPropf(FloatProperty prop);
			const mat4 &getPropm4(Mat4Property prop);
			const vec4 &getPropf4(Float4Property prop);
			int getPrope(EnumProperty prop);
			void *getProp(int prop, Enums::DataType type);

			void setOrtho (float left, float right, float bottom, float top, float near, float far);
			void setPerspective (float fov, float near, float far);

			void eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt);

			void setCamera (vec3 position, vec3 view, vec3 up);

			// Viewport
			nau::render::Viewport* getViewport (void);
			void setViewport (nau::render::Viewport* aViewport);

			// Adjusts the frustum of the current Camera to include
			// the frustum of the target camera
			// usefull for shadow mapping for instance
			void adjustMatrix (nau::scene::Camera *targetCamera);
			// This version considers only a fraction of the target camera
			// the params near and far relate to the targets camera frustum
			void adjustMatrixPlus (float cNear, float cFar, nau::scene::Camera *targetCamera);

			// Bounding Volume 	
			virtual const nau::geometry::IBoundingVolume* getBoundingVolume();

			// Renderable is the graphic representation of the camera
			// usefull for debug purposes
			nau::render::IRenderable& getRenderable();

			// used for Physics
			bool isDynamic();
			void setDynamic(bool value);
			void setPositionOffset (float value);

			//std::string& getName (void);
			//bool getLookAt();
			//void setLookAt(bool flag);
			// Spherical Coordinates
			//void setElevationAngle(float angle);
			//void setZXAngle(float angle);
			//float getElevationAngle();
			//float getZXAngle();
			// Projections
			//void setProjectionType(CameraType ct);
			//unsigned int getProjectionType();


		protected:

			static bool Init();
			static bool Inited;

			void setDefault();

			std::map<int, int> m_IntProps;
			std::map<int,vec4> m_Float4Props;
			std::map<int,float> m_FloatProps;
			std::map<int,SimpleTransform> m_Mat4Props;
			std::map<int, int> m_EnumProps;

			nau::event_::EventVec3 m_Event;
			nau::render::Viewport *m_pViewport;

			vec3 result;
			// LookAt settings
			bool m_LookAt;

			// Camera Spherical Coordinates (angles are stored in radians)
			// Spherical(0,0) means Cartesian(0,0,1)
			void setVectorsFromSpherical();

			// Physics
			float m_PositionOffset;
			bool m_IsDynamic; 

			// Projections
			//bool m_IsOrtho;

			void updateProjection();

			void setProp(Mat4Property prop, mat4 &mat);
			// Matrices
			void buildViewMatrix (void);
			void buildViewMatrixInverse(void);
			void buildProjectionMatrix();
			void buildProjectionViewMatrix(void);
			void buildTS05PVMMatrix(void);


			// The eight corners of the frustum
			enum {
				TOP_LEFT_NEAR = 0,
				TOP_RIGHT_NEAR,
				BOTTOM_RIGHT_NEAR,
				BOTTOM_LEFT_NEAR,
				TOP_LEFT_FAR,
				TOP_RIGHT_FAR,
				BOTTOM_RIGHT_FAR,
				BOTTOM_LEFT_FAR,
			};		

			//AttribSetFloat mFloatAttribs;
			//AttribSetVec4 mVec4Attribs;
			//vec3 m_LookAtPoint;

		};
	};
};

#endif //CAMERA_H

#ifndef ISCENE_H
#define ISCENE_H

#include <string>
#include <vector>

#include <nau/scene/camera.h>
#include <nau/scene/light.h>
#include <nau/geometry/frustum.h>
#include <nau/event/ilistener.h>
#include <nau/math/transformfactory.h>

namespace nau 
{
	namespace scene 
	{
		class SceneObject;

		class IScene : public IListener
		{

		protected:
			std::string m_Name;
			bool m_Compiled;
			ITransform *m_Transform;
			bool m_Visible;

		public:
			virtual void setName(std::string name) {
				m_Name = name; 
			};
			virtual std::string &getName() {
				return m_Name;
			};
		  
			virtual void add (SceneObject *aSceneObject) = 0;

			virtual void show (void) {m_Visible = true; }			
			virtual void hide (void) {m_Visible = false; }
			virtual bool isVisible (void) {return m_Visible; }

			virtual std::vector <SceneObject*>& findVisibleSceneObjects 
																(nau::geometry::Frustum &aFrustum, 
																Camera &aCamera, 
																bool conservative = false) = 0;
			virtual std::vector<SceneObject*>& getAllObjects (void) = 0;
			virtual SceneObject* getSceneObject (std::string name) = 0; 
			virtual SceneObject* getSceneObject (int index) = 0;

			virtual void getMaterialNames(std::set<std::string> *nameList) = 0;

			virtual void build (void) = 0;

			virtual void compile (void) = 0;
			bool isCompiled() { return m_Compiled;}

			virtual void unitize() = 0;

			virtual nau::math::ITransform *getTransform() = 0;
			//virtual void scale(float factor) = 0;
			//virtual void translate(float x, float y, float z) = 0;
			//virtual void rotate(float ang, float ax, float ay, float az) = 0;
			virtual void setTransform(nau::math::ITransform *t) = 0;
			virtual void transform(nau::math::ITransform *t) = 0;

			virtual nau::geometry::IBoundingVolume& getBoundingVolume (void) = 0;

			//virtual void show (void) = 0;
			//virtual void hide (void) = 0;
			//virtual bool isVisible (void) = 0;

			virtual std::string getType (void) = 0;

			virtual ~IScene(void) {};
			IScene(void): m_Compiled(false), m_Visible(true) {};

		};
	};
};

#endif

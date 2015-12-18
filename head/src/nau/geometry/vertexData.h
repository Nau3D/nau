#ifndef VERTEXDATA_H
#define VERTEXDATA_H

#include "nau/math/vec3.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace nau::math;

namespace nau
{
	namespace geometry
	{

		class VertexAttrib {

		public:
			float x,y,z,w;

			VertexAttrib() : x(0), y(0), z(0), w(0) {};
			VertexAttrib(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {};
			VertexAttrib(const VertexAttrib &v) : x(v.x), y(v.y), z(v.z), w(v.w) {};

			void
			set(float xx, float yy, float zz, float ww = 1) {

				this->x = xx;
				this->y = yy;
				this->z = zz;
				this->w = ww;
			}

			void 
			add(const VertexAttrib &v) {
				x += v.x;
				y += v.y;
				z += v.z;
				w += v.w;
			};

			bool
			operator == (const VertexAttrib &v) const {
				float tolerance = -1.0f;
				return (FloatEqual(x, v.x, tolerance) && FloatEqual(y, v.y, tolerance) && \
						FloatEqual(z, v.z, tolerance) && FloatEqual(w, v.w, tolerance));
			};

			void
			copy(const VertexAttrib &v) {

				x = v.x;
				y = v.y;
				z = v.z;
				w = v.w;
			};

			void
			normalize() {

				float m = sqrtf(x*x + y*y + z*z);
				if (m <= FLT_EPSILON) {
					m = 1;
				}
				x /= m;
				y /= m;
				z /= m;
			};

		};


		class VertexData
		{
		public:

			static const int MaxAttribs = 16;
			static  const std::string Syntax[]; 
			static unsigned int GetAttribIndex(std::string &);
			// A vertex attribute is four floats
			typedef VertexAttrib Attr ;
			static unsigned int const NOLOC = 0;

			static std::shared_ptr<VertexData> Create (const std::string &name);

			virtual ~VertexData(void);

			void setName(std::string &name);

			virtual unsigned int getNumberOfVertices() = 0;

			std::shared_ptr<std::vector<Attr>> &getDataOf (unsigned int type);
			void setDataFor (unsigned int index, 
				std::shared_ptr<std::vector<Attr>> &dataArray);

			int add (std::shared_ptr<VertexData> &aVertexData);

			virtual void prepareTriangleIDs(unsigned int sceneObjID, 
				                                       unsigned int primitiveOffset, 
													   std::vector<unsigned int> *index) = 0;
			virtual void appendVertex(unsigned int i) = 0;

			virtual void setAttributeDataFor (unsigned int type, 
				std::shared_ptr<std::vector<VertexData::Attr>> &,
				int location = -1) = 0;
			virtual void setAttributeLocationFor (unsigned int type, int location) = 0;
			void unitize(vec3 &vCenter, vec3 &vMin, vec3 &vMax);
			virtual bool compile(void) = 0;
			virtual unsigned int getBufferID(unsigned int vertexAttrib) = 0;

			virtual void resetCompilationFlag() = 0;
			virtual void bind (void) = 0;
			virtual void unbind (void) = 0;
			virtual bool isCompiled() = 0;

			virtual void setBuffer(unsigned int type, int bufferID) = 0;

		protected:
			VertexData(void);
			VertexData(const std::string &name);
			
			std::shared_ptr<std::vector<Attr>> m_InternalArrays[MaxAttribs];
			std::string m_Name;
		};
	};
};




#endif //VERTEXDATA_H

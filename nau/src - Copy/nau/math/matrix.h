#ifndef NAU_MATRIX_H
#define NAU_MATRIX_H

#include "nau/math/data.h"
#include "nau/math/vec3.h"
#include "nau/math/vec4.h"
#include "nau/math/utils.h"

#include <assert.h>
#include <cstdlib>
#include <string>
#include <string.h>

namespace nau
{
	namespace math
	{

		// ---------------------------------------------------------
		//		SQUARE MATRICES
		//----------------------------------------------------------

		template <typename T, int DIMENSION>
		class matrix: public Data{

		protected:

		public:
			T m_Matrix[DIMENSION*DIMENSION];

			matrix() {
				setIdentity();
			};

			matrix(const T* values) {

				setMatrix(values);
			}

			~matrix() {};

			Data *clone() {
				matrix<T, DIMENSION> *m = new(matrix<T, DIMENSION>);
				memcpy(m->m_Matrix, m_Matrix, DIMENSION*DIMENSION*sizeof(T));
				return m;
			}


			void *getPtr() {
				return m_Matrix;
			}

			T at(unsigned int i, unsigned int j) const {
				assert(i < DIMENSION && j < DIMENSION);
				return m_Matrix[i * DIMENSION + j];
			}

			void set(unsigned int i, unsigned int j, T value) {
				assert(i < DIMENSION && j < DIMENSION);
				m_Matrix[i * DIMENSION + j] = value;
			}

			const T* getMatrix() const{
				return m_Matrix;
			}

			void setMatrix(const T* values) {

				for (int i = 0; i < DIMENSION*DIMENSION; ++i) {
					m_Matrix[i] = values[i];
				}
			}

			void setIdentity() {
				for (int i = 0; i < DIMENSION; ++i) {
					for (int j = 0; j < DIMENSION; ++j) {
						m_Matrix[i * DIMENSION + j] = (i == j) ? (T)1 : (T)0;
					}
				}
			}

			// copy into this
			void copy(const matrix &m) {
				setMatrix(m.m_Matrix);
			}



			void transpose() {

				T aux;
				for (int i = 0; i < DIMENSION; ++i) {
					for (int j = i + 1; j < DIMENSION; ++j) {
						aux = m_Matrix[i * DIMENSION + j];
						m_Matrix[i * DIMENSION + j] = m_Matrix[j * DIMENSION + i];
						m_Matrix[j * DIMENSION + i] = aux;
					}
				}
			}

			matrix & operator += (const matrix &m) {

				for (int i = 0; i < DIMENSION*DIMENSION; ++i) {
					this->m_Matrix[i] += m.m_Matrix[i];
				}

				return (*this);
			}

			// Subtract a mat4 from this one
			matrix & operator -= (const matrix &m) {

				for (int i = 0; i < DIMENSION*DIMENSION; ++i) {
					this->m_Matrix[i] -= m.m_Matrix[i];
				}

				return (*this);
			}

			matrix & operator *= (const matrix &m) {

				this->multiply(m);

				return (*this);
			}

			matrix & operator *= (T f) {

				for (int i = 0; i < DIMENSION*DIMENSION; ++i) {
					this->m_Matrix[i] *= f;
				}

				return (*this);
			}

			// M1 = M1 * M2;
			// assumes column major matrices
			void multiply(const matrix &m) {

				T aux[DIMENSION][DIMENSION];

				const T *m1 = this->m_Matrix;
				const T *m2 = m.m_Matrix;

				for (int i = 0; i < DIMENSION; ++i) {
					for (int j = 0; j < DIMENSION; ++j) {
						aux[i][j] = (T)0;
						for (int k = 0; k < DIMENSION; ++k) {

							aux[i][j] += at(k, j) * m.at(i, k);
						}
					}
				}
				this->setMatrix((float *)aux);
			}
			//protected:

			matrix(const matrix &m) {

				setMatrix(m.m_Matrix);
			}

			const matrix &operator = (const matrix &m) {

				setMatrix(m.m_Matrix);
				return this;
			}


			std::string toString() {

				std::string s;

				for (int i = 0; i < DIMENSION; ++i) {
					s = s + std::to_string(m_Matrix[i*DIMENSION]);
					for (int j = 1; j < DIMENSION; ++j) {
						s = s + ", " + std::to_string(m_Matrix[i*DIMENSION + j]);
					}
					if (i != DIMENSION -1)
						s = s + "\n";
				}
				return "[ " + s + " ]";
			}
		};

		// ---------------------------------------------------------
		//		MATRIX 3x3
		//----------------------------------------------------------
#define M3(a,b) (a*3 + b)

		template <typename T>
		class matrix3 : public matrix < T, 3 > {

		public:
			matrix3() : matrix<T,3>() {}

			matrix3(const T *mat) : matrix<T,3>(mat) {}

			matrix3 *clone() {

				return new matrix3(this->m_Matrix);
			}

			void invert() {
				T m[9], det, invDet;
				T* mm = this->m_Matrix;
				det = mm[M3(0, 0)] * (mm[M3(2, 2)] * mm[M3(1, 1)] - mm[M3(2, 1)] * mm[M3(1, 2)]) -
					mm[M3(1, 0)] * (mm[M3(2, 2)] * mm[M3(0, 1)] - mm[M3(2, 1)] * mm[M3(0, 2)]) +
					mm[M3(2, 0)] * (mm[M3(1, 2)] * mm[M3(0, 1)] - mm[M3(1, 1)] * mm[M3(0, 2)]);

				if (FloatEqual(det, 0.0))
					return;

				invDet = 1.0f / det;

				m[M3(0, 0)] = (mm[M3(2, 2)] * mm[M3(1, 1)] - mm[M3(2, 1)] * mm[M3(1, 2)]) / invDet;
				m[M3(0, 1)] = -(mm[M3(2, 2)] * mm[M3(0, 1)] - mm[M3(2, 1)] * mm[M3(0, 2)]) / invDet;
				m[M3(0, 2)] = (mm[M3(1, 2)] * mm[M3(0, 1)] - mm[M3(1, 1)] * mm[M3(0, 2)]) / invDet;

				m[M3(1, 0)] = -(mm[M3(2, 2)] * mm[M3(1, 0)] - mm[M3(2, 0)] * mm[M3(1, 2)]) / invDet;
				m[M3(1, 1)] = (mm[M3(2, 2)] * mm[M3(0, 0)] - mm[M3(2, 0)] * mm[M3(0, 2)]) / invDet;
				m[M3(1, 2)] = -(mm[M3(1, 2)] * mm[M3(0, 0)] - mm[M3(1, 0)] * mm[M3(0, 2)]) / invDet;

				m[M3(2, 0)] = (mm[M3(2, 1)] * mm[M3(1, 0)] - mm[M3(2, 0)] * mm[M3(1, 1)]) / invDet;
				m[M3(2, 1)] = -(mm[M3(2, 1)] * mm[M3(0, 0)] - mm[M3(2, 0)] * mm[M3(0, 1)]) / invDet;
				m[M3(2, 2)] = (mm[M3(1, 1)] * mm[M3(0, 0)] - mm[M3(1, 0)] * mm[M3(0, 1)]) / invDet;

				this->setMatrix(m);
			}

			const matrix3 &operator = (const matrix3 &m) {

				this->setMatrix(m.m_Matrix);
				return *this;
			}
		};

		// ---------------------------------------------------------
		//		MATRIX 2x2
		//----------------------------------------------------------

		template <typename T>
		class matrix2 : public matrix < T, 2 > {

		public:
			matrix2() : matrix< T, 2 >() {}

			matrix2(const T *mat) : matrix< T, 2 >(mat) {}

			matrix2 *clone() {

				return new matrix2(this->m_Matrix);
			}

			void invert() {

				T* mm = this->m_Matrix;
				T aux;

				float det = mm[0] * mm[3] - mm[1] * mm[2];
				if (FloatEqual(det, 0.0))
					return;

				float detInv = 1 / det;

				aux = mm[0];
				mm[0] = mm[3] * detInv;
				mm[3] = mm[0] * detInv;

				aux = mm[1];
				mm[1] = -mm[2] * detInv;
				mm[2] = -aux * detInv;
			}

			const matrix2 &operator = (const matrix2 &m) {

				this->setMatrix(m.m_Matrix);
				return *this;
			}
		};

		// ---------------------------------------------------------
		//		MATRIX 4x4
		//----------------------------------------------------------

		template <typename T>
		class matrix4 : public matrix < T, 4 > {

		public:
			matrix4() : matrix< T, 4 >() {}

			matrix4(const T *mat) : matrix< T, 4 >(mat) {}

			matrix4 *clone() {
				return new matrix4(this->m_Matrix);
			}

			const T * getSubMat3() {

				T* mm = this->m_Matrix;
				m_SubMat3[0] = mm[0];
				m_SubMat3[1] = mm[1];
				m_SubMat3[2] = mm[2];

				m_SubMat3[3] = mm[4];
				m_SubMat3[4] = mm[5];
				m_SubMat3[5] = mm[6];

				m_SubMat3[6] = mm[8];
				m_SubMat3[7] = mm[9];
				m_SubMat3[8] = mm[10];

				return (this->m_SubMat3);
			}


			// post multiply V' = M * V  
			// works in homogeneous coordinates - divides result by w
			void transform(vec4 &v) const {

				vector4<T> aux;
				const T *m = this->m_Matrix;

				aux.x = (v.x * m[0]) + (v.y * m[4]) + (v.z * m[8]) + (v.w * m[12]);
				aux.y = (v.x * m[1]) + (v.y * m[5]) + (v.z * m[9]) + (v.w * m[13]);
				aux.z = (v.x * m[2]) + (v.y * m[6]) + (v.z * m[10]) + (v.w * m[14]);
				aux.w = (v.x * m[3]) + (v.y * m[7]) + (v.z * m[11]) + (v.w * m[15]);
				if (!FloatEqual(aux.w, 0.0))
					aux *= 1 / aux.w;

				v.copy(aux);
			}

			void transform(float *v) const {

				float aux[4];
				const T *m = this->m_Matrix;

				aux[0] = (v[0] * m[0]) + (v[1] * m[4]) + (v[2] * m[8])  + (v[3] * m[12]);
				aux[1] = (v[0] * m[1]) + (v[1] * m[5]) + (v[2] * m[9])  + (v[3] * m[13]);
				aux[2] = (v[0] * m[2]) + (v[1] * m[6]) + (v[2] * m[10]) + (v[3] * m[14]);
				aux[3] = (v[0] * m[3]) + (v[1] * m[7]) + (v[2] * m[11]) + (v[3] * m[15]);
				if (!FloatEqual(aux[3], 0.0)) {
					float k = 1 / aux[3];
					for (int i = 0; i < 4; ++i)
						aux[i] *= k;
				}

				for (int i = 0; i < 4; ++i)
					v[i] = aux[i];
			}

			// Transform (i.e. multiply) a vector (w=1) by this matrix.
			void transform3(vec4 &v) const {

				vec4 aux;
				const T *m = this->m_Matrix;

				aux.x = (v.x * m[0]) + (v.y * m[4]) + (v.z * m[8]);
				aux.y = (v.x * m[1]) + (v.y * m[5]) + (v.z * m[9]);
				aux.z = (v.x * m[2]) + (v.y * m[6]) + (v.z * m[10]);
				aux.w = v.w;
				v.copy(aux);

				return;
			}

			void transform3(float *v) const {

				float aux[3];
				const T *m = this->m_Matrix;

				aux[0] = (v[0] * m[0]) + (v[1] * m[4]) + (v[2] * m[8]);
				aux[1] = (v[0] * m[1]) + (v[1] * m[5]) + (v[2] * m[9]);
				aux[2] = (v[0] * m[2]) + (v[1] * m[6]) + (v[2] * m[10]);

				for (int i = 0; i < 3; ++i)
					v[i] = aux[i];
				return;
			}

			void transform(vec3 &v) const {

				vec3 aux;
				const T *m = this->m_Matrix;

				aux.x = (v.x * m[0]) + (v.y * m[4]) + (v.z * m[8]) + (m[12]);
				aux.y = (v.x * m[1]) + (v.y * m[5]) + (v.z * m[9]) + (m[13]);
				aux.z = (v.x * m[2]) + (v.y * m[6]) + (v.z * m[10]) + (m[14]);

				float w = (v.x * m[3]) + (v.y * m[7]) + (v.z * m[11]) + (m[15]);
				if (!FloatEqual(w, 0.0))
					aux *= (1 / w);

				v.copy(aux);

				return;
			}

			void invert()
			{
				T *mat = this->m_Matrix;
				T dst[16];
				float    tmp[12]; /* temp array for pairs                      */
				float    src[16]; /* array of transpose source matrix */
				float    det;     /* determinant                                  */
				/* transpose matrix */
				for (int i = 0; i < 4; ++i) {
					src[i] = mat[i * 4];
					src[i + 4] = mat[i * 4 + 1];
					src[i + 8] = mat[i * 4 + 2];
					src[i + 12] = mat[i * 4 + 3];
				}
				/* calculate pairs for first 8 elements (cofactors) */
				tmp[0] = src[10] * src[15];
				tmp[1] = src[11] * src[14];
				tmp[2] = src[9] * src[15];
				tmp[3] = src[11] * src[13];
				tmp[4] = src[9] * src[14];
				tmp[5] = src[10] * src[13];
				tmp[6] = src[8] * src[15];
				tmp[7] = src[11] * src[12];
				tmp[8] = src[8] * src[14];
				tmp[9] = src[10] * src[12];
				tmp[10] = src[8] * src[13];
				tmp[11] = src[9] * src[12];

				/* calculate first 8 elements (cofactors) */
				dst[0] = tmp[0] * src[5] + tmp[3] * src[6] + tmp[4] * src[7];
				dst[0] -= tmp[1] * src[5] + tmp[2] * src[6] + tmp[5] * src[7];
				dst[1] = tmp[1] * src[4] + tmp[6] * src[6] + tmp[9] * src[7];
				dst[1] -= tmp[0] * src[4] + tmp[7] * src[6] + tmp[8] * src[7];
				dst[2] = tmp[2] * src[4] + tmp[7] * src[5] + tmp[10] * src[7];
				dst[2] -= tmp[3] * src[4] + tmp[6] * src[5] + tmp[11] * src[7];
				dst[3] = tmp[5] * src[4] + tmp[8] * src[5] + tmp[11] * src[6];
				dst[3] -= tmp[4] * src[4] + tmp[9] * src[5] + tmp[10] * src[6];
				dst[4] = tmp[1] * src[1] + tmp[2] * src[2] + tmp[5] * src[3];
				dst[4] -= tmp[0] * src[1] + tmp[3] * src[2] + tmp[4] * src[3];
				dst[5] = tmp[0] * src[0] + tmp[7] * src[2] + tmp[8] * src[3];
				dst[5] -= tmp[1] * src[0] + tmp[6] * src[2] + tmp[9] * src[3];
				dst[6] = tmp[3] * src[0] + tmp[6] * src[1] + tmp[11] * src[3];
				dst[6] -= tmp[2] * src[0] + tmp[7] * src[1] + tmp[10] * src[3];
				dst[7] = tmp[4] * src[0] + tmp[9] * src[1] + tmp[10] * src[2];
				dst[7] -= tmp[5] * src[0] + tmp[8] * src[1] + tmp[11] * src[2];

				/* calculate pairs for second 8 elements (cofactors) */
				tmp[0] = src[2] * src[7];
				tmp[1] = src[3] * src[6];
				tmp[2] = src[1] * src[7];
				tmp[3] = src[3] * src[5];
				tmp[4] = src[1] * src[6];
				tmp[5] = src[2] * src[5];
				tmp[6] = src[0] * src[7];
				tmp[7] = src[3] * src[4];
				tmp[8] = src[0] * src[6];
				tmp[9] = src[2] * src[4];
				tmp[10] = src[0] * src[5];
				tmp[11] = src[1] * src[4];

				/* calculate second 8 elements (cofactors) */
				dst[8] = tmp[0] * src[13] + tmp[3] * src[14] + tmp[4] * src[15];
				dst[8] -= tmp[1] * src[13] + tmp[2] * src[14] + tmp[5] * src[15];
				dst[9] = tmp[1] * src[12] + tmp[6] * src[14] + tmp[9] * src[15];
				dst[9] -= tmp[0] * src[12] + tmp[7] * src[14] + tmp[8] * src[15];
				dst[10] = tmp[2] * src[12] + tmp[7] * src[13] + tmp[10] * src[15];
				dst[10] -= tmp[3] * src[12] + tmp[6] * src[13] + tmp[11] * src[15];
				dst[11] = tmp[5] * src[12] + tmp[8] * src[13] + tmp[11] * src[14];
				dst[11] -= tmp[4] * src[12] + tmp[9] * src[13] + tmp[10] * src[14];
				dst[12] = tmp[2] * src[10] + tmp[5] * src[11] + tmp[1] * src[9];
				dst[12] -= tmp[4] * src[11] + tmp[0] * src[9] + tmp[3] * src[10];
				dst[13] = tmp[8] * src[11] + tmp[0] * src[8] + tmp[7] * src[10];
				dst[13] -= tmp[6] * src[10] + tmp[9] * src[11] + tmp[1] * src[8];
				dst[14] = tmp[6] * src[9] + tmp[11] * src[11] + tmp[3] * src[8];
				dst[14] -= tmp[10] * src[11] + tmp[2] * src[8] + tmp[7] * src[9];
				dst[15] = tmp[10] * src[10] + tmp[4] * src[8] + tmp[9] * src[9];
				dst[15] -= tmp[8] * src[9] + tmp[11] * src[10] + tmp[5] * src[8];

				/* calculate determinant */
				det = src[0] * dst[0] + src[1] * dst[1] + src[2] * dst[2] + src[3] * dst[3];

				if (FloatEqual(det, 0.0))
					return;

				/* calculate matrix inverse */
				det = 1 / det;
				for (int j = 0; j < 16; j++)
					dst[j] *= det;

				this->setMatrix(dst);
			}

			const matrix4 &operator = (const matrix4 &m) {

				this->setMatrix(m.m_Matrix);
				return *this;
			}

			void translate(const vec3 &v) {

				matrix4<T> translation;

				translation.set(3, 0, v.x);
				translation.set(3, 1, v.y);
				translation.set(3, 2, v.z);

				multiply(translation);
			}
			
			void translate(float x, float y, float z) {

				matrix4<T> translation;

				translation.set(3, 0, x);
				translation.set(3, 1, y);
				translation.set(3, 2, z);

				multiply(translation);
			}

			void scale(const vec3 &v) {

				matrix4<T> scale;
				scale.set(0, 0, v.x);
				scale.set(1, 1, v.y);
				scale.set(2, 2, v.z);

				multiply(scale);
			}

			void scale(float x, float y, float z) {

				matrix4<T> scale;
				scale.set(0, 0, x);
				scale.set(1, 1, y);
				scale.set(2, 2, z);

				multiply(scale);
			}

			void rotate(float angle, vec3 &axis) {

				rotate(angle, axis.x, axis.y, axis.z);
			}

			void rotate(float angle, float x, float y, float z)
			{
				matrix4<T> matrix;
				float radAngle = nau::math::DegToRad(angle);
				float co = cos(radAngle);
				float si = sin(radAngle);
				float x2 = x*x;
				float y2 = y*y;
				float z2 = z*z;

				matrix.set(0, 0, x2 + (y2 + z2) * co);
				matrix.set(1, 0, x * y * (1 - co) - z * si);
				matrix.set(2, 0, x * z * (1 - co) + y * si);
				matrix.set(3, 0, 0.0f);

				matrix.set(0, 1, x * y * (1 - co) + z * si);
				matrix.set(1, 1, y2 + (x2 + z2) * co);
				matrix.set(2, 1, y * z * (1 - co) - x * si);
				matrix.set(3, 1, 0.0f);

				matrix.set(0, 2, x * z * (1 - co) - y * si);
				matrix.set(1, 2, y * z * (1 - co) + x * si);
				matrix.set(2, 2, z2 + (x2 + y2) * co);
				matrix.set(3, 2, 0.0f);

				matrix.set(0, 3, 0.0f);
				matrix.set(1, 3, 0.0f);
				matrix.set(2, 3, 0.0f);
				matrix.set(3, 3, 1.0f);

				multiply(matrix);
			}

			bool isIdentity() {

				T p_Identity[] = { 1.0f, 0.0f, 0.0f, 0.0f,
					0.0f, 1.0f, 0.0f, 0.0f,
					0.0f, 0.0f, 1.0f, 0.0f,
					0.0f, 0.0f, 0.0f, 1.0f };

				if (memcmp(this->m_Matrix, p_Identity, 16 * sizeof(float)))
					return false;
				else
					return true;
			}

		protected:

			// to return top 3x3 submatrix
			T m_SubMat3[9];

		};


		// ---------------------------------------------------------
		//		NON-SQUARE MATRICES
		//----------------------------------------------------------


		template <typename T, int LINES, int COLUMNS >
		class matrixNS: public Data{

		protected:

		public:

			T m_Matrix[LINES * COLUMNS];

			matrixNS() {
				for (int i = 0; i < LINES*COLUMNS; ++i) {
					m_Matrix[i] = (T)0;
				}
			};

			matrixNS(const T* values) {

				setMatrix(values);
			}

			~matrixNS() {};

			Data *clone() {
				matrixNS<T, LINES, COLUMNS> *m = new(matrixNS<T, LINES, COLUMNS>);
				memcpy(m->m_Matrix, m_Matrix, LINES*COLUMNS*sizeof(T));
				return m;
			}

			void *getPtr() {
				return m_Matrix;
			}

			T at(unsigned int i, unsigned int j) const {
				assert(i < LINES && j < COLUMNS);
				return this->m_Matrix[i * COLUMNS + j];
			}

			void set(unsigned int i, unsigned int j, T value) {
				assert(i < LINES && j < COLUMNS);
				this->m_Matrix[i * COLUMNS + j] = value;
			}

			const T* getMatrix() {
				return this->m_Matrix;
			}

			void setMatrix(const T* values) {

				for (int i = 0; i < LINES*COLUMNS; ++i) {
					this->m_Matrix[i] = values[i];
				}
			}

			// copy into this
			void copy(const matrixNS &m) {
				setMatrix(m.m_Matrix);
			}

			const matrixNS &operator = (const matrixNS &m) {

				setMatrix(m.m_Matrix);
				return *this;
			}

			std::string toString() {

				std::string s;

				for (int i = 0; i < LINES; ++i) {
					s = std::to_string(m_Matrix[i*COLUMNS]);
					for (int j = 1; j < COLUMNS; ++j) {
						s = s + ", " + std::to_string(this->m_Matrix[i*COLUMNS + j]);
					}
					if (i != LINES - 1)
						s = s + " \n";
				}
				return "[" + s + "]";
			}
		};

		typedef matrix2<float> mat2;
		typedef matrix3<float> mat3;
		typedef matrix4<float> mat4;
		typedef matrix2<double> dmat2;
		typedef matrix3<double> dmat3;
		typedef matrix4<double> dmat4;

		typedef matrixNS<float, 2, 3> mat2x3;
		typedef matrixNS<float, 2, 4> mat2x4;
		typedef matrixNS<float, 3, 2> mat3x2;
		typedef matrixNS<float, 3, 4> mat3x4;
		typedef matrixNS<float, 4, 3> mat4x2;
		typedef matrixNS<float, 4, 3> mat4x3;

		typedef matrixNS<double, 2, 3> dmat2x3;
		typedef matrixNS<double, 2, 4> dmat2x4;
		typedef matrixNS<double, 3, 2> dmat3x2;
		typedef matrixNS<double, 3, 4> dmat3x4;
		typedef matrixNS<double, 4, 3> dmat4x2;
		typedef matrixNS<double, 4, 3> dmat4x3;
	};
};

#endif

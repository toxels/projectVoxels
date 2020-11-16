#ifndef VOXELSWITHCUDA_GEOMETRY_CUH
#define VOXELSWITHCUDA_GEOMETRY_CUH

#include <iostream>
#include <cmath>
#include <cfloat>


class Matrix3D{
private:
    double n[3][3]{};
public:
    __device__
    Matrix3D() = default;
    __host__ __device__
    Matrix3D(double n00, double n01, double n02,
             double n10, double n11, double n12,
             double n20, double n21, double n22){
        n[0][0] = n00; n[0][1] = n10; n[0][2] = n20;
        n[1][0] = n01; n[1][1] = n11; n[1][2] = n21;
        n[2][0] = n02; n[2][1] = n12; n[2][2] = n22;
    }
    __device__
    Matrix3D(const double3& a, const double3& b, const double3& c){
        n[0][0] = a.x; n[0][1] = a.y; n[0][2] = a.z;
        n[1][0] = b.x; n[1][1] = b.y; n[1][2] = b.z;
        n[2][0] = c.x; n[2][1] = c.y; n[2][2] = c.z;
    }
    __host__ __device__
    double& operator()(int i, int j){
        return (n[i][j]);
    }
    __host__ __device__
    const double& operator()(int i, int j) const{
        return (n[i][j]);
    }
    __device__
    double3& operator[](int j){
        return (*reinterpret_cast<double3 *>(n[j]));
    }
    __device__
    const double3& operator[](int j) const{
        return (*reinterpret_cast<const double3 *>(n[j]));
    }
};




class Plane {
public:
    double x, y, z, w;
    Plane() = default;
    Plane(double nx, double ny, double nz, double d) {
        x = nx;
        y = ny;
        z = nz;
        w = d;
    }
    Plane(const double3 &n, double d) {
        x = n.x;
        y = n.y;
        z = n.z;
        w = d;
    }
    const double3& GetNormal() const {
        return (reinterpret_cast<const double3 &>(x));
    }
};




/*__device__
inline Point3D operator +(const Point3D& a, const double3& b) {
    return (Point3D(a.x + b.x, a.y + b.y, a.z + b.z));
}
__device__
inline double3 operator -(const Point3D& a, const Point3D& b) {
    return (double3(a.x - b.x, a.y - b.y, a.z - b.z));
}*/

__host__ __device__
inline uint3 operator * (const uint3 v, double s){
    return (make_uint3(v.x * s, v.y * s, v.z * s));
}
__host__ __device__
double myAbs(double a){
    return a > 0.0 ? a : -1 * a;
}
__host__ __device__
bool operator == (double3 v, double3 u){
    double eps = 0.0001;
    return (myAbs(v.x - u.x) < eps) && (myAbs(v.y - u.y) < eps) && (myAbs(v.z - u.z) < eps);
}
__host__ __device__
inline double3 operator * (const double3 v, double s){
    return (make_double3(v.x * s, v.y * s, v.z * s));
}
__host__ __device__
inline double3 operator / (const double3 v, double s){
    s = 1.0 / s;
    return (make_double3(v.x * s, v.y * s, v.z * s));
}
__host__ __device__
inline double3 operator - (const double3 v){
    return (make_double3(-v.x, -v.y, -v.z));
}
__host__ __device__
inline double Magnitude(const double3 v){
    return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}
__host__ __device__
inline double3 Normalize(const double3* v){
    return (*v / Magnitude(*v));
}
__host__ __device__
inline double3 operator + (const double3 a, const double3 b){
    return (make_double3(a.x+b.x, a.y+b.y, a.z+b.z));
}

__host__ __device__
inline double3 operator - (const double3 a, const double3 b){
    return (make_double3(a.x-b.x, a.y-b.y, a.z-b.z));
}
__device__
inline double Dot(const double3& a, const double3& b){
    return (a.x * b.x + a.y * b.y + a.z * b.z);
}
__host__ __device__
inline double3 Cross(const double3& a, const double3& b){
    return (make_double3(a.y * b.z - a.z * b.y,
                     a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x));
}
__device__
inline double3 Project(const double3& a, const double3& b){
    return (b * (Dot(a,b) / Dot(b,b)));
}
__device__
inline double3 Reject(const double3& a, const double3& b){
    return (a - b * (Dot(a,b) / Dot(b,b)));
}

__device__
Matrix3D operator* (const Matrix3D& A, const Matrix3D& B) {
    return (Matrix3D(A(0, 0) * B(0, 0) + A(0, 1) * B(1, 0) + A(0, 2) * B(2, 0),
                     A(0, 0) * B(0, 1) + A(0, 1) * B(1, 1) + A(0, 2) * B(2, 1),
                     A(0, 0) * B(0, 2) + A(0, 1) * B(1, 2) + A(0, 2) * B(2, 2),
                     A(1, 0) * B(0, 0) + A(1, 1) * B(1, 0) + A(1, 2) * B(2, 0),
                     A(1, 0) * B(0, 1) + A(1, 1) * B(1, 1) + A(1, 2) * B(2, 1),
                     A(1, 0) * B(0, 2) + A(1, 1) * B(1, 2) + A(1, 2) * B(2, 2),
                     A(2, 0) * B(0, 0) + A(2, 1) * B(1, 0) + A(2, 2) * B(2, 0),
                     A(2, 0) * B(0, 1) + A(2, 1) * B(1, 1) + A(2, 2) * B(2, 1),
                     A(2, 0) * B(0, 2) + A(2, 1) * B(1, 2) + A(2, 2) * B(2, 2)));
}
__host__ __device__
double3 operator *(const Matrix3D& M, const double3& v) {
    return (make_double3(M(0, 0) * v.x + M(0, 1) * v.y + M(0, 2) * v.z,
                     M(1, 0) * v.x + M(1, 1) * v.y + M(1, 2) * v.z,
                     M(2, 0) * v.x + M(2, 1) * v.y + M(2, 2) * v.z));
}
__device__
double Determinant(const Matrix3D& M){
    return (  M(0,0) * (M(1,1) * M(2,2) - M(1,2) * M(2,1))
              + M(0,1) * (M(1,2) * M(2,0) - M(1,0) * M(2,2))
              + M(0,2) * (M(1,0) * M(2,1) - M(1,1) * M(2,0))) ;
}
__device__
Matrix3D Inverse(const Matrix3D& M){
    const double3& a = M[0];
    const double3& b = M[1];
    const double3& c = M[2];

    double3 r0 = Cross(b,c);
    double3 r1 = Cross(c,a);
    double3 r2 = Cross(a,b);

    double invDet = 1.0 / Dot(r2, c);

    return (Matrix3D(r0.x * invDet, r0.y * invDet, r0.z * invDet,
                     r1.x * invDet, r1.y * invDet, r1.z * invDet,
                     r2.x * invDet, r2.y * invDet, r2.z * invDet));
}
__device__
Matrix3D MakeRotationX(double t){
    double c = cos(t), s = sin(t);
    return (Matrix3D(1.0, 0.0, 0.0,
                     0.0,  c,  -s,
                     0.0,  s,   c));
}
__host__ __device__
Matrix3D MakeRotationY(double t){
    double c = cos(t), s = sin(t);
    return (Matrix3D( c,  0.0,  s,
                      0.0, 1.0, 0.0,
                      -s,  0.0,  c));
}
__host__ __device__
Matrix3D MakeRotationZ(double t){
    double c = cos(t), s = sin(t);
    return (Matrix3D( c,  -s,  0.0,
                      s,   c,  0.0,
                      0.0, 0.0, 1.0));
}
// rotation through the angle t about an axis a
__host__ __device__
Matrix3D MakeRotation(double t, double3& a){ // a - unit vector
    double c = cos(t), s = sin(t), d = 1.0 - c;
    double x = a.x * d, y = a.y * d, z = a.z * d;
    double axay = x * a.y;
    double axaz = x * a. z;
    double ayaz = y * a. z;
    return (Matrix3D(c + x * a.x, axay - s * a.z, axaz + s * a.y,
                     axay + s * a.z, c + y * a.y, ayaz - s * a.x,
                     axaz - s * a.y, ayaz + s * a.x, c + z * a.z));
}
// reflection through the plane perpendicular to unit vector a
__device__
Matrix3D MakeReflection(const double3& a) {
    double x = a.x * -2.0;
    double y = a.y * -2.0;
    double z = a.z * -2.0;
    double axay = x * a.y;
    double axaz = x * a.z;
    double ayaz = y * a.z;
    return (Matrix3D(x * a.x + 1.0, axay, axaz,
                     axay, y * a.y + 1.0, ayaz,
                     axaz, ayaz, z * a.z + 1.0));
}
__device__
Matrix3D MakeScale(double sx, double sy, double sz){
    return (Matrix3D( sx, 0.0, 0.0,
                      0.0,  sy, 0.0,
                      0.0, 0.0,  sz));
}
// a scale by a factor of s along a unit direction a
__device__
Matrix3D MakeScale(float s, const double3& a) {
    s -= 1.0;
    double x = a.x * s;
    double y = a.y * s;
    double z = a.z * s;
    double axay = x * a.y;
    double axaz = x * a.z;
    double ayaz = y * a.z;
    return (Matrix3D(x * a.x + 1.0, axay, axaz,
                     axay, y * a.y + 1.0, ayaz,
                     axaz, ayaz, z * a.z + 1.0));
}
// calculates the distance between the point q and the line determined by the point p and the direction v
/*__device__
double DistPointLine(const Point3D& q, const Point3D& p, const double3& v){
    double3 a = Cross(q - p, v);
    return (sqrt(Dot(a,a) / Dot(v,v)));
}*/
// calculates the distance between two lines determined by the points p1 and p2 and the directions v1 and v2
/*__device__
double DistLineLine(const Point3D& p1, const double3& v1,
                    const Point3D& p2, const double3& v2){
    double3 dp = p2 - p1;
    double v12 = Dot(v1,v1), v22 = Dot(v2,v2), v1v2 = Dot(v1,v2);
    double det = v1v2 * v1v2 - v12 * v22;
    if(fMyAbs(det) > FLT_MIN){
        det = 1.0 / det;
        double dpvl = Dot(dp, v1);
        double dpv2 = Dot(dp, v2);
        double t1 = (v1v2 * dpv2 - v22 * dpvl) * det;
        double t2 = (v12 * dpv2 - v1v2 * dpvl) * det;
        return (Magnitude(dp + v2 * t2 - v1 * t1) );
    }
    // The lines are nearly parallel.
    double3 a = Cross(dp, v1);
    return (sqrt(Dot(a, a) / v12));
}*/

__device__
double Dot(const Plane& f, const double3& v) {
    return (f.x * v.x + f.y * v.y + f.z * v.z);
}
/*__device__
double Dot(const Plane& f, const Point3D& p) {
    return (f.x * p.x + f.y * p.y + f.z * p.z + f.w);
}*/
/*__device__
float IntersectLinePlane(const Point3D& p, const double3& v,
                         const Plane& f, Point3D *q){
    double fv = Dot(f, v);
    if (fMyAbs(fv) > FLT_MIN){
        (*q) = p + (- v * (Dot(f, p) / fv));
        return true;
    }
    return false;
}*/
























#endif //VOXELSWITHCUDA_GEOMETRY_CUH

//
// Created by miner on 18.08.2020.
//

#ifndef FOUNDATIONSOFGAMEENGINEDEVELOPMENT_GEOMETRY_H
#define FOUNDATIONSOFGAMEENGINEDEVELOPMENT_GEOMETRY_H
//
// Created by miner on 18.08.2020.
//

#include <iostream>
#include <cmath>
#include <cfloat>
class Vector3D{
public:
    double x, y, z;

    Vector3D() = default;

    Vector3D(double a, double b, double c){
        x = a;
        y = b;
        z = c;
    }

    double& operator[](int i){ // https://ravesli.com/urok-138-peregruzka-operatora-indeksatsii/
        return (&x)[i];
    }
    const double& operator[](int i) const{  // для константных объектов: только просмотр
        return (&x)[i];
    }

    Vector3D& operator*=(double s){
        x *= s;
        y *= s;
        z *= s;
        return (*this);
    }
    Vector3D&operator/=(double s){
        s = 1.0 / s;
        x *= s;
        y *= s;
        z *= s;
        return (*this);
    }
    Vector3D& operator+=(const Vector3D v){
        x += v.x;
        y += v.y;
        z += v.z;
        return (*this);
    }
    Vector3D& operator-=(const Vector3D v){
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return (*this);
    }

};

struct Point3D : Vector3D{
    Point3D() = default;
    Point3D(double a, double b, double c) : Vector3D(a,b,c) {}
};


class Matrix3D{
private:
    double n[3][3]{};
public:
    Matrix3D() = default;
    Matrix3D(double n00, double n01, double n02,
             double n10, double n11, double n12,
             double n20, double n21, double n22){
        n[0][0] = n00; n[0][1] = n10; n[0][2] = n20;
        n[1][0] = n01; n[1][1] = n11; n[1][2] = n21;
        n[2][0] = n02; n[2][1] = n12; n[2][2] = n22;
    }
    Matrix3D(const Vector3D& a, const Vector3D& b, const Vector3D& c){
        n[0][0] = a.x; n[0][1] = a.y; n[0][2] = a.z;
        n[1][0] = b.x; n[1][1] = b.y; n[1][2] = b.z;
        n[2][0] = c.x; n[2][1] = c.y; n[2][2] = c.z;
    }
    double& operator()(int i, int j){
        return (n[i][j]);
    }
    const double& operator()(int i, int j) const{
        return (n[i][j]);
    }
    Vector3D& operator[](int j){
        return (*reinterpret_cast<Vector3D *>(n[j]));
    }
    const Vector3D& operator[](int j) const{
        return (*reinterpret_cast<const Vector3D *>(n[j]));
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
    Plane(const Vector3D &n, double d) {
        x = n.x;
        y = n.y;
        z = n.z;
        w = d;
    }
    const Vector3D& GetNormal() const {
        return (reinterpret_cast<const Vector3D &>(x));
    }
};




inline Point3D operator +(const Point3D& a, const Vector3D& b) {
    return (Point3D(a.x + b.x, a.y + b.y, a.z + b.z));
}
inline Vector3D operator -(const Point3D& a, const Point3D& b) {
    return (Vector3D(a.x - b.x, a.y - b.y, a.z - b.z));
}
inline Vector3D operator * (const Vector3D v, double s){
    return (Vector3D(v.x * s, v.y * s, v.z * s));
}
inline Vector3D operator / (const Vector3D v, double s){
    s = 1.0 / s;
    return (Vector3D(v.x * s, v.y * s, v.z * s));
}
inline Vector3D operator - (const Vector3D v){
    return (Vector3D(-v.x, -v.y, -v.z));
}
inline double Magnitude(const Vector3D v){
    return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}
inline Vector3D Normalize(const Vector3D v){
    return (v / Magnitude(v));
}
inline Vector3D operator + (const Vector3D a, const Vector3D b){
    return (Vector3D(a.x+b.x, a.y+b.y, a.z+b.z));
}
inline Vector3D operator - (const Vector3D a, const Vector3D b){
    return (Vector3D(a.x-b.x, a.y-b.y, a.z-b.z));
}

inline double Dot(const Vector3D& a, const Vector3D& b){
    return (a.x * b.x + a.y * b.y + a.z * b.z);
}
inline Vector3D Cross(const Vector3D& a, const Vector3D& b){
    return (Vector3D(a.y * b.z - a.z * b.y,
                     a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x));
}
inline Vector3D Project(const Vector3D& a, const Vector3D& b){
    return (b * (Dot(a,b) / Dot(b,b)));
}
inline Vector3D Reject(const Vector3D& a, const Vector3D& b){
    return (a - b * (Dot(a,b) / Dot(b,b)));
}


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
Vector3D operator *(const Matrix3D& M, const Vector3D& v) {
    return (Vector3D(M(0, 0) * v.x + M(0, 1) * v.y + M(0, 2) * v.z,
                     M(1, 0) * v.x + M(1, 1) * v.y + M(1, 2) * v.z,
                     M(2, 0) * v.x + M(2, 1) * v.y + M(2, 2) * v.z));
}
double Determinant(const Matrix3D& M){
    return (  M(0,0) * (M(1,1) * M(2,2) - M(1,2) * M(2,1))
              + M(0,1) * (M(1,2) * M(2,0) - M(1,0) * M(2,2))
              + M(0,2) * (M(1,0) * M(2,1) - M(1,1) * M(2,0))) ;
}
Matrix3D Inverse(const Matrix3D& M){
    const Vector3D& a = M[0];
    const Vector3D& b = M[1];
    const Vector3D& c = M[2];

    Vector3D r0 = Cross(b,c);
    Vector3D r1 = Cross(c,a);
    Vector3D r2 = Cross(a,b);

    double invDet = 1.0 / Dot(r2, c);

    return (Matrix3D(r0.x * invDet, r0.y * invDet, r0.z * invDet,
                     r1.x * invDet, r1.y * invDet, r1.z * invDet,
                     r2.x * invDet, r2.y * invDet, r2.z * invDet));
}

Matrix3D MakeRotationX(double t){
    double c = cos(t), s = sin(t);
    return (Matrix3D(1.0, 0.0, 0.0,
                     0.0,  c,  -s,
                     0.0,  s,   c));
}
Matrix3D MakeRotationY(double t){
    double c = cos(t), s = sin(t);
    return (Matrix3D( c,  0.0,  s,
                      0.0, 1.0, 0.0,
                      -s,  0.0,  c));
}
Matrix3D MakeRotationZ(double t){
    double c = cos(t), s = sin(t);
    return (Matrix3D( c,  -s,  0.0,
                      s,   c,  0.0,
                      0.0, 0.0, 1.0));
}
// rotation through the angle t about an axis a
Matrix3D MakeRotation(double t, Vector3D& a){ // a - unit vector
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
Matrix3D MakeReflection(const Vector3D& a) {
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

Matrix3D MakeScale(double sx, double sy, double sz){
    return (Matrix3D( sx, 0.0, 0.0,
                      0.0,  sy, 0.0,
                      0.0, 0.0,  sz));
}
// a scale by a factor of s along a unit direction a
Matrix3D MakeScale(float s, const Vector3D& a) {
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
double DistPointLine(const Point3D& q, const Point3D& p, const Vector3D& v){
    Vector3D a = Cross(q - p, v);
    return (sqrt(Dot(a,a) / Dot(v,v)));
}
// calculates the distance between two lines determined by the points p1 and p2 and the directions v1 and v2
double DistLineLine(const Point3D& p1, const Vector3D& v1,
                    const Point3D& p2, const Vector3D& v2){
    Vector3D dp = p2 - p1;
    double v12 = Dot(v1,v1), v22 = Dot(v2,v2), v1v2 = Dot(v1,v2);
    double det = v1v2 * v1v2 - v12 * v22;
    if(fabs(det) > FLT_MIN){
        det = 1.0 / det;
        double dpvl = Dot(dp, v1);
        double dpv2 = Dot(dp, v2);
        double t1 = (v1v2 * dpv2 - v22 * dpvl) * det;
        double t2 = (v12 * dpv2 - v1v2 * dpvl) * det;
        return (Magnitude(dp + v2 * t2 - v1 * t1) );
    }
    // The lines are nearly parallel.
    Vector3D a = Cross(dp, v1);
    return (sqrt(Dot(a, a) / v12));
}


double Dot(const Plane& f, const Vector3D& v) {
    return (f.x * v.x + f.y * v.y + f.z * v.z);
}
double Dot(const Plane& f, const Point3D& p) {
    return (f.x * p.x + f.y * p.y + f.z * p.z + f.w);
}
float IntersectLinePlane(const Point3D& p, const Vector3D& v,
                         const Plane& f, Point3D *q){
    double fv = Dot(f, v);
    if (fabs(fv) > FLT_MIN){
        (*q) = p + (- v * (Dot(f, p) / fv));
        return true;
    }
    return false;
}






#endif //FOUNDATIONSOFGAMEENGINEDEVELOPMENT_GEOMETRY_H

#include <iostream>
#include "geometry.h"

#define MAP_SIZE 100


#define distToViewPort 1.0
#define Vh 1.0
#define Vw 1.8

#define imageHeight 400
#define imageWidth 400

class Camera{
public:
    Camera() = default;
    Vector3D eyePosition;
    double angleX, angleY;
    Camera(Vector3D& eye, double xAng, double yAng){
        angleX = xAng;
        angleY = yAng;
        eyePosition = eye;
    }
};

class Ray{
public:
    Vector3D source = Vector3D();
    Vector3D direction = Vector3D();
    Ray() = default;
    Ray(Vector3D m_source, Vector3D m_direction){
        source = m_source;
        direction = m_direction;
    }
};

class Box{
    public:
    Box(const Vector3D &min, const Vector3D &max) {
        bounds[0] = min; bounds[1] = max;
    }
    bool intersect(const Ray&, double t0, double t1, Vector3D& inter1Point, Vector3D& inter2Point) const;
    Vector3D bounds[2];
};

// Smits’ method
bool Box::intersect(const Ray& r, double t0, double t1, Vector3D& inter1Point, Vector3D& inter2Point) const {
    float tmin, tmax, tymin, tymax, tzmin, tzmax;
    if (r.direction.x >= 0) {
        tmin = (bounds[0].x - r.source.x) / r.direction.x;
        tmax = (bounds[1].x - r.source.x) / r.direction.x;
    }
    else {
        tmin = (bounds[1].x - r.source.x) / r.direction.x;
        tmax = (bounds[0].x - r.source.x) / r.direction.x;
    }
    if (r.direction.y >= 0) {
        tymin = (bounds[0].y - r.source.y) / r.direction.y;
        tymax = (bounds[1].y - r.source.y) / r.direction.y;
    }
    else {
        tymin = (bounds[1].y - r.source.y) / r.direction.y;
        tymax = (bounds[0].y - r.source.y) / r.direction.y;
    }
    if ( (tmin > tymax) || (tymin > tmax) )
        return false;
    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;
    if (r.direction.z >= 0) {
        tzmin = (bounds[0].z - r.source.z) / r.direction.z;
        tzmax = (bounds[1].z - r.source.z) / r.direction.z;
    }
    else {
        tzmin = (bounds[1].z - r.source.z) / r.direction.z;
        tzmax = (bounds[0].z - r.source.z) / r.direction.z;
    }
    if ( (tmin > tzmax) || (tzmin > tmax) )
        return false;
    if (tzmin > tmin)
        tmin = tzmin;
    if (tzmax < tmax)
        tmax = tzmax;
    if(( (tmin < t1) && (tmax > t0) )){
        inter1Point = r.source + r.direction * tmin;
        inter2Point = r.source + r.direction * tmax;
        return true;
    }
    return false;
}
Ray computePrimRay(Camera& cam, int i, int j){
    Vector3D projectPlaneOrigin = cam.eyePosition + Vector3D(0.,0.,distToViewPort);
    Vector3D C = projectPlaneOrigin - Vector3D(Vw / 2., Vh /2., 0);
    Vector3D V = C + Vector3D((double)j / imageWidth * Vw,(double)i / imageHeight * Vh,0.);
    Matrix3D rotate = MakeRotationY(-cam.angleX) * MakeRotationX(cam.angleY);
    V = rotate * V;
    Vector3D dir = V-cam.eyePosition;
    return Ray(cam.eyePosition, dir);
}
#define BOX_SIZE 8
int idxesToBoxIdx(int x, int y, int z){
    return x * MAP_SIZE * MAP_SIZE + y * MAP_SIZE + z;
}
enum ax{
    x,y,z
};
ax chooseDivisibleAx(Vector3D& point){
    if((int)point.x % BOX_SIZE && std::abs((int)point.x - point.x) < DBL_EPSILON)
        return x;
    if((int)point.y % BOX_SIZE && std::abs((int)point.y - point.y) < DBL_EPSILON)
        return y;
    if((int)point.z % BOX_SIZE && std::abs((int)point.z - point.z) < DBL_EPSILON)
        return z;
}
inline int sign(double x){
    return x >= 0 ? 1 : -1;
}
int traverseRay(Camera& cam, Ray& ray){
    int X = (int)(cam.eyePosition.x / BOX_SIZE);
    int Y = (int)(cam.eyePosition.y / BOX_SIZE);
    int Z = (int)(cam.eyePosition.z / BOX_SIZE);
    int startBoxIdx = idxesToBoxIdx(X, Y, Z);
    Vector3D min = Vector3D(X*BOX_SIZE, Y*BOX_SIZE, Z*BOX_SIZE);
    Vector3D max = min + Vector3D(BOX_SIZE, BOX_SIZE, BOX_SIZE);
    Box currentBox = Box(min, max);
    Vector3D A1 = Vector3D(), A2 = Vector3D();
    if(currentBox.intersect(ray, -1.0*INFINITY, INFINITY, A1, A2)){
        // firstHitBox - ?
        // TODO если будут артефакты, надо переписывать граничные случаи
        ax currentAx = chooseDivisibleAx(A2);
        int X1 = X, Y1 = Y, Z1 = Z;
        if(currentAx == x)
            X1 += sign(A2.x - A1.x);
        if(currentAx == y)
            Y1 += sign(A2.y - A1.y);
        if(currentAx == z)
            Z1 += sign(A2.z - A1.z);
        printf("%d %d %d\n", X1, Y1, Z1);
    }
    return 666;
}
#define WORLD_SIZE 100
int world[WORLD_SIZE][WORLD_SIZE][WORLD_SIZE];
int main() {
    for(auto & i : world){
        for(auto & j : i){
            for(int & k : j)
                k = rand()%2;
        }
    }
    Vector3D eyePosition = Vector3D(5.75,3.57,6.08);
    Camera cam = Camera(eyePosition, 0., 0.);
    for(int i = 0 ; i < imageHeight ; i++){
        for(int j = 0 ; j < imageWidth ; j++){
            computePrimRay(cam, i, j);
        }
    }
    Ray r = Ray(eyePosition, Vector3D(-2.02, -0.4, 1.92));
    traverseRay(cam,r);
    return 0;
}
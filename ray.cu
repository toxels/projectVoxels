#include "ray.cuh"
__device__
bool RayIntersectsTriangle(double3& rayOrigin,
    double3& rayVector,
    double3& triangleA, double3& triangleB, double3& triangleC,
    double3& outIntersectionPoint)
{
    const double EPSILON = 0.0000001;
    double3 vertex0 = triangleA;
    double3 vertex1 = triangleB;  
    double3 vertex2 = triangleC;
    double3 edge1, edge2, h, s, q;
    double a,f,u,v;
    edge1 = vertex1 - vertex0;
    edge2 = vertex2 - vertex0;
    h = Cross(rayVector, edge2);
    a = Dot(edge1, h);
    if (a > -EPSILON && a < EPSILON)
        return false;    // This ray is parallel to this triangle.
    f = 1.0/a;
    s = rayOrigin - vertex0;
    u = f * Dot(s, h);
    if (u < 0.0 || u > 1.0)
        return false;
    q = Cross(s, edge1);
    v = f * Dot(rayVector, q);
    if (v < 0.0 || u + v > 1.0)
        return false;
    // At this stage we can compute t to find out where the intersection point is on the line.
    double t = f * Dot(edge2, q);
    if (t > EPSILON) // ray intersection
    {
        outIntersectionPoint = rayOrigin + rayVector * t;
        return true;
    }
    else // This means that there is a line intersection but not a ray intersection.
        return false;
}
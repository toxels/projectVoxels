#pragma once
#include <vector>
#include "cudaHeaders.cuh"
#include "geometry.cuh"
struct Ray {
public:
    double3 source;
    double3 direction;
};
bool RayIntersectsTriangle(double3& rayOrigin,
    double3& rayVector,
    double3& triangleA, double3& triangleB, double3& triangleC,
    double3& outIntersectionPoint);
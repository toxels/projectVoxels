#pragma once
#include "voxel.cuh"
class objModel {
public:
	std::vector<double3> vertexes;
	std::vector<int3> faces;
    objModel(std::string fileName);
    
};
voxel* objToVoxel(int voxelModelSize, objModel& objModel, voxel* model);
#pragma once
#include "voxel.cuh"
class objModel {
public:
	std::vector<double3> tmpVertexes;
	std::vector<int3> tmpFaces;
	double3* vertexes;
	int3* faces;
	int numVertexes, numFaces;
    objModel(std::string fileName);
	/*
	__host__ __device__ double3 getVertex(int i) {
		return vertexes[i];
	}

	__host__ __device__ int3 getFace(int i) {
		return faces[i];
	}

	*/
};
voxel* objToVoxel(int voxelModelSize, objModel& objModel, voxel* model);
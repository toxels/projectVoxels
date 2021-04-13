#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "voxelizer.cuh"
#include "voxel.cuh"
#include "geometry.cuh"
#include "cudaHeaders.cuh"
#include <algorithm>
#include "settings.cuh"
#include "ray.cuh"
objModel::objModel(std::string fileName) {
	std::ifstream file(fileName);
	if (file.is_open()) {
        std::string parsedLineType;
        while(file >> parsedLineType){
            if(!parsedLineType.compare("v")){
                double3 vertexCoordinates;
                file >> vertexCoordinates.x >> vertexCoordinates.y >> vertexCoordinates.z;
                tmpVertexes.push_back(vertexCoordinates);
            }
            else if(!parsedLineType.compare("f")){
                int3 faceIndexes;
                file >> faceIndexes.x >> faceIndexes.y >> faceIndexes.z;
                tmpFaces.push_back(faceIndexes);
            }
        }
		numVertexes = tmpVertexes.size();
		numFaces = tmpFaces.size();
		std::cout << "num of vertexes: " << numVertexes << std::endl;
		std::cout << "num of faces: " << numFaces << std::endl;
	}
	else {
		std::cerr << "Unable opening obj model";
	}
}
__global__ void objToVoxelHelper(int voxelModelSize, double3* vertexes, int numVertexes, int3* faces, int numFaces, voxel* voxelModel);
/*
* voxelModelSize - желаемый размер кубической воксельной модели на выходе
*/
__host__
voxel* objToVoxel(int voxelModelSize, objModel& objModel, voxel* voxelModel) {
    double eps = 0.000001;
    double3 minCoordinates = make_double3(DBL_MAX, DBL_MAX, DBL_MAX), maxCoordinates = make_double3(-DBL_MAX, -DBL_MAX, -DBL_MAX);
    for (int i = 0; i < objModel.numVertexes; i++) {
        minCoordinates.x = std::min(minCoordinates.x, objModel.tmpVertexes[i].x);
        minCoordinates.y = std::min(minCoordinates.y, objModel.tmpVertexes[i].y);
        minCoordinates.z = std::min(minCoordinates.z, objModel.tmpVertexes[i].z);
        maxCoordinates.x = std::max(maxCoordinates.x, objModel.tmpVertexes[i].x);
        maxCoordinates.y = std::max(maxCoordinates.y, objModel.tmpVertexes[i].y);
        maxCoordinates.z = std::max(maxCoordinates.z, objModel.tmpVertexes[i].z);
    }
    // вылезаем из отрицательных координат
    for(int i = 0 ; i < objModel.numVertexes ; i++){
        objModel.tmpVertexes[i] = objModel.tmpVertexes[i] - minCoordinates + make_double3(eps, eps, eps);
    }
    // обновляем максимум, не проходя заново по массиву
    maxCoordinates = maxCoordinates - minCoordinates + make_double3(eps, eps, eps);
    // находим самый-самый большой максимум координата
    double maxCoordinate = std::max(std::max(maxCoordinates.x, maxCoordinates.y), maxCoordinates.z);
    double scale = BOX_SIZE * voxelModelSize / maxCoordinate;
    for(int i = 0 ; i < objModel.numVertexes ; i++){
        objModel.tmpVertexes[i] = objModel.tmpVertexes[i] * scale / 2;
    }
	
	int threadsInX = 16;
	int threadsInY = 4;
	int threadsInZ = 4;
	int blocksInX = (voxelModelSize+threadsInX-1)/threadsInX;
	int blocksInY = (voxelModelSize+threadsInY-1)/threadsInY;
	int blocksInZ = (voxelModelSize+threadsInZ-1)/threadsInZ;
	dim3 Dg = dim3(blocksInX, blocksInY, blocksInZ);
	dim3 Db = dim3(threadsInX, threadsInY, threadsInZ);
	std::cout << "CUDA mamangement" << std::endl;
	std::cout << "Before cuda mess: " << cudaGetLastError() << std::endl;
	if (cudaMallocManaged(&objModel.vertexes, sizeof(objModel.tmpVertexes[0]) * objModel.numVertexes))
		fprintf(stderr, "cuda malloc error: vertexes");
	std::cout << "After 1st malloc: " << cudaGetLastError() << std::endl;
	if (cudaMallocManaged(&objModel.faces, sizeof(objModel.tmpFaces[0]) * objModel.numFaces))
		fprintf(stderr, "cuda malloc error: faces");
	std::cout << "After 2d malloc: " << cudaGetLastError() << std::endl;
	cudaMemcpy(&objModel.vertexes[0], objModel.tmpVertexes.data(), objModel.numVertexes*sizeof(objModel.tmpVertexes[0]), cudaMemcpyHostToDevice);
	cudaMemcpy(&objModel.faces[0], objModel.tmpFaces.data(), objModel.numFaces*sizeof(objModel.tmpFaces[0]), cudaMemcpyHostToDevice);
	std::cout << "After memcpys: " <<  cudaGetLastError() << std::endl;
	cudaDeviceSynchronize();
    objToVoxelHelper<<<Dg, Db>>>(voxelModelSize, objModel.vertexes, objModel.numVertexes, objModel.faces, objModel.numFaces, voxelModel);
	std::cout << "After objVoxelHelper: " << cudaGetLastError() << std::endl;
	cudaDeviceSynchronize();
	return voxelModel;
}
__global__
void objToVoxelHelper(int voxelModelSize, double3* vertexes, int numVertexes, int3* faces, int numFaces, voxel* voxelModel) {
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int k = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int j = blockIdx.z*blockDim.z + threadIdx.z;
    
	if (i >= voxelModelSize || j >= voxelModelSize || k >= voxelModelSize)
		return;
	// сидим в частице
	// считаем координаты ее центра
	double3 currentVoxelCenterCoordinates = make_double3(
		i * BOX_SIZE + (double)BOX_SIZE / 2,
		j * BOX_SIZE + (double)BOX_SIZE / 2,
		k * BOX_SIZE + (double)BOX_SIZE / 2);

	// считаем противоположно направленные произвольные лучи, проходящие через центр текущей частицы
	Ray arbitraryRay1{ currentVoxelCenterCoordinates, make_double3(1.0, 0.0, 0.0) };
	Ray arbitraryRay2{ currentVoxelCenterCoordinates, make_double3(-1.0, 0.0, 0.0) };
	
	int counterRay1 = 0, counterRay2 = 0;
	// проходим по всем =) треугольникам и проверяем пересечения
	for (int t = 0; t < numFaces; t++) {
		double3 unused;
		if (RayIntersectsTriangle(arbitraryRay1.source, arbitraryRay1.direction,
			vertexes[faces[t].x - 1],
			vertexes[faces[t].y - 1],
			vertexes[faces[t].z - 1], unused)) {
			counterRay1++;
		}
		if (RayIntersectsTriangle(arbitraryRay2.source, arbitraryRay2.direction,
			vertexes[faces[t].x - 1],
			vertexes[faces[t].y - 1],
			vertexes[faces[t].z - 1], unused)) {
			counterRay2++;
		}
	}
	if (counterRay1 % 2 == 1 && counterRay2 % 2 == 1) {
        //__syncthreads();
		// i j k +
		// i k j
		// j i k +
		// j k i +
		// k j i +
		// k i j +
		voxelModel[i * voxelModelSize * voxelModelSize + k * voxelModelSize + j].setActive();
		voxelModel[i * voxelModelSize * voxelModelSize + k * voxelModelSize + j].setColor(0, 255, 120);
	}
}

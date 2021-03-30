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
                vertexes.push_back(vertexCoordinates);
            }
            else if(!parsedLineType.compare("f")){
                int3 faceIndexes;
                file >> faceIndexes.x >> faceIndexes.y >> faceIndexes.z;
                faces.push_back(faceIndexes);
            }
        }
        std::cout << vertexes.size() << " vertexes parsed\n";
        std::cout << faces.size() << " faces parsed\n";
	}
	else {
		std::cerr << "Unable opening obj model";
	}
}
/*
* voxelModelSize - желаемый размер кубической воксельной модели на выходе
*/
__device__ __host__
voxel* objToVoxel(int voxelModelSize, objModel& objModel, voxel* voxelModel) {
    double eps = 0.000001;
    double3 minCoordinates = make_double3(DBL_MAX, DBL_MAX, DBL_MAX), maxCoordinates = make_double3(-DBL_MAX, -DBL_MAX, -DBL_MAX);
    for (int i = 0; i < objModel.vertexes.size(); i++) {
        minCoordinates.x = std::min(minCoordinates.x, objModel.vertexes[i].x);
        minCoordinates.y = std::min(minCoordinates.y, objModel.vertexes[i].y);
        minCoordinates.z = std::min(minCoordinates.z, objModel.vertexes[i].z);
        maxCoordinates.x = std::max(maxCoordinates.x, objModel.vertexes[i].x);
        maxCoordinates.y = std::max(maxCoordinates.y, objModel.vertexes[i].y);
        maxCoordinates.z = std::max(maxCoordinates.z, objModel.vertexes[i].z);
    }
    // вылезаем из отрицательных координат
    for(int i = 0 ; i < objModel.vertexes.size() ; i++){
        objModel.vertexes[i] = objModel.vertexes[i] - minCoordinates + make_double3(eps, eps, eps);
    }
    // обновляем максимум, не проходя заново по массиву
    maxCoordinates = maxCoordinates - minCoordinates + make_double3(eps, eps, eps);
    // находим самый-самый большой максимум координата
    double maxCoordinate = std::max(std::max(maxCoordinates.x, maxCoordinates.y), maxCoordinates.z);
    double scale = BOX_SIZE * voxelModelSize / maxCoordinate;
    for(int i = 0 ; i < objModel.vertexes.size() ; i++){
        objModel.vertexes[i] = objModel.vertexes[i] * scale / 2;
    }
    for (int i = 0; i < voxelModelSize; i++) {
        for (int j = 0; j < voxelModelSize; j++) {
            for (int k = 0; k < voxelModelSize; k++) {
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
                for (int t = 0; t < objModel.faces.size(); t++) {
                    double3 unused;
                    if (RayIntersectsTriangle(arbitraryRay1.source, arbitraryRay1.direction,
                        objModel.vertexes[objModel.faces[t].x - 1],
                        objModel.vertexes[objModel.faces[t].y - 1],
                        objModel.vertexes[objModel.faces[t].z - 1], unused)) {
                        counterRay1++;
                    }
                    if (RayIntersectsTriangle(arbitraryRay2.source, arbitraryRay2.direction,
                        objModel.vertexes[objModel.faces[t].x - 1],
                        objModel.vertexes[objModel.faces[t].y - 1],
                        objModel.vertexes[objModel.faces[t].z - 1], unused)) {
                        counterRay2++;
                    }
                }
                if (counterRay1 % 2 == 1 && counterRay2 % 2 == 1) {
                    voxelModel[i * voxelModelSize * voxelModelSize + j * voxelModelSize + k].setActive();
                    voxelModel[i * voxelModelSize * voxelModelSize + j * voxelModelSize + k].setColor(0, 255, 120);
                }
            }
        }
    }
    return voxelModel;
}

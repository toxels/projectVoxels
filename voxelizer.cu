#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "voxelizer.cuh"
#include "voxel.cuh"
#include "geometry.cuh"
class objModel {
public:
	int nVertexes;
	int nFaces;
    double3 maxs = make_double3(-DBL_MAX, -DBL_MAX, -DBL_MAX);
    double3 mins = make_double3(DBL_MAX, DBL_MAX, DBL_MAX);
	std::vector<double3> vertexes;
	std::vector<int3> faces;
    objModel(std::string fileName);
    
};
objModel::objModel(std::string fileName) {
	std::ifstream file(fileName);
    nVertexes = 0;
    nFaces = 0;
	if (file.is_open()) {
        std::string parsedLineType;
        while(file >> parsedLineType){
            if(!parsedLineType.compare("v")){
                nVertexes++;
                double3 vertexCoordinates;
                file >> vertexCoordinates.x >> vertexCoordinates.y >> vertexCoordinates.z;
                vertexes.push_back(vertexCoordinates);
            }
            else if(!parsedLineType.compare("f")){
                nFaces++;
                int3 faceIndexes;
                file >> faceIndexes.x >> faceIndexes.y >> faceIndexes.z;
                faces.push_back(faceIndexes);
            }
        }
        std::cout << nVertexes << " vertexes parsed\n";
        std::cout << nFaces << " faces parsed\n";
	}
	else {
		std::cerr << "Unable opening obj model";
	}
}

voxel * objToVoxel(int voxelModelSize, objModel& objModel){
    voxel * voxelModel = (voxel*)malloc(voxelModelSize * voxelModelSize * voxelModelSize * sizeof(voxel));
    for(int i = 0 ; i < objModel.nVertexes ; i++){
        for(int j = 0 ; j < 3 ; j++){
            // вроде бы координаты заданы в [0, 1] но я могу обосраться
            //objModel.vertexes[i][j] *= map_size;
        }
    }
    return voxelModel;
}

#include "saver.cuh"
void saveVoxelModel(voxel* model, int sizeX, int sizeY, int sizeZ, const std::string& fileName) {
    headerToSave header;
    header.sizeX = sizeX;
    header.sizeY = sizeY;
    header.sizeZ = sizeZ;
    std::ofstream file(fileName, std::ios::out | std::ios::binary);
    if (!file) {
        std::cout << "Cannot open file to save the model" << std::endl;
    }
    file.write((char*) &header, sizeof(header));
    for (int i = 0; i < sizeX; i++) {
        for (int j = 0; j < sizeY; j++) {
            file.write((char*) &model[i * sizeX * sizeX + j * sizeY], sizeZ * sizeof(voxel));
        }
    }
        
}
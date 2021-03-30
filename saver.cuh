#pragma once
#include "voxel.cuh"
#include <fstream>
#include <iostream>
struct headerToSave {
    int sizeX, sizeY, sizeZ;
};

void saveVoxelModel(voxel* model, int sizeX, int sizeY, int sizeZ, const std::string& fileName);

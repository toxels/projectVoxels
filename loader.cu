#include "voxel.cuh"
#include "saver.cuh"
#include <fstream>
#include <iostream>

voxelModel* loadVoxelModel(const std::string& savedWorldFileName)
{
    std::ifstream file(savedWorldFileName, std::ios::out | std::ios::binary); // open file on "wb" mode
    if (!file) {
        std::cout << "Cannot open file to load the model" << std::endl;
        return NULL;
    }

    headerToSave header;
    file.read((char*)&header, sizeof(header));  // read the header of save-file
    // TODO не пон€тно что делать с дефайнами размера мира; если их не мен€ть, а размер мира будет другим, то возможна —ћЁ–“№
    voxel* model;
    if (cudaMallocManaged(&model, header.sizeX * header.sizeY * header.sizeZ * sizeof(model[0])))
        fprintf(stderr, "cuda malloc error: model");

    for (int i = 0; i < header.sizeX; i++) {    // reading saved array of bytes
        for (int j = 0; j < header.sizeY; j++) {
            file.read((char*)&model[i * header.sizeX * header.sizeX + j * header.sizeY], header.sizeZ * sizeof(voxel));
        }
    }
    voxelModel resultModel(header.sizeX, header.sizeY, header.sizeZ, model);
    return &resultModel;
}
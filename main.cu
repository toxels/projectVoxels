#include <iostream>
#include <fstream>
#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>
#include <SFML/Window.hpp>
#include <winuser.h>
#include <Windows.h>
#include <vector>
#include <fstream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "builtin_types.h"
#include "vector_functions.h"

#include "geometry.cuh"
#include "voxel.cuh"
#include "tga.cuh"

#include "saver.cuh"

#include "voxelizer.cuh"



#define M_PI (3.1415)
#define MAP_SIZE (200)


#define distToViewPort (1.0)
#define Vh (1.0)
#define Vw (1.0)

#define imageHeight (1024)
#define imageWidth (1024)

#define crossSize (64)
#define BOX_SIZE (4)

#define MAX_FRAMES (1000)
//#define DEBUG
#define LOAD_FROM_FILE


struct Camera {
public:
    Camera() = default;

    double3 eyePosition;
    double angleX, angleY;
    double speed;
    
    Camera(double3 &eye, double xAng, double yAng) :
            angleX(xAng), angleY(yAng), eyePosition(eye), speed(0.3) {}
};

struct Ray {
public:
    double3 source;
    double3 direction;
};

class Box {
public:
    __host__ __device__
    Box() = default;

    __host__ __device__
    Box(int X, int Y, int Z) : x(X), y(Y), z(Z) {
        updateMinMax();
    }

    __host__ __device__
    bool intersect(const Ray &, double t0, double t1, double3 &inter1Point, double3 &inter2Point) const;

    double3 bounds[2]{};

    __host__ __device__
    void inc_x() {
        x++;
        updateMinMax();
    }

    __host__ __device__
    void inc_y() {
        y++;
        updateMinMax();
    }

    __host__ __device__
    void inc_z() {
        z++;
        updateMinMax();
    }

    __host__ __device__
    void dec_x() {
        x--;
        updateMinMax();
    }

    __host__ __device__
    void dec_y() {
        y--;
        updateMinMax();
    }

    __host__ __device__
    void dec_z() {
        z--;
        updateMinMax();
    }

    __host__ __device__
    int get_x() { return x; }

    __host__ __device__
    int get_y() { return y; }

    __host__ __device__
    int get_z() { return z; }

private:
    int x, y, z;

    __host__ __device__
    void updateMinMax() {
        bounds[0] = make_double3(x * BOX_SIZE, y * BOX_SIZE, z * BOX_SIZE);
        bounds[1] = bounds[0] + make_double3(BOX_SIZE, BOX_SIZE, BOX_SIZE);
    }
};

// ищет пересечение луча и коробки, записывает точки в interPoint
__host__ __device__
bool Box::intersect(const Ray &r, double t0, double t1, double3 &inter1Point, double3 &inter2Point) const {
    double tmin, tmax, tymin, tymax, tzmin, tzmax;
    if (r.direction.x >= 0) {
        tmin = (bounds[0].x - r.source.x) / r.direction.x;
        tmax = (bounds[1].x - r.source.x) / r.direction.x;
    } else {
        tmin = (bounds[1].x - r.source.x) / r.direction.x;
        tmax = (bounds[0].x - r.source.x) / r.direction.x;
    }
    if (r.direction.y >= 0) {
        tymin = (bounds[0].y - r.source.y) / r.direction.y;
        tymax = (bounds[1].y - r.source.y) / r.direction.y;
    } else {
        tymin = (bounds[1].y - r.source.y) / r.direction.y;
        tymax = (bounds[0].y - r.source.y) / r.direction.y;
    }
    if ((tmin > tymax) || (tymin > tmax))
        return false;
    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;
    if (r.direction.z >= 0) {
        tzmin = (bounds[0].z - r.source.z) / r.direction.z;
        tzmax = (bounds[1].z - r.source.z) / r.direction.z;
    } else {
        tzmin = (bounds[1].z - r.source.z) / r.direction.z;
        tzmax = (bounds[0].z - r.source.z) / r.direction.z;
    }
    if ((tmin > tzmax) || (tzmin > tmax))
        return false;
    if (tzmin > tmin)
        tmin = tzmin;
    if (tzmax < tmax)
        tmax = tzmax;
    if (((tmin < t1) && (tmax > t0))) {
        inter1Point = r.source + r.direction * tmin;
        inter2Point = r.source + r.direction * tmax;
        return true;
    }
    return false;
}

__host__ __device__
void buildDirectionOfStraightMove(Camera *cam, double3 *move) {
    double3 origin = make_double3(0., 0., 1.);
    double3 xRotated = MakeRotationY((cam->angleX) * M_PI / 180.) * make_double3(1.0, 0.0, 0.0);
    double3 result = MakeRotationY((cam->angleX) * M_PI / 180.) * origin;
    result = MakeRotation(-cam->angleY * M_PI / 180., xRotated) * result;
    *move = result;
}

__host__ __device__
void buildDirectionOfNotSoStraightMove(Camera *cam, double3 *move) {
    double3 origin = make_double3(0., 0., 1.);
    double3 xRotated = MakeRotationY((cam->angleX) * M_PI / 180.) * make_double3(1.0, 0.0, 0.0);
    double3 result = MakeRotationY((cam->angleX) * M_PI / 180.) * origin;
    result = MakeRotation(-cam->angleY * M_PI / 180., xRotated) * result;
    double3 viewDirection = result;
    double3 y = make_double3(0., 1., 0.);
    *move = Cross(viewDirection, y);
}

__host__ __device__
Ray computePrimRay(Camera *cam, const int i, const int j) {
    double3 projectPlaneOrigin = make_double3(0, 0, distToViewPort);
    double3 rightArrow = make_double3(Vw, 0, 00), downArrow = make_double3(0.0, Vh, 0.0);
    double3 leftTopCorner = projectPlaneOrigin - rightArrow / 2 - downArrow / 2;
    double3 hitDotOnProjectPlane = leftTopCorner
                                   + rightArrow * (static_cast<double>(j) / imageWidth)
                                   + downArrow * (static_cast<double>(i) / imageHeight);
    double3 xRotated = MakeRotationY((cam->angleX) * M_PI / 180.) * make_double3(1.0, 0.0, 0.0);
    hitDotOnProjectPlane = MakeRotationY((cam->angleX) * M_PI / 180.) * hitDotOnProjectPlane;
    hitDotOnProjectPlane = MakeRotation(-cam->angleY * M_PI / 180., xRotated) * hitDotOnProjectPlane;
    hitDotOnProjectPlane = hitDotOnProjectPlane + cam->eyePosition;
    double3 dir = hitDotOnProjectPlane - cam->eyePosition;
    return {cam->eyePosition, dir};
}


__host__ __device__
bool checkWorld(Box *box, voxel* world) {
    if (box->get_x() < 0 || box->get_y() < 0 || box->get_z() < 0 || box->get_x() >= MAP_SIZE ||
        box->get_y() >= MAP_SIZE || box->get_z() >= MAP_SIZE)
        return 0;
    return world[box->get_x() * MAP_SIZE * MAP_SIZE + box->get_y() * MAP_SIZE + box->get_z()].isActive();
}


/* (x, y, z) - индексы текущего бокса в world */
__device__
double3 traverseRay(int startX, int startY, int startZ, Ray &ray, int deep, voxel* world, Box *lastBox) {
    Box currentBox = Box(startX, startY, startZ);
    ray.direction = Normalize(&ray.direction);
    double3 deltaT;

    double t_x = ((BOX_SIZE - ray.source.x) / ray.direction.x), t_y = ((BOX_SIZE - ray.source.y) / ray.direction.y), t_z = ((BOX_SIZE - ray.source.z) / ray.direction.z) ;
    if (ray.direction.x < 0) {
        deltaT.x = -BOX_SIZE / ray.direction.x;
        t_x = (floor(ray.source.x / BOX_SIZE) * BOX_SIZE
               - ray.source.x) / ray.direction.x;
    }
    else {
        deltaT.x = BOX_SIZE / ray.direction.x;
        t_x = ((floor(ray.source.x / BOX_SIZE) + 1) * BOX_SIZE
               - ray.source.x) / ray.direction.x;
    }

    if (ray.direction.y < 0) {
        deltaT.y = -BOX_SIZE / ray.direction.y;
        t_y = (floor(ray.source.y / BOX_SIZE) * BOX_SIZE
               - ray.source.y) / ray.direction.y;
    }
    else {
        deltaT.y = BOX_SIZE / ray.direction.y;
        t_y = ((floor(ray.source.y / BOX_SIZE) + 1) * BOX_SIZE
               - ray.source.y) / ray.direction.y;
    }

    if (ray.direction.z < 0) {
        deltaT.z = -BOX_SIZE / ray.direction.z;
        t_z = (floor(ray.source.z / BOX_SIZE) * BOX_SIZE
               - ray.source.z) / ray.direction.z;
    }
    else {
        deltaT.z = BOX_SIZE / ray.direction.z;
        t_z = ((floor(ray.source.z / BOX_SIZE) + 1) * BOX_SIZE
               - ray.source.z) / ray.direction.z;
    }
    while (true) {
        if (currentBox.get_x() < 0 || currentBox.get_y() < 0 || currentBox.get_z() < 0 ||
            currentBox.get_x() >= MAP_SIZE || currentBox.get_y() >= MAP_SIZE || currentBox.get_z() >= MAP_SIZE /*|| deep > MAP_SIZE * 2*/) {
            *lastBox = Box(-1, -1, -1);
            return make_double3(-1., -1., -1.);
        }
        double t = 0.;

        if (t_x < t_y) {
            if (t_x < t_z) {
                t = t_x;
                t_x += deltaT.x; // increment, next crossing along x
                if(ray.direction.x < 0)
                    currentBox.dec_x();
                else
                    currentBox.inc_x();
            }
            else {
                t = t_z;
                t_z += deltaT.z; // increment, next crossing along x
                if(ray.direction.z < 0)
                    currentBox.dec_z();
                else
                    currentBox.inc_z();
            }
        }
        else {
            if (t_y < t_z) {
                t = t_y;
                t_y += deltaT.y; // increment, next crossing along x
                if(ray.direction.y < 0)
                    currentBox.dec_y();
                else
                    currentBox.inc_y();
            }
            else {
                t = t_z;
                t_z += deltaT.z; // increment, next crossing along x
                if(ray.direction.z < 0)
                    currentBox.dec_z();
                else
                    currentBox.inc_z();
            }
        }
        if (checkWorld(&currentBox, world)) {
            *lastBox = currentBox;
            return ray.source + ray.direction * t;
        }
        deep++;
    }
}

/* копипаста траверс_рея для удаления блоков*/
__host__ __device__
bool
hitRay(int startX, int startY, int startZ, Ray &ray, int deep, Box &boxToDelete, Box &boxToAdd, voxel* world) {
    const double eps = 0.000001;
    Box currentBox = Box(startX, startY, startZ);
    while (true) {
        if (currentBox.get_x() < 0 || currentBox.get_y() < 0 || currentBox.get_z() < 0 ||
            currentBox.get_x() >= MAP_SIZE || currentBox.get_y() >= MAP_SIZE || currentBox.get_z() >= MAP_SIZE ||
            deep > 150)
            return false;
        /* A1 < A2 : точки пересечения луча и бокса */
        double3 A1 = double3(), A2 = double3();
        if (currentBox.intersect(ray, 0, INFINITY, A1, A2)) {
            boxToAdd = currentBox;
            double3 A2_normalized = A2 - currentBox.bounds[0];
            if (abs(A2_normalized.x) < eps)
                currentBox.dec_x();
            if (abs(A2_normalized.y) < eps)
                currentBox.dec_y();
            if (abs(A2_normalized.z) < eps)
                currentBox.dec_z();
            if (abs(A2_normalized.x - BOX_SIZE) < eps)
                currentBox.inc_x();
            if (abs(A2_normalized.y - BOX_SIZE) < eps)
                currentBox.inc_y();
            if (abs(A2_normalized.z - BOX_SIZE) < eps)
                currentBox.inc_z();
            if (checkWorld(&currentBox, world)) {
                boxToDelete = currentBox;
                return true;
            }
        }
        deep++;
    }

}

// TODO нужно ли делать это в хосте?
__host__ __device__
void deleteVoxel(Camera *cam, voxel* world) {
    Ray hit = computePrimRay(cam, imageWidth / 2, imageHeight / 2);
    Box boxToDelete = Box(0, 0, 0), boxToAdd = Box(0, 0, 0);
    if (hitRay(static_cast<int>(cam->eyePosition.x / BOX_SIZE),
               static_cast<int>(cam->eyePosition.y / BOX_SIZE),
               static_cast<int>(cam->eyePosition.z / BOX_SIZE),
               hit, 5, boxToDelete, boxToAdd, world)) {
        int dx[] = { 1, 0, -1, 0, 0, 0};
        int dy[] = { 0, 1, 0, -1, 0, 0 };
        int dz[] = { 0, 0, 0, 0, 1, -1 };
        for (int i = 0; i < 6; i++) {
            world[(boxToDelete.get_x() + dx[i]) * MAP_SIZE * MAP_SIZE + (boxToDelete.get_y() + dy[i]) * MAP_SIZE + boxToDelete.get_z() + dz[i]].setInactive();
        }
        world[boxToDelete.get_x() * MAP_SIZE * MAP_SIZE + boxToDelete.get_y() * MAP_SIZE + boxToDelete.get_z()].setInactive();
    }

}

__host__ __device__
void addVoxel(Camera *cam, voxel* world) {
    Ray hit = computePrimRay(cam, imageWidth / 2, imageHeight / 2);
    Box boxToAdd = Box(0, 0, 0), boxToDelete = Box(0, 0, 0);
    if (hitRay(static_cast<int>(cam->eyePosition.x / BOX_SIZE),
               static_cast<int>(cam->eyePosition.y / BOX_SIZE),
               static_cast<int>(cam->eyePosition.z / BOX_SIZE),
               hit, 5, boxToDelete, boxToAdd, world)) {
        world[boxToAdd.get_x() * MAP_SIZE * MAP_SIZE + boxToAdd.get_y() * MAP_SIZE + boxToAdd.get_z()].setActive();
    }
}



__global__
void traversePixels(uint3 *screen, Camera *cam, voxel* world, double3 *lightSource) {
    __shared__ uint3 temp[512];
    __shared__ double3 firstHitDots[512];
    __shared__ Camera sharedCam;
    __shared__ double3 firstHitDotsNormalized[512];
    sharedCam = *cam;
    double eps = 0.0000001;
    Box currBox = Box();

    uchar3 color;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = j + i * blockDim.x * gridDim.x;

    int linearThreadIdxInBlock = threadIdx.x + threadIdx.y * 16;

    Ray primRay = computePrimRay(cam, i, j);
    firstHitDots[linearThreadIdxInBlock] = traverseRay((static_cast<int>(sharedCam.eyePosition.x / BOX_SIZE)),
                                                       (static_cast<int>(sharedCam.eyePosition.y / BOX_SIZE)),
                                                       (static_cast<int>(sharedCam.eyePosition.z / BOX_SIZE)), primRay, 0, world, &currBox);
    double3 emptyConst = make_double3(-1., -1., -1.);
    if (firstHitDots[linearThreadIdxInBlock] == emptyConst) {
        /** мы не коснулись ничего = небо */
        color = make_uchar3(21, 4, 133);
    } else if (checkWorld(&currBox, world) && !world[currBox.get_x() * MAP_SIZE * MAP_SIZE + currBox.get_y() * MAP_SIZE + currBox.get_z()].isLight()) {
        __syncthreads();
        int3 coordinatesOfVoxel = make_int3(currBox.get_x(), currBox.get_y(), currBox.get_z());
        color = world[coordinatesOfVoxel.x * MAP_SIZE * MAP_SIZE + coordinatesOfVoxel.y * MAP_SIZE + coordinatesOfVoxel.z].color;
        double3 dir =  firstHitDots[linearThreadIdxInBlock] - *lightSource;
        Ray shadowRay;
        shadowRay.source = *lightSource;
        shadowRay.direction = dir;
        double3 lastLightHit = traverseRay((static_cast<int>(lightSource->x / BOX_SIZE)),
                                           (static_cast<int>(lightSource->y / BOX_SIZE)),
                                           (static_cast<int>(lightSource->z / BOX_SIZE)), shadowRay, 0, world,
                                           &currBox);
        cudaDeviceSynchronize();
        if (!(lastLightHit == firstHitDots[linearThreadIdxInBlock])) {
            /** случай когда точка падения полностью в тени */
            color = color * 0.2;
        } else {
            /** случай когда свет дошел до точки */
            /** найти на какой грани лежит точка firstHitDot */
            firstHitDotsNormalized[linearThreadIdxInBlock] =
                firstHitDots[linearThreadIdxInBlock] - make_double3(currBox.get_x() * BOX_SIZE,
                                                                    currBox.get_y() * BOX_SIZE,
                                                                    currBox.get_z() * BOX_SIZE);
            double3 normal = make_double3(0., 0., 0.);
            //cudaDeviceSynchronize();
            if (abs(firstHitDotsNormalized[linearThreadIdxInBlock].x) < eps)
                normal.x = -1.;
            if (abs(firstHitDotsNormalized[linearThreadIdxInBlock].x - BOX_SIZE) < eps)
                normal.x = +1.;
            if (abs(firstHitDotsNormalized[linearThreadIdxInBlock].y) < eps)
                normal.y = -1.;
            if (abs(firstHitDotsNormalized[linearThreadIdxInBlock].y - BOX_SIZE) < eps)
                normal.y = +1.;
            if (abs(firstHitDotsNormalized[linearThreadIdxInBlock].z) < eps)
                normal.z = -1.;
            if (abs(firstHitDotsNormalized[linearThreadIdxInBlock].z - BOX_SIZE) < eps)
                normal.z = +1.;
            double lightIntensity = 0.2;
            double cosx = Dot(normal, shadowRay.direction * -1.) / Magnitude(normal) / Magnitude(shadowRay.direction);

            //double diffuser = (Magnitude(firstHitDots[linearThreadIdxInBlock] - *lightSource));
            //cosx = 130000 * cosx / (diffuser * diffuser);

            if (cosx >= eps)
                lightIntensity += cosx * 0.8;


            if (lightIntensity > 1.)
                lightIntensity = 1.0;
            color = color * lightIntensity;
            //cudaDeviceSynchronize();
        }
    } else {
        /** куб Валера */
        color = make_uchar3(255, 255, 255);
    }
    temp[linearThreadIdxInBlock].x = (static_cast<unsigned char>(color.x));
    temp[linearThreadIdxInBlock].y = (static_cast<unsigned char>(color.y));
    temp[linearThreadIdxInBlock].z = (static_cast<unsigned char>(color.z));
    __syncthreads();
    screen[idx].x = temp[linearThreadIdxInBlock].x;
    screen[idx].y = temp[linearThreadIdxInBlock].y;
    screen[idx].z = temp[linearThreadIdxInBlock].z;

}
bool bounds(double3 pos) {
    if (pos.x >= (MAP_SIZE - 1) * BOX_SIZE || pos.y >= (MAP_SIZE - 1) * BOX_SIZE ||
        pos.z >= ((MAP_SIZE - 1) * BOX_SIZE) || pos.x <= 0 || pos.y <= 0 || pos.z <= 0)
        return false;
    return true;
}

void printDebug(Camera *cam) {
    printf("angleX: %lf\n", cam->angleX);
    printf("angleY: %lf\n", cam->angleY);
    printf("eyePosition: (%lf, %lf, %lf)\n", cam->eyePosition.x, cam->eyePosition.y, cam->eyePosition.z);
}



int main() {
    int frames = 0;
    float sumScreenRenderTime = 0.0;

    voxel *world;
    Camera *cam;
    uint3 *screen;
    double3* light;

    // TODO можно вынести в функцию загрузки/создания мира, но всё вместе, чтобы не создавать world отдельно
#ifdef LOAD_FROM_FILE
    std::ifstream file("save.dat", std::ios::out | std::ios::binary); // open file on "wb" mode
    if (!file) {
        std::cout << "Cannot open file to load the world" << std::endl;
        return 1; // возможно стоит убрать и просто генерить мир самим
    }
    
    headerToSave header;   
    file.read((char*)&header, sizeof(header));  // read the header of save-file
    // TODO не понятно что делать с дефайнами размера мира; если их не менять, а размер мира будет другим, то возможна СМЭРТЬ
    if (cudaMallocManaged(&world, header.sizeX * header.sizeY * header.sizeZ * sizeof(world[0])))
        fprintf(stderr, "cuda malloc error: world");
    
    for (int i = 0; i < header.sizeX; i++) {    // reading saved array of bytes
        for (int j = 0; j < header.sizeY; j++) {
            file.read((char*)&world[i * header.sizeX * header.sizeX + j * header.sizeY], header.sizeZ * sizeof(voxel));
        }
    }
#else
    if (cudaMallocManaged(&world, MAP_SIZE * MAP_SIZE * MAP_SIZE * sizeof(world[0])))
        fprintf(stderr, "cuda malloc error: world");

    unsigned char* heightMap = NULL;
    int heightMapWidth, heightMapHeight, heightMapChannels;
    LoadTGA("heightmap.tga", heightMap, heightMapWidth, heightMapHeight, heightMapChannels);

    unsigned char* photoTga = NULL;
    int photoTgaWidth, photoTgaHeight, photoTgaChannels;
    LoadTGA("shlepa.tga", photoTga, photoTgaWidth, photoTgaHeight, photoTgaChannels);

    std::cout << "heightMapChannels: " << heightMapChannels << std::endl;
    for (int i = 0; i < heightMapHeight; i++) {
        for (int j = 0; j < heightMapWidth; j++) {
            int x = i;
            int y = MAP_SIZE - ((heightMap[i * heightMapWidth + j]) * (MAP_SIZE / 2)) / (256 * 3);
            int z = j;
            for (; y < MAP_SIZE; y++) {
                int idx = x * MAP_SIZE * MAP_SIZE + y * MAP_SIZE + z;
                world[idx].setActive();
                if (y > MAP_SIZE - 20)
                    world[idx].setColor(19, 133, 16);
                else
                    world[idx].setColor(240, 240, 240);
            }
        }
    }
    std::cout << "photoTgaChannels: " << photoTgaChannels << std::endl;
    std::cout << "photoTgaHeight: " << photoTgaHeight << std::endl;
    std::cout << "photoTgaWidth: " << photoTgaWidth << std::endl;
    for (int i = 0; i < photoTgaHeight; i++) {
        for (int j = 0; j < photoTgaWidth; j++) {
            int x = j;
            int y = 0;
            int z = i;
            int idx = x * MAP_SIZE * MAP_SIZE + y * MAP_SIZE + z;
            world[idx].setActive();
            world[idx].setColor(photoTga[i * 3 * photoTgaWidth + j * 3], photoTga[i * 3 * photoTgaWidth + j * 3 + 1], photoTga[i * 3 * photoTgaWidth + j * 3 + 2]);
        }
    }

#endif // LOAD_FROM_FILE
    if (cudaMallocManaged(&screen, imageHeight * imageWidth * sizeof(uint3)))
        fprintf(stderr, "cuda malloc error: screen");
    if (cudaMallocManaged(&cam, sizeof(Camera)))
        fprintf(stderr, "cuda malloc error: camera");
    if (cudaMallocManaged(&light, sizeof(double3)))
        fprintf(stderr, "cuda malloc error: light");
    uint3 localLight = make_uint3(MAP_SIZE / 2, 15, MAP_SIZE / 2);
    auto *hostScreen = static_cast<uint3 *>(malloc(imageHeight * imageWidth * sizeof(uint3)));
    
    int blocksCnt = 0;
    // MAP GENERATION
    /*for (int i = 0; i < MAP_SIZE * MAP_SIZE * MAP_SIZE; i++) {
        int x, y, z;
        x = i / MAP_SIZE / MAP_SIZE;
        y = i / MAP_SIZE % MAP_SIZE;
        z = i % MAP_SIZE;
        int R = 35;
        if ((x - MAP_SIZE / 2) * (x - MAP_SIZE / 2) + (y - (MAP_SIZE - 2 * R)) * (y - (MAP_SIZE - 2 * R)) + (z - MAP_SIZE / 2) * (z - MAP_SIZE / 2) <= R * R) {
            world[i].setActive();
            world[i].setColor(rand()%256, rand()%256, rand()%256);
        }
        if (y == MAP_SIZE - 10) {
            world[i].setActive();
            world[i].setColor(0, 255, 0);
        }
        blocksCnt += world[i].isActive();
    }*/
    
    cudaDeviceSynchronize();
    //std::cout << "Num of active voxels: " << blocksCnt << "\n";
    double3 eyePosition = make_double3(64.636510, 1.0, 294.136342);
    cam->eyePosition = eyePosition;
    cam->angleX = 234.833333;
    cam->angleY = -28.666667;
    cam->speed = 5.0;


    sf::Color backgroundColor = sf::Color::Black;
    sf::RenderWindow window(sf::VideoMode(imageHeight, imageWidth), "lol");
    sf::Image image;
    image.create(imageHeight, imageWidth, sf::Color::Magenta);


    double3 *moveStraight;
    cudaMallocManaged(&moveStraight, sizeof(double3));
    double3 *moveNotStraight;
    cudaMallocManaged(&moveNotStraight, sizeof(double3));

    double t = 0.0;
    localLight.x = static_cast<int>(40 * cos(t)) + MAP_SIZE / 2;
    localLight.y = MAP_SIZE / 2;
    localLight.z = static_cast<int>(40 * sin(t)) + MAP_SIZE / 2;

    while (window.isOpen()) {
        frames++;


        world[localLight.x * MAP_SIZE * MAP_SIZE + localLight.y * MAP_SIZE + localLight.z].setInactive();

        localLight.x = static_cast<int>(40 * cos(t)) + MAP_SIZE / 2;
        localLight.z = static_cast<int>(40 * sin(t)) + MAP_SIZE / 2;

        localLight.y = MAP_SIZE / 2;
        t += 0.05;

        light->x = localLight.x * BOX_SIZE + BOX_SIZE / 2.;
        light->y = localLight.y * BOX_SIZE + BOX_SIZE / 2.;
        light->z = localLight.z * BOX_SIZE + BOX_SIZE / 2.;

        world[localLight.x * MAP_SIZE * MAP_SIZE +
            localLight.y * MAP_SIZE +
            localLight.z].setLight();

        SetCursorPos(window.getPosition().x + imageWidth / 2, window.getPosition().y + imageHeight / 2);
        window.sf::Window::setMouseCursorVisible(false);
        dim3 threads(16,16);
        dim3 blocks(imageWidth/threads.x,imageHeight/threads.y);




        traversePixels<<<blocks, threads>>>(screen, cam, world, light);
        //cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        cudaMemcpy(hostScreen, screen, imageHeight * imageWidth * sizeof(uint3), cudaMemcpyDeviceToHost);

        for (int i = 0; i < imageHeight; i++) {
            for (int j = 0; j < imageWidth; j++) {
                image.setPixel(j, i, sf::Color(hostScreen[i * imageWidth + j].x,
                                               hostScreen[i * imageWidth + j].y,
                                               hostScreen[i * imageWidth + j].z));
            }
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        sumScreenRenderTime += milliseconds;

        sf::Event event{};
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
            if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::Escape)
                    window.close();
                if (event.key.code == sf::Keyboard::O)
                    saveVoxelModel(world, MAP_SIZE, MAP_SIZE, MAP_SIZE, std::string("save.dat"));
                if (event.key.code == sf::Keyboard::I)
                    printDebug(cam);
                if (event.key.code == sf::Keyboard::Up) {
                    cam->speed++;
                }
                if (event.key.code == sf::Keyboard::Down) {
                    cam->speed--;
                }

            }
            /*if (event.type == sf::Event::MouseButtonPressed){
                if (event.mouseButton.button == sf::Mouse::Left)
                    deleteVoxel(cam, world);
                if (event.mouseButton.button == sf::Mouse::Right)
                    addVoxel(cam, world);
            }*/
        }



        sf::Texture pixelsTexture;
        pixelsTexture.loadFromImage(image);
        sf::Sprite pixels;
        pixels.setTexture(pixelsTexture, true);

        
        
        if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
            deleteVoxel(cam, world);
        }
        if (sf::Mouse::isButtonPressed(sf::Mouse::Right)) {
            addVoxel(cam, world);
        }
        POINT mousexy;
        GetCursorPos(&mousexy);
        int xt = window.getPosition().x + imageWidth / 2;
        int yt = window.getPosition().y + imageHeight / 2;
        cam->angleX += (xt - mousexy.x) / 6.;
        cam->angleY += (yt - mousexy.y) / 6.;
        SetCursorPos(xt, yt);
        if (cam->angleY > 89.)
            cam->angleY = 89.;
        if (cam->angleY < -89.)
            cam->angleY = -89.;

        buildDirectionOfStraightMove(cam, moveStraight);
        buildDirectionOfNotSoStraightMove(cam, moveNotStraight);
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::W))
            if (bounds(cam->eyePosition + Normalize(moveStraight) * cam->speed))
                cam->eyePosition = cam->eyePosition + Normalize(moveStraight) * cam->speed;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
            if (bounds(cam->eyePosition - Normalize(moveStraight) * cam->speed))
                cam->eyePosition = cam->eyePosition - Normalize(moveStraight) * cam->speed;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
            if (bounds(cam->eyePosition + Normalize(moveNotStraight) * cam->speed))
                cam->eyePosition = cam->eyePosition + Normalize(moveNotStraight) * cam->speed;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
            if (bounds(cam->eyePosition - Normalize(moveNotStraight) * cam->speed))
                cam->eyePosition = cam->eyePosition - Normalize(moveNotStraight) * cam->speed;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::LShift))
            if (bounds(cam->eyePosition + make_double3(0, cam->speed, 0)))
                cam->eyePosition.y += cam->speed;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space))
            if (bounds(cam->eyePosition - make_double3(0, cam->speed, 0)))
                cam->eyePosition.y -= cam->speed;

        window.clear(sf::Color::Magenta);
        window.draw(pixels);
        window.display();
    }
    cudaFree(world);
    cudaFree(screen);
    cudaFree(cam);
    std::cout << "frames: " << frames-1 << std::endl;
    std::cout << "sumScreenRenderTime: " << sumScreenRenderTime << std::endl;
    std::cout << "average fps: " << (frames - 1) * 1000 / sumScreenRenderTime << "\n";
    return 0;
}

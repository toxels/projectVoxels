#include <iostream>
#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>
#include <SFML/Window.hpp>
#include <winuser.h>
#include <Windows.h>
#include "geometry.cuh"
#include <chrono>

// TODO нормальные __host__ __device__
// TODO сделать класс углов || взять из готовой (cuBLAS??)
// TODO текстуры + шрифт
// TODO нормальная либа векторов
// TODO переписать сообщения об ошибках на cuda error
// TODO профилировщик
// TODO тени

//float facingRatio = std::max(0, N.dotProduct(V));

#define M_PI 3.1415
#define MAP_SIZE 100


#define distToViewPort 1.0
#define Vh 1.0
#define Vw 1.0

#define imageHeight 1024
#define imageWidth 1024

#define crossSize 64
#define BOX_SIZE 4

#define MAX_FRAMES 1000
//#define DEBUG




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
unsigned int checkWorld(Box *box, unsigned char *world) {
    if (box->get_x() < 0 || box->get_y() < 0 || box->get_z() < 0 || box->get_x() >= MAP_SIZE ||
        box->get_y() >= MAP_SIZE || box->get_z() >= MAP_SIZE)
        return 0;
    return world[box->get_x() * MAP_SIZE * MAP_SIZE + box->get_y() * MAP_SIZE + box->get_z()];
}


/* (x, y, z) - индексы текущего бокса в world */
__device__
double3 traverseRay(int startX, int startY, int startZ, Ray &ray, int deep, unsigned char *world, Box *lastBox) {
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
            currentBox.get_x() >= MAP_SIZE || currentBox.get_y() >= MAP_SIZE || currentBox.get_z() >= MAP_SIZE /*||
            deep > MAP_SIZE * 2*/) {
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
hitRay(int startX, int startY, int startZ, Ray &ray, int deep, Box &boxToDelete, Box &boxToAdd, unsigned char *world) {
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
void deleteVoxel(Camera *cam, unsigned char *world) {
    Ray hit = computePrimRay(cam, imageWidth / 2, imageHeight / 2);
    Box boxToDelete = Box(0, 0, 0), boxToAdd = Box(0, 0, 0);
    if (hitRay(static_cast<int>(cam->eyePosition.x / BOX_SIZE),
               static_cast<int>(cam->eyePosition.y / BOX_SIZE),
               static_cast<int>(cam->eyePosition.z / BOX_SIZE),
               hit, 5, boxToDelete, boxToAdd, world)) {
        world[boxToDelete.get_x() * MAP_SIZE * MAP_SIZE + boxToDelete.get_y() * MAP_SIZE + boxToDelete.get_z()] = 0;
    }

}

__host__ __device__
void addVoxel(Camera *cam, unsigned char *world) {
    Ray hit = computePrimRay(cam, imageWidth / 2, imageHeight / 2);
    Box boxToAdd = Box(0, 0, 0), boxToDelete = Box(0, 0, 0);
    if (hitRay(static_cast<int>(cam->eyePosition.x / BOX_SIZE),
               static_cast<int>(cam->eyePosition.y / BOX_SIZE),
               static_cast<int>(cam->eyePosition.z / BOX_SIZE),
               hit, 5, boxToDelete, boxToAdd, world)) {
        world[boxToAdd.get_x() * MAP_SIZE * MAP_SIZE + boxToAdd.get_y() * MAP_SIZE + boxToAdd.get_z()] = 1;
    }
}



__global__
void traversePixels(uint3 *screen, Camera *cam, unsigned char *world, double3 *lightSource) {
    __shared__ uint3 temp[512];
    __shared__ double3 firstHitDots[512];
    __shared__ Camera sharedCam;
    __shared__ double3 firstHitDotsNormalized[512];
    sharedCam = *cam;
    double eps = 0.0000001;
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
    Box currBox = Box();

    uint3 color = make_uint3(3, 196, 161); // красиво

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
        color = make_uint3(21, 4, 133);
    } else if (checkWorld(&currBox, world) == 1) {
        __syncthreads();
        double3 dir =  firstHitDots[linearThreadIdxInBlock] - *lightSource;
        Ray shadowRay;
        shadowRay.source = *lightSource;
        shadowRay.direction = dir;
        double3 lastLightHit = traverseRay((static_cast<int>(lightSource->x / BOX_SIZE)),
                                           (static_cast<int>(lightSource->y / BOX_SIZE)),
                                           (static_cast<int>(lightSource->z / BOX_SIZE)), shadowRay, 0, world,
                                           &currBox);
        cudaDeviceSynchronize();
        /*if (firstHitDots[linearThreadIdxInBlock].y / BOX_SIZE == MAP_SIZE - 10)
            color = make_uint3(198, 42, 136);*/
        //cudaDeviceSynchronize();
        if (!(lastLightHit == firstHitDots[linearThreadIdxInBlock])) {
            /** случай когда точка падения полностью в тени */
            color = color * 0.2;
        } else {
            /** случай когда свет дошел до точки */
            /** найти на какой грани лежит точка firstHitDot */
            firstHitDotsNormalized[linearThreadIdxInBlock] =
                firstHitDots[linearThreadIdxInBlock] - make_double3(round(firstHitDots[linearThreadIdxInBlock].x / BOX_SIZE.) * BOX_SIZE,
                                                                    round(firstHitDots[linearThreadIdxInBlock].y / BOX_SIZE.) * BOX_SIZE,
                                                                    round(firstHitDots[linearThreadIdxInBlock].z / BOX_SIZE.) * BOX_SIZE);
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

            double diffuser = (Magnitude(firstHitDots[linearThreadIdxInBlock] - *lightSource));
            cosx = 130000 * cosx / (diffuser * diffuser);

            if (cosx >= eps)
                lightIntensity += cosx;
            
            
            if (lightIntensity > 1.)
                lightIntensity = 1.0;
            color = color * lightIntensity;
            //cudaDeviceSynchronize();
        }
    } else {
        /** куб Валера */
        color = make_uint3(255, 255, 255);
    }
    temp[linearThreadIdxInBlock].x = (static_cast<unsigned char>(color.x));
    temp[linearThreadIdxInBlock].y = (static_cast<unsigned char>(color.y));
    temp[linearThreadIdxInBlock].z = (static_cast<unsigned char>(color.z));
    __syncthreads();
    screen[idx].x = temp[linearThreadIdxInBlock].x;
    screen[idx].y = temp[linearThreadIdxInBlock].y;
    screen[idx].z = temp[linearThreadIdxInBlock].z;

}
/*
std::chrono::milliseconds start_time;
__host__ __device__
void generateMap(unsigned int *world) {
    static double t1 = 0.001, t2 = 0.001;
    std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
    ) - start_time;
    double delta = M_PI;
    int x1 = static_cast<int>(10*cos(t1)) + MAP_SIZE / 2;
    int z1 = static_cast<int>(10*sin(t1)) + MAP_SIZE / 2;
    int y1 = static_cast<int>(t1*3.) + BOX_SIZE * 2;
    int x2 = static_cast<int>(10*cos(-t2-delta)) + MAP_SIZE / 2;
    int z2 = static_cast<int>(10*sin(-t2-delta)) + MAP_SIZE / 2;
    int y2 = static_cast<int>(t2*5.) + BOX_SIZE * 2;
    if((x1 >= MAP_SIZE || y1 >= MAP_SIZE || z1 >= MAP_SIZE) || (x2 >= MAP_SIZE || y2 >= MAP_SIZE || z2 >= MAP_SIZE))
        return;
    int idx = x1 * MAP_SIZE * MAP_SIZE + y1 * MAP_SIZE + z1;
    world[idx] = 1;
    idx = x2 * MAP_SIZE * MAP_SIZE + y2 * MAP_SIZE + z2;
    if(t1 > delta) {
        world[idx] = 1;
        t2 += 0.05;
    }
    t1 += 0.05;
    //for (int i = 0; i < MAP_SIZE * MAP_SIZE * MAP_SIZE; i++)
    //    world[i] = (rand() % 1000 == 0);
    
}
*/
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


    //start_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    int frames = 0;
    std::cout << "I'm gay!" << std::endl;
    float sumScreenRenderTime = 0.0;

    unsigned char *world;
    Camera *cam;
    uint3 *screen;
    double3 *light;
    if (cudaMallocManaged(&world, MAP_SIZE * MAP_SIZE * MAP_SIZE * sizeof(char)))
        fprintf(stderr, "cuda malloc error: world");
    if (cudaMallocManaged(&screen, imageHeight * imageWidth * sizeof(uint3)))
        fprintf(stderr, "cuda malloc error: screen");
    if (cudaMallocManaged(&cam, sizeof(Camera)))
        fprintf(stderr, "cuda malloc error: camera");
    if (cudaMallocManaged(&light, sizeof(double3)))
        fprintf(stderr, "cuda malloc error: light");
    uint3 localLight = make_uint3(MAP_SIZE / 2, 15, MAP_SIZE / 2);
    auto *hostScreen = static_cast<uint3 *>(malloc(imageHeight * imageWidth * sizeof(uint3)));

    int blocksCnt = 0;
    for (int i = 0; i < MAP_SIZE * MAP_SIZE * MAP_SIZE; i++) {
        int x, y, z;
        x = i / MAP_SIZE / MAP_SIZE;
        y = i / MAP_SIZE % MAP_SIZE;
        z = i % MAP_SIZE;
        /*if(x > MAP_SIZE/4 && y > MAP_SIZE/4 && z > MAP_SIZE/4)*/
        if (y == MAP_SIZE - 10)
            world[i] = 1;
        blocksCnt += world[i];
    }
    printf("Num of voxels: %d\n", blocksCnt);

    double3 eyePosition = make_double3(64.636510, 1.0, 294.136342);
    cam->eyePosition = eyePosition;
    cam->angleX = 234.833333;
    cam->angleY = -28.666667;
    cam->speed = 5.0;


    sf::Color backgroundColor = sf::Color::Black;
    sf::RenderWindow window(sf::VideoMode(imageHeight, imageWidth), "lol");
    sf::Image image;
    image.create(imageHeight, imageWidth, sf::Color::Magenta);

    bool drawCross = true;
    sf::Texture crossTexture;
    if (!crossTexture.loadFromFile("cross.png", sf::IntRect(0, 0, crossSize, crossSize))) {
        fprintf(stderr, "Error loading cross.jpg\n");
    }

    // TODO потом перепишем
    double3 *moveStraight;
    cudaMallocManaged(&moveStraight, sizeof(double3));
    double3 *moveNotStraight;
    cudaMallocManaged(&moveNotStraight, sizeof(double3));

    double t = 0.0;
    localLight.x = static_cast<int>(10 * cos(t)) + MAP_SIZE / 2;
    localLight.y = static_cast<int>(10 * sin(t)) + MAP_SIZE / 2;
    localLight.z = 10;

    while (/*frames++ < MAX_FRAMES &&*/ window.isOpen()) {

    

        world[localLight.x * MAP_SIZE * MAP_SIZE +
              localLight.y * MAP_SIZE +
              localLight.z] = 0;

        localLight.x = static_cast<int>(40 * cos(t)) + MAP_SIZE / 2;
        localLight.y = static_cast<int>(40 * sin(t)) + MAP_SIZE / 2;

        localLight.z = 10;
        t += 0.05;

        light->x = localLight.x * BOX_SIZE + BOX_SIZE / 2.;
        light->y = localLight.y * BOX_SIZE + BOX_SIZE / 2.;
        light->z = localLight.z * BOX_SIZE + BOX_SIZE / 2.;

        world[localLight.x * MAP_SIZE * MAP_SIZE +
              localLight.y * MAP_SIZE +
              localLight.z] = 2;

        SetCursorPos(window.getPosition().x + imageWidth / 2, window.getPosition().y + imageHeight / 2);
        window.sf::Window::setMouseCursorVisible(false);
        //int blockSize = 512;
        //int numBlocks = (imageWidth * imageHeight + blockSize - 1) / blockSize;
        dim3 threads(16,16);
        dim3 blocks(imageWidth/threads.x,imageHeight/threads.y);




        //traversePixels<<<numBlocks, blockSize>>>(screen, cam, world, light);
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
                if (event.key.code == sf::Keyboard::V)
                    drawCross = !drawCross;
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

        sf::Sprite crossSprite;
        crossSprite.setTexture(crossTexture);
        crossSprite.setPosition(imageWidth * 4 / 2. - crossSize / 2., imageHeight * 4 / 2. - crossSize / 2.);

        sf::Texture pixelsTexture;
        pixelsTexture.loadFromImage(image);
        sf::Sprite pixels;
        pixels.setTexture(pixelsTexture, true);


        /*sf::Text text;
        text.setFont(font);
        char str[256];
        sprintf(str, "(%.2lf ; %.2lf ; %.2lf)", cam.eyePosition.x, cam.eyePosition.y,   cam.eyePosition.z);
        text.setString(str);
        text.setCharacterSize(16);
        text.setFillColor(sf::Color::Green);*/

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
        /*window.draw(text);*/
        if (drawCross)
            window.draw(crossSprite);
        window.display();
    }
    cudaFree(world);
    cudaFree(screen);
    cudaFree(cam);
    std::cout << "frames: " << frames-1 << std::endl;
    std::cout << "sumScreenRenderTime: " << sumScreenRenderTime << std::endl;
    return 0;
}
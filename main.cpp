#include <iostream>
#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>
#include <SFML/Window.hpp>

#include "geometry.h"

#define MAP_SIZE 100


#define distToViewPort 1.0
#define Vh 1.0
#define Vw 1.0

#define imageHeight 200
#define imageWidth 200

//#define DEBUG

int world[MAP_SIZE][MAP_SIZE][MAP_SIZE];
class Camera{
public:
    Camera() = default;
    Vector3D eyePosition;
    double angleX, angleY;      // углы в градусах*
    Camera(Vector3D& eye, double xAng, double yAng){
        angleX = xAng;
        angleY = yAng;
        eyePosition = eye;
    }
};

class Ray{
public:
    Vector3D source = Vector3D();
    Vector3D direction = Vector3D();
    Ray() = default;
    Ray(Vector3D m_source, Vector3D m_direction){
        source = m_source;
        direction = m_direction;
    }
};

class Box{
    public:
    Box(const Vector3D &min, const Vector3D &max, int X, int Y, int Z) {
        bounds[0] = min; bounds[1] = max;
        x = X; y = Y; z = Z;
    }
    bool intersect(const Ray&, double t0, double t1, Vector3D& inter1Point, Vector3D& inter2Point) const;
    Vector3D bounds[2]{};
    int x, y, z;

};

// Smits’ method
bool Box::intersect(const Ray& r, double t0, double t1, Vector3D& inter1Point, Vector3D& inter2Point) const {
    float tmin, tmax, tymin, tymax, tzmin, tzmax;
    if (r.direction.x >= 0) {
        tmin = (bounds[0].x - r.source.x) / r.direction.x;
        tmax = (bounds[1].x - r.source.x) / r.direction.x;
    }
    else {
        tmin = (bounds[1].x - r.source.x) / r.direction.x;
        tmax = (bounds[0].x - r.source.x) / r.direction.x;
    }
    if (r.direction.y >= 0) {
        tymin = (bounds[0].y - r.source.y) / r.direction.y;
        tymax = (bounds[1].y - r.source.y) / r.direction.y;
    }
    else {
        tymin = (bounds[1].y - r.source.y) / r.direction.y;
        tymax = (bounds[0].y - r.source.y) / r.direction.y;
    }
    if ( (tmin > tymax) || (tymin > tmax) )
        return false;
    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;
    if (r.direction.z >= 0) {
        tzmin = (bounds[0].z - r.source.z) / r.direction.z;
        tzmax = (bounds[1].z - r.source.z) / r.direction.z;
    }
    else {
        tzmin = (bounds[1].z - r.source.z) / r.direction.z;
        tzmax = (bounds[0].z - r.source.z) / r.direction.z;
    }
    if ( (tmin > tzmax) || (tzmin > tmax) )
        return false;
    if (tzmin > tmin)
        tmin = tzmin;
    if (tzmax < tmax)
        tmax = tzmax;
    if(( (tmin < t1) && (tmax > t0) )){
        inter1Point = r.source + r.direction * tmin;
        inter2Point = r.source + r.direction * tmax;
        return true;
    }
    return false;
}
Ray computePrimRay(Camera& cam, int i, int j){
    Vector3D projectPlaneOrigin = cam.eyePosition + Vector3D(0.,0.,distToViewPort);
    Vector3D C = projectPlaneOrigin - Vector3D(Vw / 2., Vh /2., 0);
    Vector3D V = C + Vector3D((double)j / imageWidth * Vw,(double)i / imageHeight * Vh,0.);
    Matrix3D rotate = MakeRotationY((-cam.angleX) * M_PI / 180.) * MakeRotationX(cam.angleY * M_PI / 180.);
    V = rotate * V;
    Vector3D dir = V-cam.eyePosition;
    return Ray(cam.eyePosition, dir);
}
#define BOX_SIZE 8



bool checkWorld(int x, int y, int z){
    if(x < 0 || y < 0 || z < 0 || x >= MAP_SIZE || y >= MAP_SIZE || z >= MAP_SIZE)
        return false;
    return world[x][y][z];
}

/* (x, y, z) - индексы текущего бокса в world */
sf::Color traverseRay(int x, int y, int z, Ray& ray, int deep) {
    double eps = 0.01;
    if(x < 0 || y < 0 || z < 0 || x >= MAP_SIZE || y >= MAP_SIZE || z >= MAP_SIZE || deep > MAP_SIZE)
        return sf::Color::Black;

    /* точки, задающие стартовый бокс */
    Vector3D min = Vector3D(x*BOX_SIZE,
                            y*BOX_SIZE,
                            z*BOX_SIZE);
    Vector3D max = min + Vector3D(BOX_SIZE, BOX_SIZE, BOX_SIZE);
    Box currentBox = Box(min, max, x, y, z);

    /* A1 < A2 : точки пересечения луча и бокса */
    Vector3D A1 = Vector3D(), A2 = Vector3D();
    Box nextBox = currentBox;

    if(currentBox.intersect(ray, -1.0*INFINITY, INFINITY, A1, A2)){
        Vector3D A2_normalized = A2 - min;
        if(abs(A2_normalized.x) < eps)
            nextBox.x--;
        if(abs(A2_normalized.y) < eps)
            nextBox.y--;
        if(abs(A2_normalized.z) < eps)
            nextBox.z--;
        if(abs(A2_normalized.x - BOX_SIZE) < eps)
            nextBox.x++;
        if(abs(A2_normalized.y - BOX_SIZE) < eps)
            nextBox.y++;
        if(abs(A2_normalized.z - BOX_SIZE) < eps)
            nextBox.z++;
        
        /* проверяем nextBox на наличие вокселя */
        if(checkWorld(nextBox.x, nextBox.y, nextBox.z)) {
            return sf::Color::White;
        }
#ifdef DEBUG
        printf("hitBox1: %lf %lf %lf\nhitBox2: %lf %lf %lf\n", A1.x, A1.y, A1.z, A2.x, A2.y, A2.z);
        printf("New box: (%d, %d, %d)\n\n", nextBox.x, nextBox.y, nextBox.z);
#endif
    }
    /* перемещаемся в nextBox, запускаем функцию рекурсивно */
    traverseRay(nextBox.x, nextBox.y, nextBox.z, ray, deep+1);
}

sf::Color screen[imageHeight][imageWidth];
int main() {
    for(auto & i : world){
        for(auto & j : i){
            for(int & k : j)
                k = (rand()%1009 == 0);
        }
    }

    //world[MAP_SIZE / 2][MAP_SIZE / 2][MAP_SIZE / 2] = 1;

    double angleX = 0.0, angleY = 0.0;
    Vector3D eyePosition = Vector3D((MAP_SIZE * BOX_SIZE + 1)/ 2., (MAP_SIZE * BOX_SIZE + 1)/ 2., (MAP_SIZE * BOX_SIZE + 1)/ 2.);

    Camera cam = Camera(eyePosition, angleX, angleY);

    //Ray r = Ray(eyePosition, Vector3D(-1.73, 2.31, -3.47));





    /* Тестовый sfml-код */
    sf::Color backgroundColor = sf::Color::Black;
    sf::RenderWindow window(sf::VideoMode(imageHeight, imageWidth), "lol");
    sf::Image image;
    image.create(imageHeight, imageWidth, sf::Color::Magenta);
    clock_t lastRecompute = clock();
    double dy = 1;
    while (window.isOpen()){
        if(clock() - lastRecompute > 0) {
            for (int i = 0; i < imageHeight; i++) {
                for (int j = 0; j < imageWidth; j++) {
                    Ray currentRay = computePrimRay(cam, i, j);
                    //printf("currRay: %lf %lf %lf\n", currentRay.direction.x, currentRay.direction.y, currentRay.direction.z);
                    screen[i][j] = traverseRay(((int) cam.eyePosition.x / BOX_SIZE),
                                               ((int) cam.eyePosition.y / BOX_SIZE),
                                               ((int) cam.eyePosition.z / BOX_SIZE), currentRay, 0);
                }
            }
            lastRecompute = clock();
        }
        for(int i = 0 ; i < imageHeight ; i++){
            for(int j = 0 ; j < imageWidth ; j++){
                image.setPixel(j, i, screen[i][j]);
            }
        }

        sf::Event event{};
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
            if (event.type == sf::Event::KeyPressed)
                if (event.key.code == sf::Keyboard::Escape)
                    window.close();
        }

        sf::Texture texture;
        texture.loadFromImage(image);
        sf::Sprite sprite;
        sprite.setTexture(texture, true);


        POINT mousexy;
        GetCursorPos(&mousexy);
        int xt = window.getPosition().x + imageWidth / 2;
        int yt = window.getPosition().y + imageHeight / 2;
        //cam.angleX += (xt - mousexy.x) / 6.;
       // cam.angleY += (yt - mousexy.y) / 6.;
       // SetCursorPos(xt,yt);


        if (sf::Keyboard::isKeyPressed(sf::Keyboard::W))
            cam.eyePosition.z += 1;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
            cam.eyePosition.z -= 1;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
            cam.eyePosition.x -= 1;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
            cam.eyePosition.x += 1;

        cam.eyePosition.y += dy;

        if(cam.eyePosition.y < 0)
            dy = 1;
        if(cam.eyePosition.y >= MAP_SIZE * BOX_SIZE)
            dy = -1;
        window.clear(sf::Color::Magenta);
        window.draw(sprite);
        window.display();
    }

    return 0;
}
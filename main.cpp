#include <iostream>
#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>
#include <SFML/Window.hpp>

#include "geometry.h"

#define MAP_SIZE 100


#define distToViewPort 1.0
#define Vh 1.0
#define Vw 1.0

#define imageHeight 100
#define imageWidth 100

#define BOX_SIZE 8

//#define DEBUG

int world[MAP_SIZE][MAP_SIZE][MAP_SIZE];
class Camera{
public:
    Camera() = default;
    Vector3D eyePosition;
    // TODO сделать класс углов || взять из готовой (cuBLAS??)
    double angleX, angleY;
    double speed;
    Camera(Vector3D& eye, double xAng, double yAng){
        angleX = xAng;
        angleY = yAng;
        eyePosition = eye;
        speed = 5;
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
    Box(int X, int Y, int Z) {
        x = X; y = Y; z = Z;
        updateMinMax();
    }
    bool intersect(const Ray&, double t0, double t1, Vector3D& inter1Point, Vector3D& inter2Point) const;
    Vector3D bounds[2]{};
    void set_xyz(int X, int Y, int Z){
        x = X; y = Y; z = Z;
        updateMinMax();
    }
    void inc_x(){
        x++;
        updateMinMax();
    }
    void inc_y(){
        y++;
        updateMinMax();
    }
    void inc_z(){
        z++;
        updateMinMax();
    }
    void dec_x(){
        x--;
        updateMinMax();
    }
    void dec_y(){
        y--;
        updateMinMax();
    }
    void dec_z(){
        z--;
        updateMinMax();
    }
    int get_x(){ return x; }
    int get_y(){ return y; }
    int get_z(){ return z; }

private:
    int x, y, z;
    void updateMinMax(){
        bounds[0] = Vector3D(x*BOX_SIZE, y*BOX_SIZE, z*BOX_SIZE);
        bounds[1] = bounds[0] + Vector3D(BOX_SIZE, BOX_SIZE, BOX_SIZE);
    }
};

// ищет пересечение луча и коробки, записывает точки в interPoint
bool Box::intersect(const Ray& r, double t0, double t1, Vector3D& inter1Point, Vector3D& inter2Point) const {
    double tmin, tmax, tymin, tymax, tzmin, tzmax;
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
Vector3D rayOfView(Camera& cam){
    Vector3D origin = Vector3D(0.,0.,1.);
    Vector3D xRotated = MakeRotationY((cam.angleX) * M_PI / 180.) * Vector3D(1.0, 0.0, 0.0);
    Vector3D result = MakeRotationY((cam.angleX) * M_PI / 180.) * origin;
    result = MakeRotation(-cam.angleY * M_PI / 180., xRotated) * result;
    return result;
}
Ray computePrimRay(Camera& cam, const int i, const int j){
    Vector3D projectPlaneOrigin = Vector3D(0.,0.,distToViewPort);
    Vector3D rightArrow = Vector3D(Vw, 0.0, 0.0), downArrow = Vector3D(0.0, Vh, 0.0);
    Vector3D leftTopCorner = projectPlaneOrigin - rightArrow / 2. - downArrow / 2.;
    Vector3D hitDotOnProjectPlane = leftTopCorner
            + rightArrow * (static_cast<double>(j) / imageWidth)
            +  downArrow * (static_cast<double>(i) / imageHeight);
    Vector3D xRotated = MakeRotationY((cam.angleX) * M_PI / 180.) * Vector3D(1.0, 0.0, 0.0);

    hitDotOnProjectPlane = MakeRotationY((cam.angleX) * M_PI / 180.) * hitDotOnProjectPlane;
    // TODO переписать на cuBLAS, например
    hitDotOnProjectPlane = MakeRotation(-cam.angleY * M_PI / 180., xRotated) * hitDotOnProjectPlane;
    hitDotOnProjectPlane += cam.eyePosition;
    Vector3D dir = hitDotOnProjectPlane - cam.eyePosition;
    return {cam.eyePosition, dir};
}




bool checkWorld(Box& box){
    if(box.get_x() < 0 || box.get_y() < 0 || box.get_z() < 0 || box.get_x() >= MAP_SIZE || box.get_y() >= MAP_SIZE || box.get_z() >= MAP_SIZE)
        return false;
    return world[box.get_x()][box.get_y()][box.get_z()];
}

/* (x, y, z) - индексы текущего бокса в world */
sf::Color traverseRay(int startX, int startY, int startZ, Ray& ray, int deep) {
    const double eps = 0.000001;
    Box currentBox = Box(startX, startY, startZ);
    while(true){
        if(currentBox.get_x() < 0 || currentBox.get_y() < 0 || currentBox.get_z() < 0 ||
                currentBox.get_x() >= MAP_SIZE || currentBox.get_y() >= MAP_SIZE || currentBox.get_z() >= MAP_SIZE ||
                deep > MAP_SIZE * 4)
            return sf::Color::Black;

        /* A1 < A2 : точки пересечения луча и бокса */
        Vector3D A1 = Vector3D(), A2 = Vector3D();

        if(currentBox.intersect(ray, 0, INFINITY, A1, A2)){
            Vector3D A2_normalized = A2 - currentBox.bounds[0];
            if(abs(A2_normalized.x) < eps)
                currentBox.dec_x();
            if(abs(A2_normalized.y) < eps)
                currentBox.dec_y();
            if(abs(A2_normalized.z) < eps)
                currentBox.dec_z();
            if(abs(A2_normalized.x - BOX_SIZE) < eps)
                currentBox.inc_x();
            if(abs(A2_normalized.y - BOX_SIZE) < eps)
                currentBox.inc_y();
            if(abs(A2_normalized.z - BOX_SIZE) < eps)
                currentBox.inc_z();
            /* проверяем бокс на наличие вокселя */
            if(checkWorld(currentBox)) {
                return sf::Color::White;
            }
#ifdef DEBUG
        printf("hitBox1: %lf %lf %lf\nhitBox2: %lf %lf %lf\n", A1.x, A1.y, A1.z, A2.x, A2.y, A2.z);
        printf("New box: (%d, %d, %d)\n\n", nextBox.x, nextBox.y, nextBox.z);
#endif
        }
        deep++;
    }
}

sf::Color screen[imageHeight][imageWidth];
int main() {
    /*for(auto & i : world){
        for(auto & j : i){
            for(int & k : j)
                k = (rand()%1009 == 0);
        }
    }*/

    world[MAP_SIZE / 2][MAP_SIZE / 2][MAP_SIZE / 2] = 1;
    world[MAP_SIZE / 2 + 1][MAP_SIZE / 2][MAP_SIZE / 2] = 1;
    world[MAP_SIZE / 2 + 2][MAP_SIZE / 2][MAP_SIZE / 2] = 1;

    world[MAP_SIZE / 2 - 1][MAP_SIZE / 2 + 1][MAP_SIZE / 2] = 1;
    world[MAP_SIZE / 2 + 3][MAP_SIZE / 2 + 1][MAP_SIZE / 2] = 1;

    world[MAP_SIZE / 2 - 1][MAP_SIZE / 2 + 2][MAP_SIZE / 2] = 1;
    world[MAP_SIZE / 2 + 3][MAP_SIZE / 2 + 2][MAP_SIZE / 2] = 1;
    world[MAP_SIZE / 2][MAP_SIZE / 2 + 2][MAP_SIZE / 2] = 1;
    world[MAP_SIZE / 2 + 2][MAP_SIZE / 2 + 2][MAP_SIZE / 2] = 1;

    world[MAP_SIZE / 2][MAP_SIZE / 2 + 3][MAP_SIZE / 2] = 1;
    world[MAP_SIZE / 2 + 2][MAP_SIZE / 2 + 3][MAP_SIZE / 2] = 1;

    world[MAP_SIZE / 2][MAP_SIZE / 2 + 4][MAP_SIZE / 2] = 1;
    world[MAP_SIZE / 2 + 2][MAP_SIZE / 2 + 4][MAP_SIZE / 2] = 1;

    world[MAP_SIZE / 2][MAP_SIZE / 2 + 5][MAP_SIZE / 2] = 1;
    world[MAP_SIZE / 2 + 2][MAP_SIZE / 2 + 5][MAP_SIZE / 2] = 1;

    world[MAP_SIZE / 2 + 1][MAP_SIZE / 2 + 6][MAP_SIZE / 2] = 1;


    double angleX = 0.0, angleY = 0.0;
    Vector3D eyePosition = Vector3D((MAP_SIZE * BOX_SIZE + 1)/ 2., (MAP_SIZE * BOX_SIZE + 1)/ 2., (MAP_SIZE * BOX_SIZE + 1)/ 2.);
    Camera cam = Camera(eyePosition, angleX, angleY);






    /* Тестовый sfml-код */
    sf::Color backgroundColor = sf::Color::Black;
    sf::RenderWindow window(sf::VideoMode(imageHeight * 4, imageWidth * 4), "lol");
    sf::Image image;
    image.create(imageHeight * 4, imageWidth * 4, sf::Color::Magenta);
    clock_t lastRecompute = clock();

    while (window.isOpen()){
        SetCursorPos(window.getPosition().x + imageWidth / 2, window.getPosition().y + imageHeight / 2);
        window.sf::Window::setMouseCursorVisible(false);
        if(clock() - lastRecompute > 0) {
            for (int i = 0; i < imageHeight; i++) {
                for (int j = 0; j < imageWidth; j++) {
                    Ray currentRay = computePrimRay(cam, i, j);
                    screen[i][j] = traverseRay((static_cast<int>(cam.eyePosition.x / BOX_SIZE)),
                                               (static_cast<int>(cam.eyePosition.y / BOX_SIZE)),
                                               (static_cast<int>(cam.eyePosition.z / BOX_SIZE)), currentRay, 0);
                }
            }
            lastRecompute = clock();
        }
        for(int i = 0 ; i < imageHeight ; i++){
            for(int j = 0 ; j < imageWidth ; j++){
                for(int k = 0 ; k < 4 ; k++)
                    for(int l = 0 ; l < 4 ; l++)
                        image.setPixel(j*4+k,i*4+l,screen[i][j]);
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
        cam.angleX += (xt - mousexy.x) / 6.;
        cam.angleY += (yt - mousexy.y) / 6.;
        SetCursorPos(xt,yt);

        Vector3D move = rayOfView(cam);

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::W))
            cam.eyePosition += Normalize(move) * cam.speed;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
            cam.eyePosition -= Normalize(move) * cam.speed;
        /*if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
            cam.eyePosition.x -= 1;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
            cam.eyePosition.x += 1;*/




        window.clear(sf::Color::Magenta);
        window.draw(sprite);
        window.display();
    }

    return 0;
}
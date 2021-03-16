#pragma once
struct voxel {
    char properties; // 0b00000001 - active or not
                     // 0b00000010 - light or not
    uchar3 color;
    __host__ __device__
    bool isActive() {
        return static_cast<bool>(properties & 0b00000001);
    }
    __host__ __device__
    void setActive() {
        properties |= 0b00000001;
    }
    __host__ __device__
    void setInactive() {
        properties &= 0b11111110;
    }
    __host__ __device__
    void setColor(unsigned char r, unsigned char g, unsigned char b) {
        color.x = r;
        color.y = g;
        color.z = b;
    }
    __host__ __device__
    void setLight() {
        properties |= 0b00000010;
        setActive();
    }
    __host__ __device__
    bool isLight() {
        return static_cast<bool>(properties & 0b00000010);
    }
};
__global__ void iHateMyLife(voxel * thing) {
    thing->setInactive();
}

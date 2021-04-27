#include <vector_types.h>

struct Planet
{
	uint3 coords;
	double speed;
	double phase;
	int radius;
	double orbit;
	uchar3 color;
	Planet(){}
	Planet(uint3 coords, double speed, double phase, int radius, double orbit, uchar3 color)
	{
		this->coords = coords;
		this->speed = speed;
		this->phase = phase;
		this->radius = radius;
		this->orbit = orbit;
		this->color = color;
	}
};

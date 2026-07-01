#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "timer.h" 

#define SOFTENING 0.01f

typedef struct {
    float x, y, z, vx, vy, vz;
} Body;

void bodyForce(Body * p, float dt, int n) {
    int i, j;
    float Fx, Fy, Fz, dx, dy, dz, distSqr, invDist, invDist3;

    for (i = 0; i < n; i++) {
        Fx = 0.0f; Fy = 0.0f; Fz = 0.0f;

        for (j = 0; j < n; j++) {
            dx = p[j].x - p[i].x;
            dy = p[j].y - p[i].y;
            dz = p[j].z - p[i].z;
            distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            invDist = 1.0f / sqrtf(distSqr);
            invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        p[i].vx += dt * Fx;
        p[i].vy += dt * Fy;
        p[i].vz += dt * Fz;
    }
}

void integrate(Body * p, float dt, int n) {
    int i;
    for (i = 0; i < n; i++) {
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

int main(const int argc, const char *argv[]) {
    int num_systems = 32;       
    int bodies_per_system = 8192;
    int nIters = 20;            
    const float dt = 0.01f;
    FILE *fp;
    int total_bodies, bytes, sys, iter;
    Body *data, *system_ptr;
    float *buf;
    double totalTime, interactions_per_system, total_interactions;

    fp = fopen("galaxy_data.bin", "rb");
    if (fp) {
        fread(&num_systems, sizeof(int), 1, fp);
        fread(&bodies_per_system, sizeof(int), 1, fp);
        printf("Found dataset: %d systems of %d bodies.\n", num_systems, bodies_per_system);
    } else {
        printf("No dataset found. Using random initialization.\n");
    }

    total_bodies = num_systems * bodies_per_system;
    bytes = total_bodies * sizeof(Body);
    data = (Body *) malloc(bytes);

    if (fp) {
        fread(data, sizeof(Body), total_bodies, fp);
        fclose(fp);
    } else {
        buf = (float *) data;
        for (int i = 0; i < 6 * total_bodies; i++) {
            buf[i] = 2.0f * (rand() / (float) RAND_MAX) - 1.0f;
        }
    }

    printf("Running OpenMP simulation for %d systems...\n", num_systems);

    double start_time = omp_get_wtime();

    for (iter = 1; iter <= nIters; iter++) {
        #pragma omp parallel for schedule(static) private(system_ptr)
        for (sys = 0; sys < num_systems; sys++) {
            system_ptr = &data[sys * bodies_per_system];
            bodyForce(system_ptr, dt, bodies_per_system);
            integrate(system_ptr, dt, bodies_per_system);
        }
    }

    totalTime = omp_get_wtime() - start_time;

    interactions_per_system = (double) bodies_per_system * bodies_per_system;
    total_interactions = interactions_per_system * num_systems * nIters;

    printf("Total Time: %.3f seconds\n", totalTime);
    printf("Average Throughput: %0.3f Billion Interactions / second\n",
           1e-9 * total_interactions / totalTime);

    printf("Final position of System 0, Body 0: %.4f, %.4f, %.4f\n",
           data[0].x, data[0].y, data[0].z);
    printf("Final position of System 0, Body 1: %.4f, %.4f, %.4f\n",
           data[1].x, data[1].y, data[1].z);

    free(data);
    return 0;
}
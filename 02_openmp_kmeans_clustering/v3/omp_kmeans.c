/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         seq_kmeans.c  (sequential version)                        */
/*   Description:  Implementation of simple k-means clustering algorithm     */
/*                 This program takes an array of N data objects, each with  */
/*                 M coordinates and performs a k-means clustering given a   */
/*                 user-provided value of the number of clusters (K). The    */
/*                 clustering results are saved in 2 arrays:                 */
/*                 1. a returned array of size [K][N] indicating the center  */
/*                    coordinates of K clusters                              */
/*                 2. membership[N] stores the cluster center ids, each      */
/*                    corresponding to the cluster a data object is assigned */
/*                                                                           */
/*   Author:  Wei-keng Liao                                                  */
/*            ECE Department, Northwestern University                        */
/*            email: wkliao@ece.northwestern.edu                             */
/*                                                                           */
/*   Copyright (C) 2005, Northwestern University                             */
/*   See COPYRIGHT notice in top-level directory.                            */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "kmeans.h"


/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__inline static
float euclid_dist_2(int    numdims,  /* no. dimensions */
                    float *coord1,   /* [numdims] */
                    float *coord2)   /* [numdims] */
{
    int i;
    float ans=0.0;

    for (i=0; i<numdims; i++)
        ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);

    return(ans);
}

/*----< find_nearest_cluster() >---------------------------------------------*/
__inline static
int find_nearest_cluster(int     numClusters, /* no. clusters */
                         int     numCoords,   /* no. coordinates */
                         float  *object,      /* [numCoords] */
                         float **clusters)    /* [numClusters][numCoords] */
{
    int   index, i;
    float dist, min_dist;

    /* find the cluster id that has min distance to object */
    index    = 0;
    min_dist = euclid_dist_2(numCoords, object, clusters[0]);

    for (i=1; i<numClusters; i++) {
        dist = euclid_dist_2(numCoords, object, clusters[i]);
        /* no need square root */
        if (dist < min_dist) { /* find the min and its array index */
            min_dist = dist;
            index    = i;
        }
    }
    return(index);
}

/*----< seq_kmeans() >-------------------------------------------------------*/
int omp_kmeans(float **objects,      /* in: [numObjs][numCoords] */
               int     numCoords,    /* no. features */
               int     numObjs,      /* no. objects */
               int     numClusters,  /* no. clusters */
               float   threshold,    /* % objects change membership */
               int    *membership,   /* out: [numObjs] */
               float **clusters)     /* out: [numClusters][numCoords] */
{
    int      i, j, k, t, loop=0, index;
    int     *newClusterSize; /* Global final sums */
    float  **newClusters;    /* Global final sums */
    float    delta;          /* Global delta */
    int      nthreads;       /* Number of threads */
    int    **all_local_ClusterSize; /* Shared array for thread-local counts */
    float  **all_local_Clusters_flat; /* Shared array for thread-local sums (flat) */

    /* Get max threads and allocate shared storage for local data */
    #pragma omp parallel
    {
        #pragma omp master
        nthreads = omp_get_num_threads();
    } // Implicit barrier

    all_local_ClusterSize = (int**) malloc(nthreads * sizeof(int*));
    assert(all_local_ClusterSize != NULL);
    int* all_local_ClusterSize_cont = (int*) calloc(nthreads * numClusters, sizeof(int));
    assert(all_local_ClusterSize_cont != NULL);
    for(t=0; t<nthreads; t++)
        all_local_ClusterSize[t] = all_local_ClusterSize_cont + t * numClusters;


    all_local_Clusters_flat = (float**) malloc(nthreads * sizeof(float*));
    assert(all_local_Clusters_flat != NULL);
    float* all_local_Clusters_cont = (float*) calloc(nthreads * numClusters * numCoords, sizeof(float));
    assert(all_local_Clusters_cont != NULL);
     for(t=0; t<nthreads; t++)
        all_local_Clusters_flat[t] = all_local_Clusters_cont + t * numClusters * numCoords;


    /* Allocate memory for common arrays (as before) */
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);
    newClusters    = (float**) malloc(numClusters * sizeof(float*));
    assert(newClusters != NULL);
    newClusters[0] = (float*)  calloc(numClusters * numCoords, sizeof(float));
    assert(newClusters[0] != NULL);
    for (i=1; i<numClusters; i++)
        newClusters[i] = newClusters[i-1] + numCoords;

    /* Initialize membership */
    for (i=0; i<numObjs; i++) membership[i] = -1;

    /* ================================================================== */
    /* >>> START OF THE SINGLE PARALLEL REGION <<< */
    /* ================================================================== */
    #pragma omp parallel \
            shared(objects, clusters, membership, newClusters, newClusterSize, delta, loop, \
                   all_local_ClusterSize, all_local_Clusters_flat, nthreads) \
            private(i, j, k, t, index)
    {
        int tid = omp_get_thread_num();
        int* local_newClusterSize = all_local_ClusterSize[tid]; // My slice
        float* local_newClusters_flat = all_local_Clusters_flat[tid]; // My slice
        
        float** local_newClusters = (float**) malloc(numClusters * sizeof(float*));
        assert(local_newClusters != NULL);
        for(k=0; k<numClusters; k++)
             local_newClusters[k] = local_newClusters_flat + k * numCoords;

        float local_delta; // Thread-private delta

        do {
            local_delta = 0.0;

            // Zero out MY thread-local arrays
            memset(local_newClusterSize, 0, numClusters * sizeof(int));
            memset(local_newClusters_flat, 0, numClusters * numCoords * sizeof(float));
            // Implicit barrier before the first worksharing loop (#omp for)

            /* === Hotspot 1: Update MY LOCAL sums (NO ATOMICS) === */
            #pragma omp for schedule(static) private(j, index) nowait
            for (i=0; i<numObjs; i++) {
                index = find_nearest_cluster(numClusters, numCoords, objects[i], clusters);

                if (membership[i] != index) {
                    local_delta += 1.0; // Update private delta
                    membership[i] = index;
                }

                local_newClusterSize[index]++;
                for (j=0; j<numCoords; j++)
                    local_newClusters_flat[index * numCoords + j] += objects[i][j];
            }
            // NO implicit barrier because of nowait

            // --- Manual Reduction for delta ---
            #pragma omp barrier // Wait for all threads to finish loop & local_delta calculation
            #pragma omp single
            { delta = 0.0; } // Reset global delta once
            #pragma omp atomic // Each thread adds its computed local_delta safely
            delta += local_delta;
            #pragma omp barrier // Ensure delta is fully summed before proceeding

            /* === PARALLEL REDUCTION STEP (Over Clusters) === */
            #pragma omp for schedule(static) private(j, t)
            for (i = 0; i < numClusters; ++i) { // i represents cluster index 'c'
                int totalSize = 0;
                // Sum sizes for cluster 'i' across all threads 't'
                for (t = 0; t < nthreads; ++t) {
                    totalSize += all_local_ClusterSize[t][i];
                }
                newClusterSize[i] = totalSize; // Update global size

                // Sum coordinates for cluster 'i' across all threads 't'
                for (j = 0; j < numCoords; ++j) {
                    float temp_sum_coord = 0.0f;
                    for (t = 0; t < nthreads; ++t) {
                        temp_sum_coord += all_local_Clusters_flat[t][i * numCoords + j];
                    }
                    newClusters[i][j] = temp_sum_coord; // Update global sum
                }
            }
            // Implicit barrier after the parallel reduction loop

            /* === Hotspot 2: Calculate new centers (using global sums) === */
            #pragma omp for schedule(static) private(j)
            for (i=0; i<numClusters; i++) {
                if (newClusterSize[i] > 0) {
                     for (j=0; j<numCoords; j++) {
                        clusters[i][j] = newClusters[i][j] / newClusterSize[i];
                    }
                }
            }
            // Implicit barrier

            /* Check convergence condition */
            #pragma omp master
            {
                delta /= numObjs;
                loop++;
            }
            #pragma omp barrier // Ensure all threads see updated delta/loop

        } while (delta > threshold && loop < 500);

        free(local_newClusters);

    } /* >>> END OF THE SINGLE PARALLEL REGION <<< */
    /* ================================================================== */

    /* Clean up globally allocated shared storage */
    free(all_local_ClusterSize_cont);
    free(all_local_ClusterSize);
    free(all_local_Clusters_cont);
    free(all_local_Clusters_flat);

    /* Clean up global result arrays */
    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    return 1;
}


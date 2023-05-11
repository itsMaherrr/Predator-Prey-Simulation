
from pycuda.compiler import SourceModule

kernel_code = """
                #define PI  3.14159265358979323846

                __global__ void surviving_preys(float *preys, float *predators, int *collision, int radius, int predators_num) {
                    const int i = blockIdx.x;
                    const int j = threadIdx.x;
                    const int prey_x = preys[2*i];
                    const int prey_y = preys[2*i + 1];
                    const int predator_x = predators[2*j];
                    const int predator_y = predators[2*j + 1];
                    float distance = sqrt(pow((prey_x - predator_x), 2) + pow((prey_y - predator_y), 2));
                    if (distance <= (radius * 2)) {
                        collision[(i * predators_num + j) * 2] = 1;
                    }
                }
                
                __global__ void find_angles(double *predators, double *entities, int *result, int NPredators, int NEntities) {
                    const int i = blockIdx.x * blockDim.x + threadIdx.x;
                    const int predator_idx = i / NEntities;
                    const int entity_idx = (blockIdx.x * blockDim.x + threadIdx.x) - NEntities * predator_idx;
                    double pred_vision_angle = predators[3*predator_idx + 2];
                    const double vector_x = cos(pred_vision_angle);
                    const double vector_y = sin(pred_vision_angle);
                    if (predator_idx < NPredators) {
                        const int entity_x = entities[2 * entity_idx];
                        const int entity_y = entities[2 * entity_idx + 1];
                        const int predator_x = predators[3 * predator_idx];
                        const int predator_y = predators[3 * predator_idx + 1];
                        double pred_ent_vector_x = entity_x - predator_x;
                        double pred_ent_vector_y = entity_y - predator_y;
                        float distance = sqrt(pow(pred_ent_vector_x, 2) + pow(pred_ent_vector_y, 2));
                        double pred_ent_angle = atan2(vector_x * pred_ent_vector_y - vector_y * pred_ent_vector_x, vector_x * pred_ent_vector_x + vector_y * pred_ent_vector_y);
                        pred_ent_angle = pred_ent_angle * (double) 180.0 / (double) PI;
                        result[2*i] = (__double2int_rn(pred_ent_angle) + 360) % 360;
                        result[2*i + 1] = __double2int_rn(distance);
                    }
                }
        """

mod = SourceModule(kernel_code)

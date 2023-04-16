
from pycuda.compiler import SourceModule

kernel_code = """
                __global__ void surviving_preys(float *preys, float *predators, int *fed_predators, int *survived_preys, int radius) {
                    const int i = blockIdx.x;
                    const int j = threadIdx.x;
                    const int prey_x = preys[2*i];
                    const int prey_y = preys[2*i + 1];
                    const int predator_x = predators[2*j];
                    const int predator_y = predators[2*j + 1];
                    float distance = sqrt(pow((prey_x - predator_x), 2) + pow((prey_y - predator_y), 2));
                    if (distance <= (radius * 2)) {
                        survived_preys[2*i + 1] = 0;
                        fed_predators[j]++;
                    }
                }
                
                __global__ void find_angles(float *predators, float *entities,float *result, int NPredators, int NEntities) {
                    const int i = blockIdx.x * blockDim.x + threadIdx.x;
                    const int predator_idx = i / NEntities;
                    const int entity_idx = (blockIdx.x * blockDim.x + threadIdx.x) - NEntities * predator_idx;
                    float pred_vision_angle = predators[3*predator_idx + 2];
                    const double vector[2] = {cos(pred_vision_angle), sin(pred_vision_angle)};
                    if (predator_idx < NPredators) {
                        const int entity_x = entities[2 * entity_idx];
                        const int entity_y = entities[2 * entity_idx + 1];
                        const int predator_x = predators[3 * predator_idx];
                        const int predator_y = predators[3 * predator_idx + 1];
                        double pred_ent_vector[2] = {(double) entity_x - predator_x, (double) entity_y - predator_y};
                        float distance = sqrt(pow(pred_ent_vector[0], 2) + pow(pred_ent_vector[1], 2));
                        float pred_ent_angle = (float) atan2(vector[0]*pred_ent_vector[1]-vector[1]*pred_ent_vector[0], vector[0]*pred_ent_vector[0]+vector[1]*pred_ent_vector[1]);
                        pred_ent_angle = pred_ent_angle * 180.0 / (22/7);
                        result[2*i] = pred_ent_angle;
                        result[2*i + 1] = distance;
                    }
                }
        """

mod = SourceModule(kernel_code)
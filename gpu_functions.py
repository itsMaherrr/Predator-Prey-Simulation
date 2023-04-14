
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
        """

mod = SourceModule(kernel_code)
import os
import time

import tensorflow as tf

# Optional: reduce log clutter
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("TensorFlow version:", tf.__version__)

# List physical devices
gpus = tf.config.list_physical_devices("GPU")
cpus = tf.config.list_physical_devices("CPU")
print("GPUs:", gpus)
print("CPUs:", cpus)

# Check if Metal GPU is available
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Metal GPU memory growth set successfully.")
    except Exception as e:
        print("Could not set GPU memory growth:", e)

# Simple tensor operation
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[1.0, 1.0], [0.0, 1.0]])

# CPU timing
start_time = time.time()
with tf.device("/CPU:0"):
    c_cpu = tf.matmul(a, b)
cpu_time = time.time() - start_time
print("CPU matmul result:\n", c_cpu)
print(f"CPU execution time: {cpu_time:.6f} seconds")

# GPU timing (if available)
if gpus:
    start_time = time.time()
    with tf.device("/GPU:0"):
        c_gpu = tf.matmul(a, b)
    gpu_time = time.time() - start_time
    print("GPU matmul result:\n", c_gpu)
    print(f"GPU execution time: {gpu_time:.6f} seconds")

# Compare with default placement
start_time = time.time()
c_default = tf.matmul(a, b)
default_time = time.time() - start_time
print("Default placement matmul result:\n", c_default)
print(f"Default placement execution time: {default_time:.6f} seconds")


# Compare with default placement
start_time = time.time()
c_default = tf.matmul(a, b)
default_time = time.time() - start_time
print("Default placement matmul result:\n", c_default)
print(f"Default placement execution time: {default_time:.6f} seconds")

# Compare with default placement
start_time = time.time()
c_default = tf.matmul(a, b)
default_time = time.time() - start_time
print("Default placement matmul result:\n", c_default)
print(f"Default placement execution time: {default_time:.6f} seconds")


# Compare with default placement
start_time = time.time()
c_default = tf.matmul(a, b)
default_time = time.time() - start_time
print("Default placement matmul result:\n", c_default)
print(f"Default placement execution time: {default_time:.6f} seconds")


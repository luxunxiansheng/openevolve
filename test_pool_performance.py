#!/usr/bin/env python3
"""
Performance test for RayTaskPool
"""

import time
import math
from typing import List

from openevolve.utils.ray_task_pool import RayTaskPool, RAY_AVAILABLE


def cpu_intensive_task(n: int) -> float:
    """CPU-intensive task for testing"""
    result = 0.0
    for i in range(n * 1000):
        result += math.sqrt(i + 1) * math.sin(i)
    return result


async def async_cpu_task(n: int) -> float:
    """Async wrapper for CPU task"""
    return cpu_intensive_task(n)


def io_simulation_task(delay: float) -> str:
    """Simulate I/O bound task"""
    time.sleep(delay)
    return f"Task completed after {delay}s"


async def async_io_task(delay: float) -> str:
    """Async I/O simulation"""
    await asyncio.sleep(delay)
    return f"Async task completed after {delay}s"


async def test_taskpool_cpu(num_tasks: int, task_size: int, concurrency: int) -> tuple:
    """Test TaskPool with CPU-intensive tasks"""
    print(f"Testing TaskPool: {num_tasks} CPU tasks, size={task_size}, concurrency={concurrency}")

    pool = TaskPool(max_concurrency=concurrency)
    start_time = time.time()

    # Create tasks
    tasks = [pool.create_task(async_cpu_task, task_size) for _ in range(num_tasks)]

    # Wait for completion
    results = await asyncio.gather(*tasks)

    end_time = time.time()
    duration = end_time - start_time

    print(f"TaskPool CPU: {duration:.2f}s, {len(results)} results")
    return duration, len(results)


async def test_taskpool_io(num_tasks: int, delay: float, concurrency: int) -> tuple:
    """Test TaskPool with I/O tasks"""
    print(f"Testing TaskPool: {num_tasks} I/O tasks, delay={delay}s, concurrency={concurrency}")

    pool = TaskPool(max_concurrency=concurrency)
    start_time = time.time()

    # Create tasks
    tasks = [pool.create_task(async_io_task, delay) for _ in range(num_tasks)]

    # Wait for completion
    results = await asyncio.gather(*tasks)

    end_time = time.time()
    duration = end_time - start_time

    print(f"TaskPool I/O: {duration:.2f}s, {len(results)} results")
    return duration, len(results)


def test_raytaskpool_cpu(num_tasks: int, task_size: int, concurrency: int) -> tuple:
    """Test RayTaskPool with CPU-intensive tasks"""
    if not RAY_AVAILABLE:
        print("Ray not available, skipping RayTaskPool tests")
        return 0.0, 0

    print(
        f"Testing RayTaskPool: {num_tasks} CPU tasks, size={task_size}, concurrency={concurrency}"
    )

    pool = RayTaskPool(max_concurrency=concurrency)
    start_time = time.time()

    # Submit tasks
    for _ in range(num_tasks):
        pool.submit(cpu_intensive_task, task_size)

    # Get results
    results = pool.get_results()

    end_time = time.time()
    duration = end_time - start_time

    print(f"RayTaskPool CPU: {duration:.2f}s, {len(results)} results")
    return duration, len(results)


def test_raytaskpool_io(num_tasks: int, delay: float, concurrency: int) -> tuple:
    """Test RayTaskPool with I/O simulation tasks"""
    if not RAY_AVAILABLE:
        print("Ray not available, skipping RayTaskPool tests")
        return 0.0, 0

    print(f"Testing RayTaskPool: {num_tasks} I/O tasks, delay={delay}s, concurrency={concurrency}")

    pool = RayTaskPool(max_concurrency=concurrency)
    start_time = time.time()

    # Submit tasks
    for _ in range(num_tasks):
        pool.submit(io_simulation_task, delay)

    # Get results
    results = pool.get_results()

    end_time = time.time()
    duration = end_time - start_time

    print(f"RayTaskPool I/O: {duration:.2f}s, {len(results)} results")
    return duration, len(results)


async def run_performance_tests():
    """Run comprehensive performance tests"""
    print("=== Task Pool Performance Comparison ===\n")

    # Test configurations
    configs = [
        {"num_tasks": 10, "concurrency": 4, "task_size": 100, "io_delay": 0.1},
        {"num_tasks": 20, "concurrency": 8, "task_size": 200, "io_delay": 0.05},
        {"num_tasks": 50, "concurrency": 10, "task_size": 50, "io_delay": 0.02},
    ]

    results = []

    for i, config in enumerate(configs, 1):
        print(f"\n--- Test Configuration {i} ---")
        print(f"Tasks: {config['num_tasks']}, Concurrency: {config['concurrency']}")

        # CPU-intensive tests
        print("\n** CPU-intensive tasks **")
        taskpool_cpu_time, taskpool_cpu_count = await test_taskpool_cpu(
            config["num_tasks"], config["task_size"], config["concurrency"]
        )

        raytaskpool_cpu_time, raytaskpool_cpu_count = test_raytaskpool_cpu(
            config["num_tasks"], config["task_size"], config["concurrency"]
        )

        # I/O simulation tests
        print("\n** I/O simulation tasks **")
        taskpool_io_time, taskpool_io_count = await test_taskpool_io(
            config["num_tasks"], config["io_delay"], config["concurrency"]
        )

        raytaskpool_io_time, raytaskpool_io_count = test_raytaskpool_io(
            config["num_tasks"], config["io_delay"], config["concurrency"]
        )

        # Store results
        results.append(
            {
                "config": config,
                "taskpool_cpu": taskpool_cpu_time,
                "raytaskpool_cpu": raytaskpool_cpu_time,
                "taskpool_io": taskpool_io_time,
                "raytaskpool_io": raytaskpool_io_time,
            }
        )

        # Print comparison
        if RAY_AVAILABLE and raytaskpool_cpu_time > 0:
            cpu_speedup = taskpool_cpu_time / raytaskpool_cpu_time
            io_speedup = taskpool_io_time / raytaskpool_io_time
            print(f"\nSpeedup - CPU: {cpu_speedup:.2f}x, I/O: {io_speedup:.2f}x")

        print("-" * 50)

    # Summary
    print("\n=== SUMMARY ===")
    if RAY_AVAILABLE:
        cpu_speedups = []
        io_speedups = []

        for result in results:
            if result["raytaskpool_cpu"] > 0:
                cpu_speedups.append(result["taskpool_cpu"] / result["raytaskpool_cpu"])
            if result["raytaskpool_io"] > 0:
                io_speedups.append(result["taskpool_io"] / result["raytaskpool_io"])

        if cpu_speedups:
            print(f"Average CPU speedup (Ray/TaskPool): {statistics.mean(cpu_speedups):.2f}x")
        if io_speedups:
            print(f"Average I/O speedup (Ray/TaskPool): {statistics.mean(io_speedups):.2f}x")
    else:
        print("Ray not available - only TaskPool results shown")


if __name__ == "__main__":
    asyncio.run(run_performance_tests())

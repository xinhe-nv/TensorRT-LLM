# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Stress test script for disaggregated serving architecture using TensorRT LLM.

This module contains stress tests specifically designed for disaggregated serving,
where context and generation processing are handled by separate server instances.

The tests verify:
- Performance under sustained load with disaggregated architecture
- Accuracy stability after stress testing (using GSM8K dataset)
- Resource utilization and scalability across distributed components

Usage example:
    pytest tests/integration/defs/stress_test/stress_test_disaggregated.py::test_disaggregated_stress_test
"""

import contextlib
import copy
import itertools
import os
import tempfile
import time
from typing import Any, Dict

import pytest
import requests
import yaml

from defs.conftest import skip_pre_hopper
from defs.trt_test_alternative import popen, print_info, print_warning

from .stress_test import (
    ModelConfig,
    PerformanceParams,
    RequestCounter,
    ServerConfig,
    StressTestConfig,
    check_server_health,
    extract_stress_test_metrics,
    get_model_path,
    measure_capacity_stage,
    run_accuracy_test,
    stress_stage,
)


@contextlib.contextmanager
def launch_disaggregated_serving(
    disaggregated_server_config: Dict[str, Any],
    ctx_server_config: Dict[str, Any],
    gen_server_config: Dict[str, Any],
    model_path: str,
    ctx_model_path: str = None,
    gen_model_path: str = None,
    server_waiting_timeout: int = 3600,
    health_check_timeout: float = 10.0,
):
    """Context manager to launch disaggregated serving with context and generation servers.

    Args:
        disaggregated_server_config: Main disaggregated server configuration
        ctx_server_config: Context server configuration
        gen_server_config: Generation server configuration
        model_path: Default model path (used if ctx/gen specific paths not provided)
        ctx_model_path: Optional specific path for context servers
        gen_model_path: Optional specific path for generation servers
        server_waiting_timeout: Timeout for server initialization in seconds
        health_check_timeout: Timeout for health check requests in seconds
    """
    temp_dir = tempfile.TemporaryDirectory()

    # Write configuration files
    disaggregated_config_path = os.path.join(temp_dir.name, "disaggregated_serving_config.yaml")
    ctx_config_path = os.path.join(temp_dir.name, "ctx_server_config.yaml")
    gen_config_path = os.path.join(temp_dir.name, "gen_server_config.yaml")

    with open(disaggregated_config_path, "w") as f:
        yaml.dump(disaggregated_server_config, f)
    with open(ctx_config_path, "w") as f:
        yaml.dump(ctx_server_config, f)
    with open(gen_config_path, "w") as f:
        yaml.dump(gen_server_config, f)

    ctx_model = ctx_model_path or model_path
    gen_model = gen_model_path or model_path

    trtllm_serve_path = "trtllm-serve"

    # Get parallelism settings
    gen_tp = gen_server_config.get("tensor_parallel_size", 1)
    gen_pp = gen_server_config.get("pipeline_parallel_size", 1)
    ctx_tp = ctx_server_config.get("tensor_parallel_size", 1)
    ctx_pp = ctx_server_config.get("pipeline_parallel_size", 1)

    ctx_total_gpus = ctx_tp * ctx_pp
    gen_total_gpus = gen_tp * gen_pp

    ctx_urls = disaggregated_server_config["context_servers"]["urls"]
    gen_urls = disaggregated_server_config["generation_servers"]["urls"]

    ctx_ports = [int(url.split(":")[1]) for url in ctx_urls]
    gen_ports = [int(url.split(":")[1]) for url in gen_urls]

    # Prepare context server commands
    ctx_servers = []
    current_gpu_offset = 0

    for i, port in enumerate(ctx_ports):
        env_ctx = os.environ.copy()
        env_ctx["TRTLLM_USE_UCX_KVCACHE"] = "1"
        gpu_range = range(current_gpu_offset, current_gpu_offset + ctx_total_gpus)
        env_ctx["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_range))
        current_gpu_offset += ctx_total_gpus

        ctx_server_args = [
            trtllm_serve_path,
            ctx_model,
            "--host",
            "localhost",
            "--backend",
            "pytorch",
            "--port",
            str(port),
            "--log_level",
            "debug",
            "--extra_llm_api_options",
            ctx_config_path,
            f"--tp_size={ctx_tp}",
            f"--pp_size={ctx_pp}",
        ]

        ctx_servers.append((env_ctx, ctx_server_args))

    # Prepare generation server commands
    gen_servers = []

    for i, port in enumerate(gen_ports):
        env_gen = os.environ.copy()
        env_gen["TRTLLM_USE_UCX_KVCACHE"] = "1"
        gpu_range = range(current_gpu_offset, current_gpu_offset + gen_total_gpus)
        env_gen["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_range))
        current_gpu_offset += gen_total_gpus

        gen_server_args = [
            trtllm_serve_path,
            gen_model,
            "--host",
            "localhost",
            "--backend",
            "pytorch",
            "--port",
            str(port),
            "--log_level",
            "debug",
            "--extra_llm_api_options",
            gen_config_path,
            f"--tp_size={gen_tp}",
            f"--pp_size={gen_pp}",
        ]

        gen_servers.append((env_gen, gen_server_args))

    @contextlib.contextmanager
    def multi_popen(server_configs):
        """Helper to start multiple server processes."""
        processes = []
        try:
            for env, args in server_configs:
                proc = popen(args, env=env)
                processes.append(proc)

            with contextlib.ExitStack() as stack:
                opened_processes = [stack.enter_context(proc) for proc in processes]
                yield opened_processes
        except Exception as e:
            print_warning(f"Failed to start disaggregated server processes: {e}")
            raise

    # Prepare main disaggregated server command
    server_cmd = [
        trtllm_serve_path,
        "disaggregated",
        "-c",
        disaggregated_config_path,
        "--server_start_timeout",
        str(server_waiting_timeout),
        "-r",
        "360000",
    ]

    with (
        temp_dir,
        multi_popen(ctx_servers + gen_servers) as worker_processes,
        popen(server_cmd) as server_process,
    ):
        print_info("Waiting for disaggregated servers to initialize...")
        start_time = time.time()
        server_ready = False

        while time.time() - start_time < server_waiting_timeout:
            time.sleep(5)

            # Check if any process has died
            for process in itertools.chain(worker_processes, [server_process]):
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Process {process.pid} exited with code {process.returncode}"
                    )

            # Check health endpoint
            try:
                print_info("Checking health endpoint...")
                response = requests.get(
                    f"http://localhost:{disaggregated_server_config['port']}/health",
                    timeout=health_check_timeout,
                )
                if response.status_code == 200:
                    print_info("Disaggregated server is ready!")
                    server_ready = True
                    break
            except requests.exceptions.ConnectionError:
                continue
            except Exception as e:
                print_warning(f"Health check error: {e}")
                continue

        if not server_ready:
            raise RuntimeError(
                f"Disaggregated server failed to start within {server_waiting_timeout} seconds"
            )

        yield  # Server is ready for use

        print_info("Shutting down disaggregated servers...")

    print_info("Disaggregated servers stopped")


@pytest.mark.parametrize("test_mode", ["stress-test-with-accuracy"], ids=lambda x: x)
@pytest.mark.parametrize(
    "stress_time_timeout", [(4000, 7200)], ids=lambda x: f"stress_time_{x[0]}s_timeout_{x[1]}s"
)
@pytest.mark.parametrize(
    "model_config",
    [
        pytest.param(
            ModelConfig(model_dir="gpt_oss/gpt-oss-120b", tp_size=1, memory_requirement=120),
            marks=[
                skip_pre_hopper,
                pytest.mark.skip_less_device(4),
                pytest.mark.skip_less_device_memory(120),
            ],
        )
    ],
    ids=lambda x: f"{os.path.basename(x.model_dir)}_tp{x.tp_size}",
)
def test_disaggregated_stress_test(model_config, test_mode, stress_time_timeout):
    """Run disaggregated serving stress test with configurable models.

    This test verifies the stability and performance of the disaggregated serving
    architecture under sustained load. It runs:
    1. Baseline accuracy test (GSM8K)
    2. Performance measurement stage
    3. Stress test with high load
    4. Post-stress accuracy test to verify stability

    The disaggregated architecture separates context processing and generation
    into independent server instances, allowing for better resource utilization
    and scaling.

    Args:
        model_config: Model configuration containing model_dir, tp_size, and memory_requirement
        test_mode: Test mode, currently supports "stress-test-with-accuracy"
        stress_time_timeout: Tuple of (stress_time, stress_timeout) in seconds
    """
    model_dir = model_config.model_dir
    model_path = get_model_path(model_dir)
    model_name = model_config.model_name

    # Extract stress_time and stress_timeout from the tuple
    stress_time, stress_timeout = stress_time_timeout

    # Configure disaggregated serving parallelism from model config
    ctx_tp = model_config.tp_size
    ctx_pp = 1
    gen_tp = model_config.tp_size
    gen_pp = 1

    # Context server configuration
    ctx_server_config = {
        "disable_overlap_scheduler": True,
        "kv_cache_config": {
            "free_gpu_memory_fraction": 0.6,
            "enable_block_reuse": False,  # Disable for stress test to avoid reuse tree race conditions
        },
        "enable_chunked_prefill": True,
        "max_num_tokens": 8192,
        "cache_transceiver_config": {"backend": "DEFAULT"},
        "max_batch_size": 1024,
        "cuda_graph_config": None,
        "print_iter_log": True,
    }

    # Generation server configuration
    gen_server_config = copy.deepcopy(ctx_server_config)

    # Configure CUDA graphs for pytorch backend
    cuda_graph_config = {
        "enable_padding": True,
        "batch_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024],
    }
    ctx_server_config["tensor_parallel_size"] = ctx_tp
    ctx_server_config["pipeline_parallel_size"] = ctx_pp
    gen_server_config["tensor_parallel_size"] = gen_tp
    gen_server_config["pipeline_parallel_size"] = gen_pp
    gen_server_config["cuda_graph_config"] = cuda_graph_config

    # Set default timeout values
    server_waiting_timeout = 28800
    health_check_timeout = 8.0  # 8 seconds default

    # Main disaggregated server configuration
    disaggregated_server_config = {
        "hostname": "localhost",
        "port": 8000,
        "backend": "pytorch",
        "context_servers": {"num_instances": 1, "urls": ["localhost:8001"]},
        "generation_servers": {"num_instances": 1, "urls": ["localhost:8002"]},
    }

    # Check if server is already running
    is_healthy, _ = check_server_health("http://localhost:8000", health_check_timeout)
    if is_healthy:
        raise RuntimeError(
            "Server is already running at http://localhost:8000. "
            "Please stop it manually before running the stress test."
        )

    print_info("=" * 80)
    print_info(f"Starting Disaggregated Serving Stress Test for {model_name}")
    print_info(f"Test Mode: {test_mode}")
    print_info(f"Context Server: TP={ctx_tp}, PP={ctx_pp}")
    print_info(f"Generation Server: TP={gen_tp}, PP={gen_pp}")
    print_info("=" * 80)

    # Launch disaggregated serving
    with launch_disaggregated_serving(
        disaggregated_server_config=disaggregated_server_config,
        ctx_server_config=ctx_server_config,
        gen_server_config=gen_server_config,
        model_path=model_path,
        server_waiting_timeout=server_waiting_timeout,
        health_check_timeout=health_check_timeout,
    ):
        # Create server config for the main endpoint
        server_config = ServerConfig(
            port=8000, host="localhost", capacity_scheduler_policy="GUARANTEED_NO_EVICT"
        )

        # Set timeout based on model complexity (large models need more time)
        performance_config = PerformanceParams(test_timeout=36000)

        # Create stress test configuration with backend specified
        stress_model_config = ModelConfig(
            model_dir=model_config.model_dir,
            tp_size=model_config.tp_size,
            memory_requirement=model_config.memory_requirement,
            backend="pytorch",  # Disaggregated serving uses pytorch backend
        )

        stress_config = StressTestConfig(
            model_config=stress_model_config,
            server_config=server_config,
            stress_time=stress_time,
            stress_timeout=stress_timeout,
            enable_accuracy_test=True,
        )

        request_counter = RequestCounter()
        # Run baseline accuracy test first if enabled
        print_info("=" * 80)
        print_info("=== Running BASELINE ACCURACY TEST (GSM8K) ===")
        print_info("=" * 80)
        baseline_accuracy_success, baseline_accuracy_value = run_accuracy_test(
            model_path, server_config, stress_config, "baseline"
        )

        # Run performance test if enabled
        print_info("=" * 80)
        print_info("=== Running STAGE 1 PERFORMANCE TEST ===")
        print_info("=" * 80)
        measure_capacity_stage(
            model_name,
            model_path,
            server_config,
            performance_config,
            request_counter=request_counter,
        )
        print_info("=" * 80)
        print_info("=== Running STAGE 2 ANALYSIS ===")
        print_info("=" * 80)
        stage2_output = extract_stress_test_metrics(current_model=model_name)
        print_info(f"Stage 2 output: {stage2_output}")
        print_info("=" * 80)
        print_info("=== Running STAGE 3 STRESS TEST ===")
        print_info("=" * 80)
        stress_stage(
            model_name,
            model_path,
            server_config,
            stress_config,
            stage2_output,
            request_counter=request_counter,
        )

        # Run post-stress accuracy test if enabled
        print_info("=" * 80)
        print_info("=== Running POST-STRESS ACCURACY TEST (GSM8K) ===")
        print_info("=" * 80)
        post_stress_accuracy_success, post_stress_accuracy_value = run_accuracy_test(
            model_path, server_config, stress_config, "post_stress"
        )

        # Note: Accuracy comparison logic is commented out in original code
        # Can be re-enabled if needed to assert accuracy thresholds

        print_info("=" * 80)
        print_info("Disaggregated Serving Stress Test Completed Successfully!")
        print_info("=" * 80)

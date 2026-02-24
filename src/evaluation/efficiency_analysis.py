"""
efficiency_analysis.py — Phase 8: Efficiency Analysis
=======================================================

Measures:
  - Parameter count (total and trainable)
  - FLOPs (via fvcore or ptflops)
  - Inference time (GPU latency, CPU latency)
  - Peak memory usage

Compares CGRN against:
  - Standard BERT+ResNet fusion baseline (~125M + 25M = 150M params)
  - Static DistilBERT+MobileNet fusion

Highlights lightweight design advantage.
"""

import time
import json
import logging
import os
import gc
import torch
import torch.nn as nn
from typing import Dict, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# Efficiency Metrics Container
# =============================================================================

@dataclass
class EfficiencyMetrics:
    model_name:             str
    total_params:           int
    trainable_params:       int
    non_trainable_params:   int
    inference_time_ms_cpu:  float   = 0.0
    inference_time_ms_gpu:  float   = 0.0
    peak_memory_mb_gpu:     float   = 0.0
    flops:                  Optional[int] = None      # MACs
    flops_human:            str           = "N/A"
    model_size_mb:          float   = 0.0

    def __str__(self):
        lines = [
            f"Model: {self.model_name}",
            f"  Parameters  : {self.total_params:>12,} total  | "
            f"{self.trainable_params:>12,} trainable",
            f"  Model Size  : {self.model_size_mb:.1f} MB",
            f"  Latency CPU : {self.inference_time_ms_cpu:.2f} ms",
            f"  Latency GPU : {self.inference_time_ms_gpu:.2f} ms",
            f"  Peak GPU Mem: {self.peak_memory_mb_gpu:.1f} MB",
            f"  FLOPs       : {self.flops_human}",
        ]
        return "\n".join(lines)


# =============================================================================
# Efficiency Analyzer
# =============================================================================

class EfficiencyAnalyzer:
    """
    Profiles and compares model efficiency metrics.

    Usage
    -----
    >>> analyzer = EfficiencyAnalyzer()
    >>> metrics = analyzer.profile_model(cgrn_model, "CGRN", sample_batch)
    >>> print(metrics)
    """

    def __init__(self, n_warmup: int = 5, n_repeat: int = 50):
        self.n_warmup = n_warmup
        self.n_repeat = n_repeat

    # -------------------------------------------------------------------------
    def count_parameters(self, model: nn.Module) -> Dict[str, int]:
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            "total":         total,
            "trainable":     trainable,
            "non_trainable": total - trainable,
        }

    # -------------------------------------------------------------------------
    def model_size_mb(self, model: nn.Module) -> float:
        """Estimate model size from parameter storage."""
        total_bytes = sum(
            p.numel() * p.element_size() for p in model.parameters()
        )
        return total_bytes / (1024 ** 2)

    # -------------------------------------------------------------------------
    def measure_latency_cpu(
        self,
        model: nn.Module,
        sample_batch: Dict,
        forward_fn=None,
    ) -> float:
        """
        Measures average CPU inference time in milliseconds.
        Returns mean over n_repeat trials after n_warmup warmup passes.
        """
        model.eval().cpu()
        batch_cpu = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in sample_batch.items()
        }

        _forward = forward_fn or self._default_forward

        with torch.no_grad():
            # Warmup
            for _ in range(self.n_warmup):
                _forward(model, batch_cpu)

            # Measure
            times = []
            for _ in range(self.n_repeat):
                t0 = time.perf_counter()
                _forward(model, batch_cpu)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

        return sum(times) / len(times)

    # -------------------------------------------------------------------------
    def measure_latency_gpu(
        self,
        model: nn.Module,
        sample_batch: Dict,
        forward_fn=None,
    ) -> float:
        """
        Measures average GPU inference time in milliseconds using CUDA events.
        Returns 0.0 if CUDA is not available.
        """
        if not torch.cuda.is_available():
            return 0.0

        device = torch.device("cuda")
        model.eval().to(device)
        batch_gpu = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in sample_batch.items()
        }

        _forward = forward_fn or self._default_forward

        with torch.no_grad():
            # Warmup
            for _ in range(self.n_warmup):
                _forward(model, batch_gpu)
            torch.cuda.synchronize()

            # Measure with CUDA events
            times = []
            for _ in range(self.n_repeat):
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt   = torch.cuda.Event(enable_timing=True)
                start_evt.record()
                _forward(model, batch_gpu)
                end_evt.record()
                torch.cuda.synchronize()
                times.append(start_evt.elapsed_time(end_evt))

        return sum(times) / len(times)

    # -------------------------------------------------------------------------
    def measure_peak_gpu_memory(
        self,
        model: nn.Module,
        sample_batch: Dict,
        forward_fn=None,
    ) -> float:
        """Returns peak GPU memory usage in MB during forward pass."""
        if not torch.cuda.is_available():
            return 0.0

        device = torch.device("cuda")
        model.eval().to(device)
        batch_gpu = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in sample_batch.items()
        }

        _forward = forward_fn or self._default_forward
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            _forward(model, batch_gpu)

        peak_bytes = torch.cuda.max_memory_allocated()
        return peak_bytes / (1024 ** 2)

    # -------------------------------------------------------------------------
    def _default_forward(self, model, batch):
        """Default forward pass for CGRN models."""
        if hasattr(model, "forward") and callable(model.forward):
            input_ids      = batch.get("input_ids")
            attention_mask = batch.get("attention_mask")
            images         = batch.get("images")
            if input_ids is not None and images is not None:
                return model(input_ids, attention_mask, images)
        raise ValueError("Provide a custom forward_fn for non-standard model inputs.")

    # -------------------------------------------------------------------------
    def try_compute_flops(
        self,
        model: nn.Module,
        sample_batch: Dict,
        forward_fn=None,
    ) -> Optional[int]:
        """
        Attempts FLOPs computation via fvcore or ptflops.
        Returns None if neither library is available.
        """
        # Try ptflops (thop)
        try:
            from thop import profile as thop_profile
            input_ids      = sample_batch.get("input_ids")
            attention_mask = sample_batch.get("attention_mask")
            images         = sample_batch.get("images")
            if input_ids is not None and images is not None:
                macs, _ = thop_profile(
                    model,
                    inputs=(input_ids, attention_mask, images),
                    verbose=False,
                )
                return int(macs)
        except Exception:
            pass

        logger.debug("FLOPs computation skipped (thop/fvcore not installed).")
        return None

    # -------------------------------------------------------------------------
    def profile_model(
        self,
        model:        nn.Module,
        model_name:   str,
        sample_batch: Dict,
        forward_fn=None,
    ) -> EfficiencyMetrics:
        """
        Full efficiency profiling of a model.

        Parameters
        ----------
        model        : the PyTorch model to profile
        model_name   : display name
        sample_batch : dict with 'input_ids', 'attention_mask', 'images'
        forward_fn   : optional custom forward function

        Returns
        -------
        EfficiencyMetrics
        """
        logger.info(f"Profiling: {model_name}")

        params      = self.count_parameters(model)
        size_mb     = self.model_size_mb(model)
        lat_cpu     = self.measure_latency_cpu(model, sample_batch, forward_fn)
        lat_gpu     = self.measure_latency_gpu(model, sample_batch, forward_fn)
        peak_mem    = self.measure_peak_gpu_memory(model, sample_batch, forward_fn)
        flops       = self.try_compute_flops(model, sample_batch, forward_fn)

        flops_human = "N/A"
        if flops is not None:
            if flops >= 1e9:
                flops_human = f"{flops/1e9:.2f} GMACs"
            elif flops >= 1e6:
                flops_human = f"{flops/1e6:.2f} MMACs"
            else:
                flops_human = f"{flops:,} MACs"

        metrics = EfficiencyMetrics(
            model_name=model_name,
            total_params=params["total"],
            trainable_params=params["trainable"],
            non_trainable_params=params["non_trainable"],
            inference_time_ms_cpu=lat_cpu,
            inference_time_ms_gpu=lat_gpu,
            peak_memory_mb_gpu=peak_mem,
            flops=flops,
            flops_human=flops_human,
            model_size_mb=size_mb,
        )

        logger.info(str(metrics))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return metrics

    # -------------------------------------------------------------------------
    def print_efficiency_table(self, results: List[EfficiencyMetrics]):
        """Print formatted efficiency comparison table."""
        print("\n" + "=" * 90)
        print("CGRN Efficiency Analysis")
        print("=" * 90)
        header = (
            f"| {'Model':<22} | {'Params':>10} | {'Size MB':>7} | "
            f"{'CPU ms':>7} | {'GPU ms':>7} | {'GPU Mem MB':>10} | {'FLOPs':>12} |"
        )
        sep = "|" + ("-" * 24) + "|" + ("-" * 12) + "|" + ("-" * 9) + \
              "|" + ("-" * 9) + "|" + ("-" * 9) + "|" + ("-" * 12) + "|" + ("-" * 14) + "|"
        print(header)
        print(sep)

        for r in results:
            params_str = f"{r.total_params/1e6:.1f}M"
            print(
                f"| {r.model_name:<22} | {params_str:>10} | {r.model_size_mb:>7.1f} | "
                f"{r.inference_time_ms_cpu:>7.1f} | {r.inference_time_ms_gpu:>7.2f} | "
                f"{r.peak_memory_mb_gpu:>10.1f} | {r.flops_human:>12} |"
            )

        print("=" * 90 + "\n")

    # -------------------------------------------------------------------------
    def save_results(self, results: List[EfficiencyMetrics], output_path: str):
        data = []
        for r in results:
            data.append({
                "model_name":            r.model_name,
                "total_params":          r.total_params,
                "trainable_params":      r.trainable_params,
                "model_size_mb":         r.model_size_mb,
                "inference_time_ms_cpu": r.inference_time_ms_cpu,
                "inference_time_ms_gpu": r.inference_time_ms_gpu,
                "peak_memory_mb_gpu":    r.peak_memory_mb_gpu,
                "flops":                 r.flops,
                "flops_human":           r.flops_human,
            })
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Efficiency results saved → {output_path}")

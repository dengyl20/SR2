#!/usr/bin/env python3
"""
Augmentation Ablation Evaluation Script

Uses the existing evaluation infrastructure with a custom ARC evaluator
that limits augmentations per base task. This ensures full compatibility
with the existing evaluation pipeline.

At max_augmentations=1000, behaves exactly like the standard evaluation.

Usage:
    python run_augmentation_ablation_eval.py \
        --checkpoint checkpoints/model/checkpoint.pt \
        --data-path data/arc-aug-1000 \
        --augmentation-counts 1,10,100,1000 \
        --output-dir augmentation_ablation_results \
        --random-seed 42
"""

import os
import json
import argparse
from pathlib import Path
from typing import List

from evaluate_trained_model import evaluate_checkpoint


def run_augmentation_ablation(
    checkpoint_path: str,
    data_path: str, 
    output_dir: str,
    augmentation_counts: List[int],
    random_seed: int = 42,
    batch_size: int = 512
):
    """
    Run augmentation ablation study using the existing evaluation infrastructure.
    
    For each augmentation count, creates a custom evaluator configuration and
    runs evaluation. Results are saved to separate subdirectories.
    """
    
    import os
    is_rank_0 = os.environ.get("LOCAL_RANK", "0") == "0"
    
    if is_rank_0:
        print("=== Augmentation Ablation Study ===")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Data path: {data_path}")
        print(f"Augmentation counts: {augmentation_counts}")
        print(f"Output dir: {output_dir}")
        print(f"Random seed: {random_seed}")
        print(f"Batch size: {batch_size}")
        print("")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_summary = {}
    
    for i, max_aug in enumerate(augmentation_counts):
        if is_rank_0:
            print(f"=== Running evaluation {i+1}/{len(augmentation_counts)}: max_augmentations={max_aug} ===")
        
        # Create output subdirectory for this run
        run_output_dir = output_dir / f"max_aug_{max_aug}"
        run_output_dir.mkdir(exist_ok=True)
        
        # Configure the custom evaluator using the expected format
        from pretrain import EvaluatorConfig
        
        config_overrides = {
            "global_batch_size": batch_size,
            "evaluators": [
                EvaluatorConfig(
                    name="arc_augmentation_ablation@AugmentationAblationARC",  # Module@Class format
                    submission_K=2,
                    aggregated_voting=True,
                    pass_Ks=[1, 2, 5, 10, 100, 1000],
                    max_augmentations_per_task=max_aug,
                    random_seed=random_seed
                )
            ]
        }
        
        # Run evaluation using the existing infrastructure
        if is_rank_0:
            print(f"Starting evaluation with max {max_aug} augmentations...")
        try:
            evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                data_path=data_path,
                output_dir=str(run_output_dir),
                config_overrides=config_overrides,
                wandb_project=None,  # Don't log to wandb
                wandb_run_name=None,
                save_predictions=False
            )
            
            # Load the metrics that were saved
            metrics_file = run_output_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    
                # Extract key metrics for summary
                summary_metrics = {}
                for key, value in metrics.items():
                    if "pass@" in key:
                        summary_metrics[key] = value
                    elif "augmentations" in key:
                        summary_metrics[key] = value
                        
                results_summary[max_aug] = {
                    "status": "success", 
                    "metrics": summary_metrics
                }
                
                if is_rank_0:
                    print(f"✓ Evaluation completed for max_augmentations={max_aug}")
                    print(f"  Key metrics: {summary_metrics}")
            else:
                if is_rank_0:
                    print(f"⚠ Evaluation completed but no metrics file found for max_augmentations={max_aug}")
                results_summary[max_aug] = {"status": "no_metrics"}
                
        except Exception as e:
            import traceback
            # Only print errors from rank 0 to avoid spam in multi-GPU setup
            if is_rank_0:
                print(f"✗ Evaluation failed for max_augmentations={max_aug}: {e}")
                print("Full traceback:")
                traceback.print_exc()
            results_summary[max_aug] = {"status": "failed", "error": str(e)}
            continue
            
        if is_rank_0:
            print("")
    
    # Save overall summary
    summary_file = output_dir / "ablation_summary.json"
    
    summary_data = {
        "experiment_config": {
            "checkpoint_path": checkpoint_path,
            "data_path": data_path,
            "augmentation_counts": augmentation_counts,
            "random_seed": random_seed,
            "batch_size": batch_size
        },
        "results": results_summary
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Print final summary (only from rank 0)
    if is_rank_0:
        print("=== AUGMENTATION ABLATION SUMMARY ===")
        print(f"Results saved to: {output_dir}")
        print("")
        
        success_count = sum(1 for r in results_summary.values() if r.get("status") == "success")
        print(f"Successful evaluations: {success_count}/{len(augmentation_counts)}")
        
        if success_count > 0:
            print("\nPerformance by augmentation count:")
            print("Max Aug | Pass@1  | Pass@2  | Pass@10 | Pass@100| Augs Used")
            print("--------|---------|---------|---------|---------|----------")
            
            for max_aug in augmentation_counts:
                if results_summary[max_aug].get("status") == "success":
                    metrics = results_summary[max_aug]["metrics"]
                    
                    # Extract pass rates
                    pass1 = metrics.get("AugmentationAblationARC/pass@1", 0.0)
                    pass2 = metrics.get("AugmentationAblationARC/pass@2", 0.0) 
                    pass10 = metrics.get("AugmentationAblationARC/pass@10", 0.0)
                    pass100 = metrics.get("AugmentationAblationARC/pass@100", 0.0)
                    
                    # Extract augmentation info
                    augs_used = metrics.get("AugmentationAblationARC/total_augmentations_used", 0)
                    
                    print(f"{max_aug:7d} | {pass1:7.3f} | {pass2:7.3f} | {pass10:7.3f} | {pass100:7.3f} | {augs_used:8.0f}")
                else:
                    status = results_summary[max_aug].get("status", "unknown")
                    print(f"{max_aug:7d} | {status:7s} | {status:7s} | {status:7s} | {status:7s} | {status:8s}")
        
        failed_runs = [max_aug for max_aug, r in results_summary.items() if r.get("status") != "success"]
        if failed_runs:
            print(f"\nFailed runs: {failed_runs}")
            
        print(f"\nDetailed results in: {output_dir}")
        print(f"Summary file: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Run augmentation ablation evaluation")
    
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data-path", 
        required=True,
        help="Path to test dataset (should have many augmentations)"
    )
    parser.add_argument(
        "--augmentation-counts",
        default="1,10,100,1000",
        help="Comma-separated list of max augmentations per task to test"
    )
    parser.add_argument(
        "--output-dir",
        default="augmentation_ablation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for augmentation sampling"
    )
    parser.add_argument(
        "--batch-size", 
        type=int,
        default=512,
        help="Batch size for evaluation"
    )
    
    args = parser.parse_args()
    
    # Parse augmentation counts
    augmentation_counts = [int(x.strip()) for x in args.augmentation_counts.split(",")]
    
    run_augmentation_ablation(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        output_dir=args.output_dir,
        augmentation_counts=augmentation_counts,
        random_seed=args.random_seed,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
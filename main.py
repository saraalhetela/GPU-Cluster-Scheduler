"""
main.py -- GPU cluster scheduler, single entry point.

Ties together what run_training.py and evaluate.py currently do as two
separate manual steps into the "one command produces the full result"
deliverable called for in the timeline doc's immediate next steps:

    load data -> split for training -> train DuelingMLP -> compare against
    FCFS/Always-Max/Priority on the full shared-32-GPU-pool simulation ->
    save results.json

Usage:
    python main.py
"""
import json
import os

import pandas as pd

import config
import data_preprocessing as dp
from model import DuelingMLP
from train import train_agent
from baseline import run_all_baselines, compute_hourly_demand
from evaluate import evaluate_agent_shared_pool, evaluate_agent
from export_trace import trace_priority, trace_agent_shared_pool
import utils


def main():
    print(f"Device: {config.DEVICE}")

    print("\nLoading data...")
    prices = dp.load_price_data()
    train_jobs, val_jobs = dp.load_job_data(split=True)
    # Baseline/shared-pool evaluation need the WHOLE queue (they're a
    # whole-queue contention simulation, not a per-job train/test split --
    # same set baseline.py and evaluate.py's own main() already use).
    full_jobs = dp.load_job_data()
    print(f"  {len(train_jobs)} train jobs, {len(val_jobs)} val jobs "
          f"({len(full_jobs)} total jobs used for the shared-pool evaluation)")

    print("\nInitializing model...")
    model = DuelingMLP(config.STATE_DIM, config.ACTION_DIM).to(config.DEVICE)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print(f"\nTraining for {config.MAX_STEPS} steps "
          f"(checkpoint/val every {config.CHECKPOINT_FREQ} steps)...")
    trained_model, train_rewards, val_rewards = train_agent(
        model, train_jobs, val_jobs, prices, device=config.DEVICE
    )

    print("\nSaving final checkpoint...")
    import torch
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    final_ckpt = f"{config.CHECKPOINT_DIR}/dqn_final.pt"
    torch.save(trained_model.state_dict(), final_ckpt)
    print(f"  -> {final_ckpt}")

    best_ckpt = f"{config.CHECKPOINT_DIR}/ckpt_best.pt"
    if os.path.exists(best_ckpt):
        print(f"\nLoading best-validation checkpoint for evaluation ({best_ckpt})...")
        trained_model.load_state_dict(torch.load(best_ckpt))
        trained_model.eval()
    else:
        print("\n(No ckpt_best.pt found -- evaluating final-step weights. "
              "Make sure the train.py best-checkpoint change is applied.)")

    print("\nPlotting training curves...")
    utils.plot_profits(train_rewards, title="Training Episode Rewards",
                        filename="train_rewards.png")
    utils.plot_val_curve(val_rewards, filename="val_rewards.png")

    print("\nRunning baseline comparison (shared 32-GPU pool, all jobs competing)...")
    baseline_df = run_all_baselines(jobs=full_jobs, prices=prices)

    print("Running agent through the SAME shared-pool simulation (fair comparison)...")
    shared_row = evaluate_agent_shared_pool(trained_model, full_jobs, prices, device=config.DEVICE)

    print("\n" + "=" * 78)
    print("  HEADLINE COMPARISON -- same shared 32-GPU pool for everyone")
    print("=" * 78)
    full_df = pd.concat([baseline_df, pd.DataFrame([shared_row])], ignore_index=True)
    print(full_df.to_string(index=False))

    best_baseline = baseline_df.loc[baseline_df["deadline_success_rate_pct"].idxmax()]
    delta = shared_row["deadline_success_rate_pct"] - best_baseline["deadline_success_rate_pct"]
    cost_delta = shared_row["total_cost"] - best_baseline["total_cost"]
    print(f"\n  Agent vs. best baseline ({best_baseline['policy']}): "
          f"{delta:+.1f} points deadline success rate, at "
          f"${cost_delta:+.2f} cost delta")
    print("=" * 78)

    print("\nRunning isolated per-job evaluation (diagnostic only -- NOT "
          "comparable to the table above, no shared-capacity constraint, "
          "don't put this number in the pitch)...")
    full_demand = compute_hourly_demand(full_jobs)
    isolated_row, per_job = evaluate_agent(
        trained_model, full_jobs, prices, demand_curve=full_demand, device=config.DEVICE
    )
    print(f"  isolated: cost={isolated_row['total_cost']}, "
          f"completed={isolated_row['jobs_completed']}, "
          f"success_rate={isolated_row['deadline_success_rate_pct']}% "
          f"-- inflated vs. the headline number above, ignore for reporting")

    # --- results.json -----------------------------------------------------
    # Per-policy fields as specified in the timeline doc: gpu_utilization_pct,
    # total_cost, jobs_completed, deadlines_missed, deadline_success_rate_pct
    # for each baseline plus rl_agent_shared_pool -- NOT alpha/win_rate,
    # those were stock-trading leftovers from two pivots ago.
    policies = {row["policy"]: row for row in baseline_df.to_dict(orient="records")}
    policies[shared_row["policy"]] = shared_row

    results = {
        "policies": policies,
        "best_baseline": best_baseline["policy"],
        "deadline_success_rate_delta_pts": round(delta, 1),
        "cost_delta_vs_best_baseline": round(cost_delta, 2),
        "n_train_jobs": len(train_jobs),
        "n_val_jobs": len(val_jobs),
        "n_jobs_evaluated": len(full_jobs),
        "isolated_diagnostic": {
            **isolated_row,
            "note": ("NOT capacity-constrained -- inflated vs. the shared-pool "
                     "number above, exclude from pitch/reporting"),
        },
    }

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results.json")

    # --- trace.json for the dashboard demo --------------------------------
    # Reuses export_trace.py's functions directly (not a reimplementation)
    # so this is guaranteed to match `python export_trace.py` run standalone.
    print("\nGenerating hour-by-hour trace for the dashboard demo...")
    priority_trace = trace_priority(full_jobs, prices)
    agent_trace = trace_agent_shared_pool(trained_model, full_jobs, prices, device=config.DEVICE)
    trace_out = {
        "n_jobs": len(full_jobs),
        "total_gpus": config.TOTAL_CLUSTER_GPUS,
        "episode_length": config.EPISODE_LENGTH,
        "priority": priority_trace,
        "rl_agent_shared_pool": agent_trace,
    }
    with open("trace.json", "w") as f:
        json.dump(trace_out, f, indent=2)
    print("  -> trace.json")

    # --- serve the dashboard demo -------------------------------------
    _serve_dashboard()


def _serve_dashboard(trace_out, preferred_port=8000):
    from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

    if not os.path.exists("dashboard_demo.html"):
        print("\n(dashboard_demo.html not found in this folder -- skipping "
              "auto-serve. Copy it here and rerun to view the demo.)")
        return

    run_filename = _build_run_html(trace_out)

    # Detect Google Colab.
    try:
        import google.colab

        print("\n" + "=" * 78)
        print("GOOGLE COLAB DETECTED")
        print("=" * 78)
        print(f"Dashboard generated successfully:")
        print(f"  {run_filename}")
        print("\nRun the following in a NEW notebook cell:\n")

        print("from google.colab import files")
        print(f'files.download("{run_filename}")')

        print("\nThen open the downloaded HTML file in your browser.")
        print("=" * 78)

        return

    except ImportError:
        pass

    # ---------- Local machine ----------
    class NoCacheHandler(SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            super().end_headers()

    server = None
    port = preferred_port

    for candidate in range(preferred_port, preferred_port + 10):
        try:
            server = ThreadingHTTPServer(("localhost", candidate), NoCacheHandler)
            port = candidate
            break
        except OSError:
            continue

    if server is None:
        print(f"\nCouldn't bind a port near {preferred_port}.")
        return

    url = f"http://localhost:{port}/{run_filename}"

    print("\n" + "=" * 78)
    print(f"DEMO READY --> {url}")
    print(f"Serving from: {os.getcwd()}")
    print("=" * 78)
    print("(Ctrl+C to stop the server.)")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.shutdown()


if __name__ == "__main__":
    main()

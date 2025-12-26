import copy
import json
import logging
import os
import pickle
import signal
import sys
from contextlib import redirect_stderr, redirect_stdout
from multiprocessing import Pool
from pathlib import Path

import astropy.constants as const
import astropy.units as u
import numpy as np
from tqdm import tqdm

# Import the AYO loader
from yieldplotlib.load.ayo import AYOInputFile
import EXOSIMS.MissionSim as MissionSim

# =============================================================================
# Configuration
# =============================================================================

# Simulation Settings
NUM_RUNS = 100  # Number of simulation runs per wavelength
START_SEED = 0  # Starting seed value
NUM_PROCESSES = 5  # Number of parallel processes
DETECTION_WAVELENGTH_NM = 500  # Constant for all runs

# Path Configuration
# Assumes driver.py is in the root or similar depth to previous scripts.
# Adjust these paths if your directory structure differs.
BASE_DIR = Path(".")
AYO_INPUT_DIR = (
    BASE_DIR / "yieldplotlib/input"
)  # Directory containing .ayo files and run_base_new.json
OUTPUT_JSON_DIR = (
    BASE_DIR / "hwo-mission-control/2025/Natasha_JATIS/input"
)  # Where generated JSONs go
OUTPUT_RESULTS_DIR = (
    BASE_DIR / "hwo-mission-control/2025/Natasha_JATIS/output"
)  # Where sim results go
BASE_EXOSIMS_FILE = AYO_INPUT_DIR / "run_base_new.json"

# Wavelength Configurations (microns)
# The script will convert these to nm for the EXOSIMS export
WAVELENGTH_CONFIGS = {
    "h2o_o2": [
        0.752,
        0.768,
        0.784,
        0.801,
        0.819,
        0.83,
        0.848,
        0.867,
        0.879,
        0.898,
        0.911,
    ],
    "h2o_only": [
        0.72,
        0.736,
        0.752,
        0.768,
        0.784,
        0.801,
        0.819,
        0.83,
        0.848,
        0.867,
        0.879,
        0.898,
        0.911,
        0.93,
        0.944,
        0.957,
        0.978,
        0.992,
    ],
}


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    print("\nReceived interrupt signal. Cleaning up...")
    sys.exit(1)


def generate_exosims_configs():
    """
    Generates EXOSIMS JSON files for all defined wavelengths.

    Returns:
        list: A list of run_names (filenames without extension) to be simulated.
    """
    generated_run_names = []

    print(f"\n{'='*80}")
    print("GENERATING EXOSIMS CONFIGURATIONS")
    print(f"{'='*80}")

    OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

    for base_name, wavelengths_microns in WAVELENGTH_CONFIGS.items():
        ayo_path = AYO_INPUT_DIR / f"{base_name}.ayo"

        if not ayo_path.exists():
            print(f"Error: AYO file not found at {ayo_path}")
            continue

        print(f"Loading AYO source: {base_name}")
        ayo_input = AYOInputFile(ayo_path)

        for wl_micron in wavelengths_microns:
            # Convert micron to nm (e.g., 0.72 -> 720)
            wl_nm = int(round(wl_micron * 1000))

            # Construct specific run name
            run_name = f"{base_name}_{wl_nm}nm"
            output_file = OUTPUT_JSON_DIR / f"{run_name}.json"

            # Check if file already exists
            if output_file.exists():
                print(f"  -> Skipped: {run_name}.json (already exists)")
                generated_run_names.append(run_name)
                continue

            # Export using the AYO library
            # Note: cachedir set to match previous script convention
            ayo_input.export_exosims(
                str(output_file),
                base_file=str(BASE_EXOSIMS_FILE),
                detection_wavelength_nm=DETECTION_WAVELENGTH_NM,
                characterization_wavelength_nm=wl_nm,
                cachedir="$HOME/.EXOSIMS/2025/Natasha_JATIS",
            )

            generated_run_names.append(run_name)
            print(f"  -> Generated: {run_name}.json (Char λ: {wl_nm} nm)")

    return generated_run_names


def classify_planets(SS):
    """
    Classify planets by Kopparapu bin and Earth-like status.
    Adapted from EXOSIMS SubtypeCompleteness.
    """
    TL = SS.TargetList
    SU = SS.SimulatedUniverse
    plan2star = SU.plan2star
    Rp = SU.Rp.to_value(u.earthRad)
    a = SU.a.to_value(u.AU)
    e = SU.e

    star_lum = TL.L[plan2star] * u.Lsun
    earth_Lp = const.L_sun / (1 * (1 + (0.0167**2) / 2)) ** 2
    Lp = (star_lum / (a * (1 + (e**2) / 2)) ** 2 / earth_Lp).decompose().value

    Rp_bins = np.array([0.5, 1.0, 1.75, 3.5, 6.0, 14.3])
    all_Rp_types = np.array(
        ["Rocky", "Super-Earth", "Sub-Neptune", "Sub-Jovian", "Jovian"]
    )
    L_bins = np.array(
        [
            [182, 1.0, 0.28, 0.0035],
            [187, 1.12, 0.30, 0.0030],
            [188, 1.15, 0.32, 0.0030],
            [220, 1.65, 0.45, 0.0030],
            [220, 1.65, 0.40, 0.0025],
        ]
    )

    Rp_bin = np.digitize(Rp, Rp_bins) - 1
    Rp_bin = np.clip(Rp_bin, 0, len(all_Rp_types) - 1)
    Rp_types = all_Rp_types[Rp_bin]

    all_L_types = np.array(["Hot", "Warm", "Cold"])
    specific_L_bins = L_bins[Rp_bin, :]
    L_bin = np.zeros(len(Lp))
    for i in range(len(Lp)):
        L_bin[i] = np.digitize(Lp[i], specific_L_bins[i]) - 1
    L_bin = np.clip(L_bin, 0, len(all_L_types) - 1).astype(int)
    L_types = all_L_types[L_bin]
    subtypes = [f"{L_type} {Rp_type}" for L_type, Rp_type in zip(L_types, Rp_types)]

    scaled_a = a / np.sqrt(star_lum.to(u.Lsun).value)
    lower_a = 0.95
    upper_a = 1.67
    lower_R = 0.8 / np.sqrt(scaled_a)
    upper_R = 1.4
    earth_a_cond = (lower_a <= scaled_a) & (scaled_a < upper_a)
    earth_Rp_cond = (lower_R <= Rp) & (Rp < upper_R)

    is_earth = earth_a_cond & earth_Rp_cond
    return subtypes, is_earth


def run_single_simulation(seed, specs, output_dir, debug=False):
    """Run a single simulation with given seed and specs."""
    specs["seed"] = seed
    sim = MissionSim.MissionSim(**specs)
    SS = sim.SurveySimulation
    subtypes, is_earth = classify_planets(SS)

    results_file = output_dir / f"results_{seed:03}.json"
    planet_file = output_dir / f"planet_info_{seed:03}.pkl"

    if results_file.exists() and planet_file.exists():
        if debug:
            print(f"Seed {seed} already completed, skipping...")
        return

    SS.run_sim()
    results = SS.mission_stats

    with open(results_file, "w") as f_out:
        json.dump(results, f_out, indent=2)

    with open(planet_file, "wb") as f_out:
        pickle.dump({"subtypes": subtypes, "is_earth": is_earth}, f_out)

    if debug:
        print(f"Completed seed {seed}")


def run_single_simulation_wrapper(args):
    """Wrapper function for parallel execution."""
    seed, specs, output_dir, debug = args
    if debug:
        run_single_simulation(seed, specs, output_dir, debug)
    else:
        with open(os.devnull, "w") as devnull, redirect_stdout(
            devnull
        ), redirect_stderr(devnull):
            run_single_simulation(seed, specs, output_dir, debug)


def compile_results(output_dir, num_runs, start_seed):
    """Compile results from all simulation runs in a specific directory."""
    num_chars_list = []

    for seed in range(start_seed, start_seed + num_runs):
        results_file = output_dir / f"results_{seed:03}.json"
        if not results_file.exists():
            continue

        with open(results_file, "r") as f:
            results = json.load(f)

        num_chars = sum(results.get("chars", {}).values())
        num_chars_list.append(num_chars)

    if len(num_chars_list) == 0:
        return None

    return {
        "mean": np.mean(num_chars_list),
        "std": np.std(num_chars_list, ddof=1),
        "min": np.min(num_chars_list),
        "max": np.max(num_chars_list),
        "n_runs": len(num_chars_list),
    }


def analyze_wavelength_yields(comparison, output_dir):
    """
    Organize results by configuration and wavelength to analyze yield drops.
    """
    print("\n" + "=" * 80)
    print("YIELD vs WAVELENGTH ANALYSIS")
    print("=" * 80)

    analysis_data = {}

    for config_name in WAVELENGTH_CONFIGS.keys():
        analysis_data[config_name] = []
        print(f"\nConfiguration: {config_name}")
        print(f"{'Wavelength (nm)':<20} {'Mean Yield':<15} {'Std Dev':<15}")
        print("-" * 60)

        # Sort keys to ensure wavelength order
        # Filter keys belonging to this config
        relevant_runs = [
            k for k in comparison.keys() if k.startswith(config_name + "_")
        ]

        # Sort by wavelength extracted from string
        relevant_runs.sort(key=lambda x: int(x.split("_")[-1].replace("nm", "")))

        for run_name in relevant_runs:
            wl_str = run_name.split("_")[-1].replace("nm", "")
            wl = int(wl_str)
            stats = comparison[run_name]

            # Store for file export
            analysis_data[config_name].append(
                {
                    "wavelength_nm": wl,
                    "mean_yield": stats["mean"],
                    "std_yield": stats["std"],
                }
            )

            print(f"{wl:<20} {stats['mean']:<15.2f} {stats['std']:<15.2f}")

    # Save detailed analysis
    analysis_file = output_dir / "yield_vs_wavelength_analysis.json"
    with open(analysis_file, "w") as f:
        json.dump(analysis_data, f, indent=2)

    print(f"\nAnalysis saved to: {analysis_file}")


def run_scenario(run_name, output_dir_base, num_runs, start_seed, num_processes, debug):
    """Run simulations for a single scenario."""
    input_path = OUTPUT_JSON_DIR / f"{run_name}.json"

    # Parse run_name to extract config and wavelength
    # Format: "h2o_o2_752nm" -> config="h2o_o2", wavelength="752nm"
    parts = run_name.split("_")
    wavelength = parts[-1]  # e.g., "752nm"
    config_name = "_".join(parts[:-1])  # e.g., "h2o_o2"

    # Create hierarchical directory structure: config/wavelength
    output_dir = output_dir_base / config_name / wavelength

    if not input_path.exists():
        print(f"Warning: Input file not found: {input_path}")
        return None

    with open(input_path, "r") as f:
        base_specs = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(start_seed, start_seed + num_runs))
    run_args = [(seed, copy.deepcopy(base_specs), output_dir, debug) for seed in seeds]

    if debug:
        print(f"Running {run_name} (Serial)...")
        for run_arg in run_args:
            run_single_simulation_wrapper(run_arg)
    else:
        pool = None
        try:
            pool = Pool(processes=num_processes)
            list(
                tqdm(
                    pool.imap_unordered(run_single_simulation_wrapper, run_args),
                    total=len(run_args),
                    desc=f"Running {run_name}",
                    leave=False,
                )
            )
        finally:
            if pool is not None:
                pool.close()
                pool.join()

    return output_dir


def main():
    debug = "--debug" in sys.argv

    logging.basicConfig(
        filename="simulation.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"\n{'='*80}")
    print("MISSION SIMULATION DRIVER - WAVELENGTH SCAN")
    print(f"{'='*80}")

    # 1. Generate Input Files
    run_names = generate_exosims_configs()

    print(f"\n{'='*80}")
    print(f"STARTING SIMULATIONS ({NUM_RUNS} seeds per config)")
    print(f"{'='*80}")

    # 2. Run Simulations
    full_stats = {}

    for i, run_name in enumerate(run_names):
        print(f"[{i+1}/{len(run_names)}] Scenario: {run_name}")

        output_dir = run_scenario(
            run_name,
            OUTPUT_RESULTS_DIR,
            NUM_RUNS,
            START_SEED,
            NUM_PROCESSES,
            debug,
        )

        if output_dir:
            stats = compile_results(output_dir, NUM_RUNS, START_SEED)
            if stats:
                full_stats[run_name] = stats

    # 3. Analyze Results
    if full_stats:
        analyze_wavelength_yields(full_stats, OUTPUT_RESULTS_DIR)

        # Also save raw comparison
        comparison_file = OUTPUT_RESULTS_DIR / "full_comparison_raw.json"
        with open(comparison_file, "w") as f:
            # Helper for numpy types
            def default(o):
                if isinstance(o, np.integer):
                    return int(o)
                if isinstance(o, np.floating):
                    return float(o)
                if isinstance(o, np.ndarray):
                    return o.tolist()
                raise TypeError

            json.dump(full_stats, f, indent=2, default=default)

    print(f"\n{'='*80}")
    print("ALL SIMULATIONS COMPLETED")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

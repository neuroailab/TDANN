import subprocess
from spacetorch.paths import analysis_config_dir

# change this to point to your python executable
PYEXEC = "/share/kalanit/users/eshedm/tdann/.venv/bin/python"

# change this to point to benchmark script
BMARK_SCRIPT = "/share/kalanit/users/eshedm/tdann/scripts/run_brainscore_benchmarks.py"
SEEDS = range(5)
LW_MODS = ["_lw01", "", "_lwx2", "_lwx5", "_lwx10", "_lwx100"]
BENCHMARKS = [
    "tolias.Cadena2017-pls",
    "dicarlo.MajajHong2015.V4-pls",
    "dicarlo.MajajHong2015.IT-pls",
]
BMARK_STRING = ",".join(BENCHMARKS)

PREFIX_LOOKUP = {
    "SimCLR": "simclr",
    "Old SCL": "simclr",
    "Supervised": "supswap_supervised",
}


def main():
    bases = {
        "SimCLR": "simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3",
        "Old SCL": "simclr_spatial_resnet18_swappedon_SineGrating2019_old_scl",
        "Supervised": "supervised_spatial_resnet18_swappedon_SineGrating2019",
    }

    for base_name, base in bases.items():
        prefix = PREFIX_LOOKUP[base_name]
        for lw_modifier in LW_MODS:
            print(f"\t{lw_modifier}")
            for seed in SEEDS:
                seed_str = f"_seed_{seed}" if seed > 0 else ""
                target = f"{base}{lw_modifier}{seed_str}"
                analysis_config = analysis_config_dir / prefix / f"{target}.yaml"

                cmd = [
                    PYEXEC,
                    BMARK_SCRIPT,
                    "--benchmark_identifiers",
                    BMARK_STRING,
                    "--config",
                    str(analysis_config),
                ]
                subprocess.run(cmd, stdout=subprocess.PIPE)


if __name__ == "__main__":
    main()

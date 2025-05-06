from __future__ import annotations
import click
from pathlib import Path
from src.kernels.configurations import OpticalDotProductConfiguration

DEFAULT_OUTPUT_PATH = Path(__file__).parent.parent / "config" / "scripted"

def main(
    output_path: Path = DEFAULT_OUTPUT_PATH,
):
    if output_path.exists():
        raise FileExistsError(f"Output path {output_path} already exists.")
    output_path.mkdir(parents=True, exist_ok=True)
    for i in range(2, 16+1, 2):
        config = OpticalDotProductConfiguration()

        config.adc_cfg.quantization_bitwidth=8
        config.input_dac_cfg.quantization_bitwidth=8
        config.weight_dac_cfg.quantization_bitwidth=i
        config.save_to_path(output_path / f"WDAC{i}_IDAC8_ADC8.yaml")

    for i in range(2, 16+1, 2):
        config = OpticalDotProductConfiguration()

        config.adc_cfg.quantization_bitwidth=8
        config.input_dac_cfg.quantization_bitwidth=i
        config.weight_dac_cfg.quantization_bitwidth=8
        config.save_to_path(output_path / f"WDAC8_IDAC{i}_ADC8.yaml")

    for i in range(2, 16+2, 2):
        config = OpticalDotProductConfiguration()

        config.adc_cfg.quantization_bitwidth=i
        config.input_dac_cfg.quantization_bitwidth=8
        config.weight_dac_cfg.quantization_bitwidth=8
        config.save_to_path(output_path / f"WDAC8_IDAC8_ADC{i}.yaml")

if __name__ == "__main__":
    click.echo("Running make_config_script.py to output path:")
    click.echo(DEFAULT_OUTPUT_PATH.resolve().as_posix())
    click.echo("  (this is using default parameters for OpticalDotProductConfiguration)")
    click.confirm("Are you sure you want to continue?", abort=True)
    main()
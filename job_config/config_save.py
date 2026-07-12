from pathlib import Path

import yaml


class ConfigSave:
    def __init__(self, config, output_path: Path):
        self.config = config
        self.output_path = output_path

    def save_config(self):
        output_path = Path(self.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "config.yml", "w") as config_file:
            yaml.dump(self.config, config_file)

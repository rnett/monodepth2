from carla_dataset.config import Rain, expand_wildcards
from tqdm import tqdm

if __name__ == '__main__':
    configs = expand_wildcards(rain=Rain.Clear, sunset=False)
    pbar = tqdm(configs, ncols=200)
    for config in pbar:
        # config.cylindrical_data.download()
        pbar.set_postfix_str(f"Downloading {config.pinhole_data.download_file}")
        print()
        file = config.pinhole_data.download()
        # config.pose_data.download()

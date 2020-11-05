from carla_dataset.config import Rain, expand_wildcards
from tqdm import tqdm

if __name__ == '__main__':
    configs = expand_wildcards(rain=Rain.Clear, sunset=False)
    pbar = tqdm(configs, ncols=200)
    for config in pbar:
        pbar.set_postfix_str(f"Downloading {config.pinhole_data.download_file}")
        print()
        # config.cylindrical_data.download()
        config.pinhole_data.download(force=True)
        config.pose_data.download(force=True)

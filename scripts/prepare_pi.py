"""Assemble the existing standalone Pi directory layout without changing model bytes."""
from pathlib import Path
import argparse
import shutil

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, default=ROOT / 'deployment/pi')
    args = parser.parse_args()
    target = args.destination.resolve()
    target.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'deployment/pi'
    if source.resolve() != target:
        for name in ('demo_replay.py', 'wifi_replay_receiver.py', 'live_ecg_pi_waveform.py',
                     'pi_app_bridge.py', 'requirements.txt', 'README.md'):
            shutil.copyfile(source / name, target / name)
        shutil.copytree(source / 'model', target / 'model', dirs_exist_ok=True)
    relative = Path('model_bundles/se_last_mask_seed7/code/src')
    shutil.copytree(ROOT / relative / 'afib_edge', target / '_training_src' / relative / 'afib_edge',
                    dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    print(f'Pi layout ready: {target}')
    print('Install requirements.txt in the Pi virtual environment before running receivers.')

if __name__ == '__main__':
    main()

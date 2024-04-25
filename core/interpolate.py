import torch
from pathlib import Path
from PIL import Image

def interpolate_result(result_path):

    result_path = Path(result_path)
    
    result_names = list(result_path.glob('*result.png'))

    interpolate_path = result_path.parent / 'output'
    interpolate_path.mkdir(parents= True , exist_ok= "True")

    for rname in result_names:
        result = Image.open(rname)
        upsample_img = result.resize((2232, 1488), resample= Image.LANCZOS)
        # upsample_img = result

        upsample_img.save('{}/{}.png'.format(str(interpolate_path), rname.stem.rsplit('_')[1].zfill(4)))

if __name__ == "__main__":
    interpolate_result('./experiments/defocus_240424_163039/results')
    

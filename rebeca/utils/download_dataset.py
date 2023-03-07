# A script to download OpenAI contractor data or BASALT datasets
# using the shared .json files (index file).
#
# Json files are in format:
# {"basedir": <prefix>, "relpaths": [<relpath>, ...]}
#
# The script will download all files in the relpaths list,
# or maximum of set number of demonstrations,
# to a specified directory.
#
# python utils/download_dataset.py --json-file utils/waterfall-Jul-28.json --output-dir data/MakeWaterfall/ --num-demos 1000

import cv2
import glob
import json
import argparse
import urllib.request
import requests
import os
from tqdm import tqdm

def relpaths_to_download(relpaths, output_dir):
    def read_json(file_name):
        with open(file_name.replace('mp4', 'jsonl'), 'r') as json_file:
            text = json.loads('['+''.join(json_file.readlines()).replace('\n', ',')+']')

    data_path = '/'.join(relpaths[0].split('/')[:-1])
    non_defect=[]
    for vid_name in glob.glob(os.path.join(output_dir,'*.mp4')):
        try:
            vid = cv2.VideoCapture(vid_name)
            read_json(vid_name.replace('mp4', 'jsonl'))
            if vid.isOpened():
                non_defect.append(os.path.join(data_path, vid_name.split('/')[-1]))
        except:
            continue

    relpaths = set(relpaths)
    non_defect = set(non_defect)
    diff_to_download = relpaths.difference(non_defect)
    print('total:', len(relpaths), '| exist:', len(non_defect), '| downloading:', len(diff_to_download))
    return diff_to_download

def download_file(url, outpath):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(outpath, 'wb') as f:
            pbar = tqdm(total=int(r.headers['Content-Length']), leave=False, unit='B', unit_scale=True, unit_divisor=1024)
            chunk_size = 256 * 1024
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    pbar.update(len(chunk))

def main(args):
    with open(args.json_file, "r") as f:
        data = f.read()
    data = eval(data)
    basedir = data["basedir"]
    relpaths = data["relpaths"]
    if args.num_demos is not None:
        relpaths = relpaths[:args.num_demos]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    relpaths = relpaths_to_download(relpaths, args.output_dir)

    main_pbar = tqdm(total=len(relpaths))

    for i, relpath in enumerate(relpaths):
        url = basedir + relpath
        filename = os.path.basename(relpath)
        outpath = os.path.join(args.output_dir, filename)
        percent_done = 100 * i / len(relpaths)
        # print(f"[{percent_done:.0f}%] Downloading {outpath}")
        try:
            # urllib.request.urlretrieve(url, outpath)
            download_file(url, outpath)
        except Exception as e:
            print(f"\tError downloading {url}: {e}. Cleaning up mp4")
            os.remove(outpath)

        # Also download corresponding .jsonl file
        jsonl_url = url.replace(".mp4", ".jsonl")
        jsonl_filename = filename.replace(".mp4", ".jsonl")
        jsonl_outpath = os.path.join(args.output_dir, jsonl_filename)
        try:
            urllib.request.urlretrieve(jsonl_url, jsonl_outpath)
        except Exception as e:
            print(f"\tError downloading {jsonl_url}: {e}. Cleaning up mp4")
            os.remove(outpath)
        
        main_pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download OpenAI contractor datasets")
    parser.add_argument("--json-file", type=str, required=True, help="Path to the index .json file")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--num-demos", type=int, default=None, help="Maximum number of demonstrations to download")
    args = parser.parse_args()
    main(args)
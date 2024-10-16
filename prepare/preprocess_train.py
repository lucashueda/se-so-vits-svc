import os
import random
import pandas as pd
import argparse

def print_error(info):
    print(f"\033[31m File isn't existed: {info}\033[0m")

## HARD CODED YET
def map_style_emolj(path):
    if("amused" in path):
        return 0
    elif("anger" in path):
        return 1
    elif("disgust" in path):
        return 2
    elif("sleepiness" in path):
        return 3
    else: # Neutral both to LJ and EMOVDB
        return 4

## HARD CODED YET
def map_style_cpqd(path):
    if("eps_animado" in path):
        return 0
    elif("eps_rispido" in path):
        return 1
    elif("eps_acolhedor" in path):
        return 2
    else: # Neutral for all others
        return 3

IndexBySinger = False
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--map", help="Mapping function", dest="map", required=True)

    args = parser.parse_args()
    print(f'Using mapping = {args.map}')

    os.makedirs("./files/", exist_ok=True)

    rootPath = "./data_svc/waves-32k/"
    all_items = []
    styles = [] # just to plot how many and which style do we have
    for spks in os.listdir(f"./{rootPath}"):
        if not os.path.isdir(f"./{rootPath}/{spks}"):
            continue
        print(f"./{rootPath}/{spks}")
        for file in os.listdir(f"./{rootPath}/{spks}"):
            if file.endswith(".wav"):
                file = file[:-4]

                if (IndexBySinger == False):
                    path_spk = f"./data_svc/speaker/{spks}/{file}.spk.npy"
                else:
                    path_spk = f"./data_svc/singer/{spks}.spk.npy"

                path_wave = f"./data_svc/waves-32k/{spks}/{file}.wav"
                path_spec = f"./data_svc/specs/{spks}/{file}.pt"
                path_pitch = f"./data_svc/pitch/{spks}/{file}.pit.npy"
                path_hubert = f"./data_svc/hubert/{spks}/{file}.vec.npy"
                path_whisper = f"./data_svc/whisper/{spks}/{file}.ppg.npy"

                if(args.map == 'emolj'):
                    style_id = map_style_emolj(path_wave)
                elif(args.map == 'cpqd'):
                    style_id = map_style_cpqd(path_wave)
                else:
                    print('ERROR use an valid argument!')

                has_error = 0
                if not os.path.isfile(path_spk):
                    print_error(path_spk)
                    has_error = 1
                if not os.path.isfile(path_wave):
                    print_error(path_wave)
                    has_error = 1
                if not os.path.isfile(path_spec):
                    print_error(path_spec)
                    has_error = 1
                if not os.path.isfile(path_pitch):
                    print_error(path_pitch)
                    has_error = 1
                if not os.path.isfile(path_hubert):
                    print_error(path_hubert)
                    has_error = 1
                if not os.path.isfile(path_whisper):
                    print_error(path_whisper)
                    has_error = 1
                if has_error == 0:
                    all_items.append(
                        f"{path_wave}|{path_spec}|{path_pitch}|{path_hubert}|{path_whisper}|{path_spk}|{style_id}")
                    styles.append(style_id)

    # Printing styles
    print("In total we have these styles value counts: ", pd.DataFrame({'styles': styles})['styles'].value_counts())

    random.shuffle(all_items)
    valids = all_items[:10]
    valids.sort()
    trains = all_items[10:]
    # trains.sort()
    fw = open("./files/valid.txt", "w", encoding="utf-8")
    for strs in valids:
        print(strs, file=fw)
    fw.close()
    fw = open("./files/train.txt", "w", encoding="utf-8")
    for strs in trains:
        print(strs, file=fw)
    fw.close()

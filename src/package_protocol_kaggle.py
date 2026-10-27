"""Package protocol verification code alongside the existing private data input."""
import argparse
import json
from pathlib import Path
import zipfile
from evaluation import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--username',default='deepamahuja')
    args=p.parse_args()
    folder=ROOT / 'kaggle/input/protocol_runners'
    folder.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((ROOT / 'configs/training_protocol.json').read_text())
    paths=list((ROOT / 'src').glob('*.py'))+[ROOT / 'configs/training_protocol.json',ROOT / 'LICENSE']
    paths += [ROOT / name for name in protocol['baseline_inventory']
              if name.startswith('results/large/') and (ROOT / name).exists()]
    with zipfile.ZipFile(folder / 'protocol_runner.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths: archive.write(path,path.relative_to(ROOT))
    dataset={'title':'Prompt detection protocol runners v1','id':f'{args.username}/prompt-detection-protocol-runners-v1',
             'licenses':[{'name':'MIT'}],'description':'Private experiment code, frozen protocol, and aggregate baseline reports. No raw prompts or model weights; uses the existing private WildJailbreak input.'}
    (folder / 'dataset-metadata.json').write_text(json.dumps(dataset,indent=2)+'\n')
    metadata={'id':f'{args.username}/prompt-detection-protocol-resume-check',
              'title':'Prompt detection protocol resume check','code_file':'train.py',
              'language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_internet':True,
              'machine_shape':'NvidiaTeslaT4',
              'docker_image':'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461',
              'docker_image_pinning_type':'original','dataset_sources':[f'{args.username}/wildjailbreak-100k-grouped',dataset['id']],
              'competition_sources':[],'kernel_sources':[]}
    (ROOT / 'kaggle/protocol_smoke/kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(folder)


if __name__=='__main__': main()

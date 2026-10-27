"""Create Kaggle inputs and notebook definitions for repeated neural-model training."""
import argparse
import json
from pathlib import Path
import zipfile
from evaluation import ROOT

BOOTSTRAP='''"""Train prompt classifiers from a committed source snapshot."""
from pathlib import Path
import json
import subprocess
import sys
import zipfile
archives=list(Path('/kaggle/input').rglob('training_source.zip'))
if len(archives)==1:
    source=Path('/kaggle/working/training_source')
    with zipfile.ZipFile(archives[0]) as archive: archive.extractall(source)
else:
    scripts=list(Path('/kaggle/input').rglob('src/run_kaggle_training.py'))
    if len(scripts)!=1: raise RuntimeError('Cannot locate the training source snapshot')
    source=scripts[0].parents[1]
TASKS = __TASKS__
subprocess.run([sys.executable,str(source/'src/run_kaggle_training.py'),
                '--source-dir',str(source),'--tasks',json.dumps(TASKS)],check=True)
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--username',default='deepamahuja')
    args=p.parse_args()
    folder=ROOT / 'kaggle/input/training_runners'; folder.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(folder / 'training_source.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in (args.source_dir / 'src').glob('*.py'): archive.write(path,path.relative_to(args.source_dir))
        for filename in ['configs/training_protocol.json','source_revision.json','LICENSE']:
            archive.write(args.source_dir / filename,filename)
    dataset={'title':'Prompt detection training runners','id':f'{args.username}/prompt-detection-training-runners',
        'licenses':[{'name':'MIT'}],'description':'Training code and configuration for repeated-seed prompt classification. Raw prompts and pretrained weights are provided by the existing private training input.'}
    (folder / 'dataset-metadata.json').write_text(json.dumps(dataset,indent=2)+'\n')
    schedules={
        'distilbert_training':('prompt-detection-distilbert-seeds','Prompt detection distilbert seeds',
            [{'model':'distilbert','seed':seed,'device':'cuda'} for seed in [4442,4443]]),
        'bilstm_training':('prompt-detection-bilstm-distilbert-seeds','Prompt detection bilstm distilbert seeds',
            [{'model':'bilstm','seed':seed,'device':'cuda'} for seed in [4442,4443,4444]]+
            [{'model':'distilbert','seed':4444,'device':'cuda'}]),
    }
    for directory,(slug,title,tasks) in schedules.items():
        kernel=ROOT / 'kaggle' / directory; kernel.mkdir(parents=True,exist_ok=True)
        (kernel / 'train.py').write_text(BOOTSTRAP.replace('__TASKS__',repr(tasks)))
        metadata={'id':f'{args.username}/{slug}','title':title,'code_file':'train.py','language':'python',
            'kernel_type':'script','is_private':True,'enable_gpu':True,'enable_internet':True,
            'machine_shape':'NvidiaTeslaT4','docker_image_pinning_type':'original',
            'docker_image':'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461',
            'dataset_sources':[f'{args.username}/wildjailbreak-100k-grouped',dataset['id']],
            'competition_sources':[],'kernel_sources':[]}
        (kernel / 'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(folder)


if __name__=='__main__': main()

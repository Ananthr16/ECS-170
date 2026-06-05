'''
Stage 5 convenience runner: trains and evaluates the GCN on all three datasets
(Cora, Citeseer, Pubmed) as separate subprocesses.
Run from the project root:  python script/stage_5_script/run_all_stage5.py
'''

import os
import subprocess
import sys


SCRIPTS = [
    'script_gcn_cora.py',
    'script_gcn_citeseer.py',
    'script_gcn_pubmed.py',
]


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    env = os.environ.copy()
    env['PYTHONPATH'] = template_root + os.pathsep + env.get('PYTHONPATH', '')

    for script_name in SCRIPTS:
        script_path = os.path.join(script_dir, script_name)
        print('************ Running', script_name, '************')
        subprocess.check_call([sys.executable, script_path], cwd=template_root, env=env)

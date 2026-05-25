import os
import subprocess
import sys


SCRIPTS = [
    'script_rnn_classification.py',
    'script_lstm_classification.py',
    'script_gru_classification.py',
    'script_rnn_generation.py',
    'script_lstm_generation.py',
    'script_gru_generation.py',
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

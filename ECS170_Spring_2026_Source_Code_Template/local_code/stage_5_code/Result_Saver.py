'''
Concrete ResultModule class for Stage 5 experiment outputs.
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import os
import pickle

from local_code.base_class.result import result


class Result_Saver(result):
    data = None
    fold_count = None
    result_destination_folder_path = None
    result_destination_file_name = None

    def save(self):
        print('saving results...')
        suffix = str(self.fold_count) if self.fold_count is not None else '0'
        path = self.result_destination_folder_path + self.result_destination_file_name + '_' + suffix
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)

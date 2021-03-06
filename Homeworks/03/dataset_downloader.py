import tarfile
import os
import urllib.request

DATASETS_REPO = 'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/'
ARC_EXT = '.tar.gz'
TEMP_ARC_FILENAME = 'temp' + ARC_EXT

class DatasetDownloader:
    def __init__(self, dataset_names, data_path):
        self.dataset_names = dataset_names
        self.data_path = data_path

    def download(self):
        for dataset_name in self.dataset_names:
            if self._is_downloaded(dataset_name):
                print(dataset_name, 'dataset is already downloaded')
                continue

            print('Downloading', dataset_name)
            dataset_url = DATASETS_REPO + dataset_name + ARC_EXT
            arc_file = os.path.join(self.data_path, TEMP_ARC_FILENAME)
            urllib.request.urlretrieve(dataset_url, arc_file)
            with tarfile.open(arc_file, 'r:gz') as arc:
                print('Extracting', dataset_name)
                arc.extractall(self.data_path)
            os.remove(arc_file)

    def _is_downloaded(self, dataset_name):
        expected_path = os.path.join(self.data_path, dataset_name)
        exists = os.path.exists(expected_path)
        is_dir = os.path.isdir(expected_path)
        return exists and is_dir

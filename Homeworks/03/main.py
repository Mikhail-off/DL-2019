from dataset_downloader import  DatasetDownloader

DATASET_NAMES = ['facades', 'summer2winter_yosemite']

DATA_PATH = 'data/'

def main():
    dataset_downloader = DatasetDownloader(DATASET_NAMES, DATA_PATH)
    dataset_downloader.download()

if __name__ == '__main__':
    main()
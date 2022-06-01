from data_loader import load_dataset, SROIEDataLoader


if __name__ == "__main__":
    load_dataset([SROIEDataLoader()], force=True)
from pathlib import Path
from .core import DataLoader, DataSet

dataset = ["car","iris","wine"]
datadir = Path(__file__).absolute().parent
datasetdir = [datadir.joinpath(i) for i in dataset]

Carloader,Irisloader,Wineloader = [DataLoader(str(ddir.joinpath('{dset}.names'.format(dset=dset))),
            str(ddir.joinpath('{dset}.data'.format(dset=dset)))) for ddir,dset in zip(
                datasetdir,dataset)]

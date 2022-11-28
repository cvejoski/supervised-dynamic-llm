from .dataloaders import CaltechPedastrianDataLoader, CityPersonsDataLoader, EuroCityPersonsDataLoader, ADataLoader

all_dataloaders = {
    "caltech": CaltechPedastrianDataLoader,
    "citypersons": CityPersonsDataLoader,
    "eurocitypersons": EuroCityPersonsDataLoader,
}

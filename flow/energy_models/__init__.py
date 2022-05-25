"""Contains all available energy models in Flow."""

# base network class
from flow.energy_models.base_energy import BaseEnergyModel

# custom energy models
from flow.energy_models.poly_fit_autonomie import PolyFitModel, PFMCompactSedan, PFMMidsizeSedan, PFM2019RAV4,\
    PFMMidsizeSUV, PFMLightDutyPickup, PFMClass3PNDTruck, PFMClass8TractorTrailer
from flow.energy_models.power_demand import PowerDemandModel, PDMCombustionEngine, PDMElectric
from flow.energy_models.toyota_energy import ToyotaModel, PriusEnergy, TacomaEnergy

__all__ = [
    "BaseEnergyModel",
    "PolyFitModel",
    "PFMCompactSedan",
    "PFMMidsizeSedan",
    "PFM2019RAV4",
    "PFMMidsizeSUV",
    "PFMLightDutyPickup",
    "PFMClass3PNDTruck",
    "PFMClass8TractorTrailer",
    "PowerDemandModel",
    "PDMCombustionEngine",
    "PDMElectric",
    "ToyotaModel",
    "PriusEnergy",
    "TacomaEnergy",
]

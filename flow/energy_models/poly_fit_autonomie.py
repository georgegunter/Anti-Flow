"""Script containing the vehicle power demand model energy classes."""
from abc import ABCMeta
import numpy as np
import os
from scipy.io import loadmat

from flow.energy_models.base_energy import BaseEnergyModel

GRAMS_PER_SEC_TO_GALS_PER_HOUR = 1.119  # 1.119 gal/hr = 1g/s

DIR_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model_coefficients")


def load_coeffs(filename, mass, conversion=33.43e3, v_max_fit=40):
    """Load in model coefficients from MATLAB files."""
    mat = loadmat(os.path.join(DIR_PATH, filename))
    mat = {key: val.item() for key, val in mat.items() if type(val) == np.ndarray}
    mat['mass'] = mass
    mat['conversion'] = conversion
    mat['v_max_fit'] = v_max_fit
    return mat


COMPACT_SEDAN_COEFFS = load_coeffs("Compact_coeffs.mat", mass=1450)
MIDSIZE_SEDAN_COEFFS = load_coeffs("midBase_coeffs.mat", mass=1743)
RAV4_2019_COEFFS = load_coeffs("RAV4_coeffs.mat", mass=1717)
MIDSIZE_SUV_COEFFS = load_coeffs("midSUV_coeffs.mat", mass=1897)
LIGHT_DUTY_PICKUP_COEFFS = load_coeffs("Pickup_coeffs.mat", mass=2173)
CLASS3_PND_TRUCK_COEFFS = load_coeffs("Class3PND_coeffs.mat", mass=5943)
CLASS8_TRACTOR_TRAILER_COEFFS = load_coeffs("Class8Tractor_coeffs.mat", mass=25104)


class PolyFitModel(BaseEnergyModel, metaclass=ABCMeta):
    """Simplified Polynomial Fit base energy model class.

    Calculate fuel consumption of a vehicle based on polynomial
    fit of Autonomie models. Polynomial functional form is
    informed by physics derivation and resembles power demand
    models.
    """

    def __init__(self, coeffs_dict):
        """
        Initialize the PolyFitModel class, using a dictionary of coefficients.

        It is not recommended to supply custom coefficients (it is also not possible to instantiate the PolyFitModel
        class directly since it is an abstract class). The child classes below instantiate versions of this class
        using the fitted coefficients provided in the *.mat files. For more details, see docs/PolyFitModels.pdf.

        Parameters
        ----------
        coeffs_dict : dict
            model coefficients, including:
            * "mass" (int | float): mass of the vehicle, for reference only
            * "C0" (float): C0 fitted parameter
            * "C1" (float): C1 fitted parameter
            * "C2" (float): C2 fitted parameter
            * "C3" (float): C3 fitted parameter
            * "p0" (float): p0 fitted parameter
            * "p1" (float): p1 fitted parameter
            * "p2" (float): p2 fitted parameter
            * "q0" (float): q0 fitted parameter
            * "q1" (float): q1 fitted parameter
            * "beta0" (float): minimum fuel consumption
            * "b1" (float): coeff 1 for modelling infeasible range
            * "b2" (float): coeff 2 for modelling infeasible range
            * "b3" (float): coeff 3 for modelling infeasible range
            * "b4" (float): coeff 4 for modelling infeasible range
            * "b5" (float): coeff 5 for modelling infeasible range
            * "conversion" (float): unit conversion from gal/hr to Watts
            * "v_max_fit" (float): assumed max velocity for modelling infeasible range
        """
        super(PolyFitModel, self).__init__()

        self.mass = coeffs_dict['mass']
        self.state_coeffs = np.array([coeffs_dict['C0'],
                                      coeffs_dict['C1'],
                                      coeffs_dict['C2'],
                                      coeffs_dict['C3'],
                                      coeffs_dict['p0'],
                                      coeffs_dict['p1'],
                                      coeffs_dict['p2'],
                                      coeffs_dict['q0'],
                                      coeffs_dict['q1']])
        self.beta0 = coeffs_dict['beta0']
        self.b1 = coeffs_dict['b1']
        self.b2 = coeffs_dict['b2']
        self.b3 = coeffs_dict['b3']
        self.b4 = coeffs_dict['b4']
        self.b5 = coeffs_dict['b5']
        self.conversion = coeffs_dict['conversion']
        self.v_max_fit = coeffs_dict['v_max_fit']

    def get_instantaneous_fuel_consumption(self, accel, speed, grade):
        """Calculate the instantaneous fuel consumption.

        Parameters
        ----------
        accel : float
            Instantaneous acceleration of the vehicle
        speed : float
            Instantaneous speed of the vehicle
        grade : float
            Instantaneous road grade of the vehicle

        Returns
        -------
        float
        """
        state_variables = np.array([1,
                                    speed,
                                    speed**2,
                                    speed**3,
                                    accel,
                                    accel*speed,
                                    accel*speed**2,
                                    max(accel, 0)**2,
                                    max(accel, 0)**2*speed])
        fc = np.dot(self.state_coeffs, state_variables)
        fc = max(fc, self.beta0)  # assign min fc when polynomial is below the min
        return fc * GRAMS_PER_SEC_TO_GALS_PER_HOUR

    def flag_infeasible_accel(self, accel, speed, grade):
        """Return True if speed/accel pair outside of feasible range."""
        max_accel = self.b4 * speed + self.b5
        if speed != 0:
            speed_ratio = speed / self.v_max_fit
            max_accel += self.b1 * speed_ratio**self.b2 * (1.0 - speed_ratio)**self.b3
        return accel > max_accel.real

    def get_instantaneous_power(self, accel, speed, grade):
        """See parent class."""
        return self.get_instantaneous_fuel_consumption(accel, speed, grade) * self.conversion


class PFMCompactSedan(PolyFitModel):
    """Simplified Polynomial Fit Model for CompactSedan.

    Model is fitted to semi-principled model derived from Autonomie software.
    """

    def __init__(self, coeffs_dict=COMPACT_SEDAN_COEFFS):
        super(PFMCompactSedan, self).__init__(coeffs_dict=coeffs_dict)


class PFMMidsizeSedan(PolyFitModel):
    """Simplified Polynomial Fit Model for MidsizeSedan.

    Model is fitted to semi-principled model derived from Autonomie software.
    """

    def __init__(self, coeffs_dict=MIDSIZE_SEDAN_COEFFS):
        super(PFMMidsizeSedan, self).__init__(coeffs_dict=coeffs_dict)


class PFM2019RAV4(PolyFitModel):
    """Simplified Polynomial Fit Model for 2019RAV4.

    Model is fitted to semi-principled model derived from Autonomie software.
    """

    def __init__(self, coeffs_dict=RAV4_2019_COEFFS):
        super(PFM2019RAV4, self).__init__(coeffs_dict=coeffs_dict)


class PFMMidsizeSUV(PolyFitModel):
    """Simplified Polynomial Fit Model for MidsizeSUV.

    Model is fitted to semi-principled model derived from Autonomie software.
    """

    def __init__(self, coeffs_dict=MIDSIZE_SUV_COEFFS):
        super(PFMMidsizeSUV, self).__init__(coeffs_dict=coeffs_dict)


class PFMLightDutyPickup(PolyFitModel):
    """Simplified Polynomial Fit Model for LightDutyPickup.

    Model is fitted to semi-principled model derived from Autonomie software.
    """

    def __init__(self, coeffs_dict=LIGHT_DUTY_PICKUP_COEFFS):
        super(PFMLightDutyPickup, self).__init__(coeffs_dict=coeffs_dict)


class PFMClass3PNDTruck(PolyFitModel):
    """Simplified Polynomial Fit Model for Class3PNDTruck.

    Model is fitted to semi-principled model derived from Autonomie software.
    """

    def __init__(self, coeffs_dict=CLASS3_PND_TRUCK_COEFFS):
        super(PFMClass3PNDTruck, self).__init__(coeffs_dict=coeffs_dict)


class PFMClass8TractorTrailer(PolyFitModel):
    """Simplified Polynomial Fit Model for Class8TractorTrailer.

    Model is fitted to semi-principled model derived from Autonomie software.
    """

    def __init__(self, coeffs_dict=CLASS8_TRACTOR_TRAILER_COEFFS):
        super(PFMClass8TractorTrailer, self).__init__(coeffs_dict=coeffs_dict)

"""Energy models."""
import math
import numpy as np


GRAMS_PER_SEC_TO_GALS_PER_HOUR = {
    'diesel': 1.119,  # 1.119 gal/hr = 1g/s
    'gasoline': 1.268,  # 1.268 gal/hr = 1g/s
}

GRAMS_TO_JOULES = {
    'diesel': 42470,
    'gasoline': 42360,
}

RAV4_2019_COEFFS = {
    'beta0': 0.013111753095302022,
    'vc': 5.98,
    'p1': 0.047436831067050676,
    'C0': 0.14631964767035743,
    'C1': 0.012179045946260292,
    'C2': 0,
    'C3': 2.7432588728174234e-05,
    'p0': 0.04553801347643801,
    'p2': 0.0018022443124799303,
    'q0': 0,
    'q1': 0.02609037187916979,
    'b1': 7.1780386096154185,
    'b2': 0.053537268955100234,
    'b3': 0.27965662935753677,
    'z0': 1.4940081773441736,
    'z1': 1.2718495543500672,
    'ver': '2.0',
    'mass': 1717,
    'fuel_type': 'gasoline'}


class PolyFitModel(object):
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
            * "z0" (float): z0 parameter for grade dependency
            * "z1" (float): z1 parameter for grade dependency
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
                                      coeffs_dict['q1'],
                                      coeffs_dict['z0'],
                                      coeffs_dict['z1']])
        self.beta0 = coeffs_dict['beta0']
        self.vc = coeffs_dict['vc']
        self.b1 = coeffs_dict['b1']
        self.b2 = coeffs_dict['b2']
        self.b3 = coeffs_dict['b3']
        self.fuel_type = coeffs_dict['fuel_type']

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
                                    accel * speed,
                                    accel * speed**2,
                                    max(accel, 0)**2,
                                    max(accel, 0)**2 * speed,
                                    grade,
                                    grade * speed])
        fc = np.dot(self.state_coeffs, state_variables)
        lower_bound = (speed <= self.vc) * self.beta0
        fc = max(fc, lower_bound)  # assign min fc when polynomial is below the min
        return fc * GRAMS_PER_SEC_TO_GALS_PER_HOUR[self.fuel_type]

    def flag_infeasible_accel(self, accel, speed, grade):
        """Return True if speed/accel pair outside of feasible range."""
        max_accel = self.b1 * math.exp(-self.b2 * speed) + self.b3
        return accel > max_accel.real

    def get_instantaneous_power(self, accel, speed, grade):
        """See parent class."""
        conversion = GRAMS_TO_JOULES[self.fuel_type] / GRAMS_PER_SEC_TO_GALS_PER_HOUR[self.fuel_type]
        return self.get_instantaneous_fuel_consumption(accel, speed, grade) * conversion


class PFM2019RAV4(PolyFitModel):
    """Simplified Polynomial Fit Model for 2019RAV4.

    Model is fitted to semi-principled model derived from Autonomie software.
    """

    def __init__(self, coeffs_dict=RAV4_2019_COEFFS):
        super(PFM2019RAV4, self).__init__(coeffs_dict=coeffs_dict)

# This file was made so that we don't need to reload the data everytime an API call to calculator.py is made.
import numpy as np

from ezpit.elem_tables import (
    AFF_ELEMENTS,
    AFF_PARAMETERS,
    COMPTON_ATOMIC_NUMBERS,
    COMPTON_ELEMENTS,
    COMPTON_PARAMETERS,
)


class ElementData:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):  # Prevent re-initialization
            return

        self.aff_element_dict = {name.lower(): i for i, name in enumerate(AFF_ELEMENTS)}
        self.aff_parm = np.array(AFF_PARAMETERS)

        self.compton_atomic_number = np.array(COMPTON_ATOMIC_NUMBERS)
        self.compton_element_dict = {name.lower(): i for i, name in enumerate(COMPTON_ELEMENTS)}
        self.compton_parameter_only = np.array(COMPTON_PARAMETERS)

        self._initialized = True
        ElementData._instance = self

    def aff_element_to_index(self):
        return self.aff_element_dict

    def compton_element_to_index(self):
        return self.compton_element_dict

    def get_aff_scattering_factors(self, atom_names):
        """
        Returns atomic form factor parameters for given atoms.
        """
        return np.array([self.aff_parm[self.aff_element_dict[name.lower()]] for name in atom_names])

    def get_compton_scattering_factors(self, atom_names):
        """
        Returns Compton scattering parameters and atomic numbers for given atoms.
        """
        scat_factors = []
        atomic_numbers = []
        for name in atom_names:
            idx = self.compton_element_dict.get(name.lower())
            if idx is None:
                raise ValueError(f"No Compton data for atom: {name}")
            scat_factors.append(self.compton_parameter_only[idx])
            atomic_numbers.append(self.compton_atomic_number[idx])
        return np.array(scat_factors), atomic_numbers

    def get_compton_parameter_only(self):
        return np.array(self.compton_parameter_only)

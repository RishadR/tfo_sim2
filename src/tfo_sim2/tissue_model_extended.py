"""
A set of special tissue models extending the base TissueModel class.
"""

from pathlib import Path
from typing import List
from tfo_sim2.four_layer_model_optical_props_table import *
from typing import Literal
import numpy as np
from tfo_sim2.tissue_model import TissueModel


class LapitanTissueModel(TissueModel):
    """
    A tissue model based on the works of Lapitan et al., representing a layered structure.

    From "Justification of the Photoplethysmography Sensor Configuration by Monte Carlo Modeling of the Pulse Waveform"

    From "Justification of the Photoplethysmography Sensor Configuration by Monte Carlo Modeling of the Pulse Waveform"

    The model length and width are fixed at 101 mm x 101 mm, while the depth is 58 mm.

    Layer ordering (top → bottom): Air → Epidermis → Dermis → Subcutaneous

    ASCII side-view (top is towards the detector/light source):

        top / detector
            ↓
        +────────────+   <- Air (tag 0 - upper z)
        |    AIR     |
        +────────────+
        | Epidermis  |   <- Epidermis (tag 1, ~1 mm, z = 57)
        +────────────+
        |   Dermis   |   <- Dermis (tag 2, ~7 mm)
        +────────────+
        | Subcutane- |   <- Subcutaneous / adipose (tag 3, ~50 mm)
        |   ous      |
        +────────────+
            ↑
         depth (bottom - lower z)
    """

    def __init__(
        self,
        wavelength: Literal[660, 810, 940] = 660,
        Vb_arterial: float = 0.05,
        Vb_venous: float = 0.05,
    ):
        """
        Initialize the Lapitan tissue model.

        Args:
            wavelength: Wavelength in nm (660, 810, or 940).
            Vb_arterial: Arterial blood volume fraction. (Refer to Fig. 2 from the paper)
            Vb_venous: Venous blood volume fraction. (Venous does not pulsate, so can be kept constant)

        Notes:
            This assumes a constant saturation.
        """
        super().__init__(name="LapitanTissueModel")
        self.wavelength = wavelength

        # Epidermis properties
        self.epi_thickness = 1  # mm
        self.epi_V_melanin = 0.1
        self.epi_V_water = 0.2
        self.epi_Vb_arterial = 0.0
        self.epi_Vb_venous = 0.0

        # Dermis properties
        self.derm_thickness = 7  # mm
        self.derm_V_melanin = 0.0
        self.derm_V_water = 0.6
        self.Vb_arterial = Vb_arterial
        self.Vb_venous = Vb_venous

        # Subcutene properties
        self.subcut_thickness = 50  # well technically infinite
        self.subcut_V_melanin = 0.0
        self.subcut_V_water = 0.15
        self.subcut_Vb_arterial = 0.025
        self.subcut_Vb_venous = 0.025

        # Optical Properties
        ## Mu_a
        self.mu_a_melanin_dict = {660: 26.94, 810: 13.62, 940: 8.3}  # in mm^-1
        self.mu_a_water_dict = {660: 0.00032, 810: 0.0267, 940: 0.036}
        self.mu_a_arterial_dict = {660: 0.195, 810: 0.402, 940: 0.643}
        self.mu_a_venous_dict = {660: 0.642, 810: 0.417, 940: 0.577}
        self.mu_a_fat_dict = {660: 0.065, 810: 0.138, 940: 0.144}

        ## Mu_s
        self.mu_s_epidermis_dict = {660: 22.74, 810: 18.4, 940: 16.1}
        self.mu_s_dermis_dict = {660: 13.95, 810: 11.11, 940: 9.74}
        self.mu_s_subcut_dict = {660: 12.28, 810: 10.27, 940: 9.2}

    def _mu_a_baseline(self) -> float:
        """
        Baseline absorption coefficient as a function of wavelength.

        Args:
            wavelength: Wavelength in nm.
        Returns:
            Baseline absorption coefficient.
        """
        # Placeholder implementation; replace with actual formula
        return 0.1 * 7.84 * 1e8 * self.wavelength**-3.255  # in mm^-1

    def _generate_properties(self):
        """
        Generate optical properties for the layered tissue model.
        Sets self._prop with the optical properties for each layer.
        """
        mu_a_melanin = self.mu_a_melanin_dict[self.wavelength]
        mu_a_water = self.mu_a_water_dict[self.wavelength]
        mu_a_arterial = self.mu_a_arterial_dict[self.wavelength]
        mu_a_venous = self.mu_a_venous_dict[self.wavelength]
        mu_a_fat = self.mu_a_fat_dict[self.wavelength]

        # Epidermis
        epi_V_rest = (
            1
            - self.epi_V_melanin
            - self.epi_V_water
            - self.epi_Vb_arterial
            - self.epi_Vb_venous
        )
        mu_a_epi = (
            self.epi_V_melanin * mu_a_melanin
            + self.epi_V_water * mu_a_water
            + epi_V_rest * self._mu_a_baseline()
        )
        mu_s_epi = self.mu_s_epidermis_dict[self.wavelength]
        g_epi = 0.9
        n_epi = 1.4

        # Dermis
        derm_V_rest = (
            1
            - self.derm_V_melanin
            - self.derm_V_water
            - self.Vb_arterial
            - self.Vb_venous
        )
        mu_a_derm = (
            self.derm_V_melanin * mu_a_melanin
            + self.derm_V_water * mu_a_water
            + self.Vb_arterial * mu_a_arterial
            + self.Vb_venous * mu_a_venous
            + derm_V_rest * self._mu_a_baseline()
        )
        mu_s_derm = self.mu_s_dermis_dict[self.wavelength]
        g_derm = 0.9
        n_derm = 1.4

        # Subcutene
        subcut_V_rest = (
            1
            - self.subcut_V_melanin
            - self.subcut_V_water
            - self.subcut_Vb_arterial
            - self.subcut_Vb_venous
        )
        mu_a_subcut = (
            +self.subcut_V_water * mu_a_water
            + self.subcut_Vb_arterial * mu_a_arterial
            + self.subcut_Vb_venous * mu_a_venous
            + subcut_V_rest * mu_a_fat
        )
        mu_s_subcut = self.mu_s_subcut_dict[self.wavelength]
        g_subcut = 0.9
        n_subcut = 1.4

        self._prop = [
            [0, 0, 1, 1],  # background
            [mu_a_epi, mu_s_epi, g_epi, n_epi],  # Epidermis
            [mu_a_derm, mu_s_derm, g_derm, n_derm],  # Dermis
            [mu_a_subcut, mu_s_subcut, g_subcut, n_subcut],  # Subcutene
        ]

    def _generate_volume(self):
        """
        Generate a layered volume representing the tissue structure.
        Sets self._vol with the tissue tags for each voxel.
        """
        epi_size = int(self.epi_thickness)  # assuming 1 mm per voxel
        derm_size = int(self.derm_thickness)
        subcut_size = int(self.subcut_thickness)

        vol_epi = np.ones((101, 101, epi_size), dtype="uint8") * 1
        vol_derm = np.ones((101, 101, derm_size), dtype="uint8") * 2
        vol_subcut = np.ones((101, 101, subcut_size), dtype="uint8") * 3
        vol_air = np.zeros((101, 101, 2), dtype="uint8")
        self._vol = np.concatenate([vol_subcut, vol_derm, vol_epi, vol_air], axis=2)


class DanModel4LayerX(TissueModel):
    """
    Create a slightly modified flat version of Daniel D Fong's ICCPS model. This model has 4 layers (As opposed to 8).
    The extended version allows for a wider range of avaialbe wavelengths(Passed to the model using the
    SimulationParameters.wavelength parameter) rather using the default waveint, which only allows for 2 wavelengths

    Current source wavelength range: 600nm to 1000nm (Affects the mu_a and mu_s)

    Model width and length are fixed at 221 mm x 221 mm
    get topmost pixel using self.topmost_pixel()

    Layer ordering (top → bottom): Air → Maternal Wall → Maternal Uterus → Amniotic Fluid → Fetal Tissue

    ASCII side-view (top is towards the detector/light source):

        top / detector
            ↓
        +──────────────+   <- Air (tag 0 - upper z)
        |     AIR      |
        +──────────────+
        | Maternal     |   <- Maternal Wall (tag 1, ~2 mm)
        | Wall         |
        +──────────────+
        | Maternal     |   <- Maternal Uterus (tag 2, ~4 mm)
        | Uterus       |
        +──────────────+
        | Amniotic     |   <- Amniotic Fluid (tag 3, ~1 mm)
        | Fluid        |
        +──────────────+
        | Fetal        |   <- Fetal Tissue (tag 4, ~50 mm)
        | Tissue       |
        +──────────────+
            ↑
         depth (bottom - lower z)


    """

    def __init__(
        self,
        wavelength: float,
        epi_thickness: int = 2,
        derm_thickness: int = 4,
        maternal_hb_conc: float = 15.0,
        maternal_saturation: float = 1.0,
        fetal_saturation: float = 0.6,
        fetal_hb_conc: float = 15.0,
        include_amniotic_fluid: bool = True,
    ):
        """
        Initialize the DanModel4LayerX tissue model.

        Args:
            wavelength: Wavelength in nm (between 600nm and 1000nm)
            epi_thickness: Maternal wall thickness in mm (default: 2mm)
            derm_thickness: Maternal uterus thickness in mm (default: 4mm)
            maternal_hb_conc: Maternal hemoglobin concentration (default: 15.0 g/dL)
            maternal_saturation: Maternal oxygen saturation (0-1, default: 1.0)
            fetal_saturation: Fetal oxygen saturation (0-1, default: 0.6)
            fetal_hb_conc: Fetal hemoglobin concentration (default: 15.0 g/dL)
            include_amniotic_fluid: Whether to include amniotic fluid layer (default: True)
        """
        super().__init__(name="Dan4LayerX")
        self.wavelength = wavelength
        self.maternal_hb_conc = maternal_hb_conc
        self.maternal_saturation = maternal_saturation
        self.fetal_hb_conc = fetal_hb_conc
        self.fetal_saturation = fetal_saturation
        self.include_amniotic_fluid = include_amniotic_fluid
        self.epi_thickness = epi_thickness
        self.derm_thickness = derm_thickness
        self.fetal_thickness = 50  # fixed 50mm fetal tissue layer
        self._prop = None

    def _generate_properties(self):
        ## Sanity Check
        if self.wavelength < 600 or self.wavelength > 1000:
            raise ValueError("Source wavelength should be between 600nm and 1000nm")

        ## Get interpolated values for the optical properties
        l1_mu_s = get_maternal_wall_mu_s(self.wavelength)
        l1_mu_a = get_blood_filled_tissue_mu_a(
            0.1,
            self.maternal_hb_conc,
            self.maternal_saturation,
            self.wavelength,
        )

        l2_mu_s = get_maternal_uterus_mu_s(self.wavelength)
        l2_mu_a = get_maternal_uterus_mu_a(self.wavelength)

        l3_mu_s = get_water_mu_s(self.wavelength)
        l3_mu_a = get_water_mu_a(self.wavelength)

        l4_mu_s = get_fetal_tissue_mu_s(self.wavelength)
        l4_mu_a = get_blood_filled_tissue_mu_a(
            0.1, self.fetal_hb_conc, self.fetal_saturation, self.wavelength
        )

        air_properties = [0.0, 0.0, 1.0, 1.0]
        self._prop = [
            air_properties,
            [l1_mu_a, l1_mu_s, 0.9, 1.4],
            [l2_mu_a, l2_mu_s, 0.9, 1.4],
            [l3_mu_a, l3_mu_s, 0.9, 1.33],
            [l4_mu_a, l4_mu_s, 0.9, 1.4],
        ]

    def _generate_volume(self):
        epi_size = int(self.epi_thickness)  # assuming 1 mm per voxel
        derm_size = int(self.derm_thickness)
        amniotic_size = 1 if self.include_amniotic_fluid else 0
        fetal_size = self.fetal_thickness
        side_span = 221

        vol_epi = np.ones((side_span, side_span, epi_size), dtype="uint8") * 1
        vol_derm = np.ones((side_span, side_span, derm_size), dtype="uint8") * 2
        vol_amniotic = np.ones((side_span, side_span, amniotic_size), dtype="uint8") * 3
        vol_fetal = np.ones((side_span, side_span, fetal_size), dtype="uint8") * 4
        vol_air = np.zeros((side_span, side_span, 2), dtype="uint8")
        self._vol = np.concatenate(
            [vol_fetal, vol_amniotic, vol_derm, vol_epi, vol_air], axis=2
        )

    def topmost_pixel(self) -> int:
        height = self.fetal_thickness + self.derm_thickness + self.epi_thickness
        height += 1 if self.include_amniotic_fluid else 0
        return height - 1


__all__ = ["LapitanTissueModel", "DanModel4LayerX"]

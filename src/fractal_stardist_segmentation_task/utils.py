"""Pydantic models for advanced iterator configuration."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class MaskingConfiguration(BaseModel):
    """Masking configuration."""

    mode: Literal["Table Name", "Label Name"] = "Table Name"
    """
    Mode of masking to be applied. If "Table Name", the identifier refers to a
    masking table name.
    If "Table Name", the identifier refers to a masking table name.
    If "Label Name", the identifier refers to a label image name.
    """
    identifier: str | None = None
    """
    Name of the masking table or label image depending on the mode.
    """


class IteratorConfiguration(BaseModel):
    """Advanced Masking configuration."""

    masking: MaskingConfiguration | None = Field(
        default=None, title="Masking Iterator Configuration"
    )
    """
    If configured, the segmentation will be only saved within the mask region.
    will be applied based on the provided masking configuration.
    """
    roi_table: str | None = Field(default=None, title="Iterate Over ROIs")
    """
    Name of a ROI table. If provided, the segmentation will be performed for each ROI
    in the specified ROI table.
    """


class CreateMaskingRoiTable(BaseModel):
    """Create Masking ROI Table Configuration.

    Attributes:
        mode (Literal["Create Masking ROI Table"]): Mode to create masking ROI table.
        table_name (str): Name of the masking ROI table to be created.
            Defaults to "{label_name}_masking_ROI_table", where {label_name} is
            the name of the label image used for segmentation.
    """

    mode: Literal["Create Masking ROI Table"] = "Create Masking ROI Table"
    table_name: str = "{label_name}_masking_ROI_table"

    def get_table_name(self, label_name: str) -> str:
        """Get the actual table name by replacing placeholder.

        Args:
            label_name (str): Name of the label image used for segmentation.

        Returns:
            str: Actual name of the masking ROI table.
        """
        return self.table_name.format(label_name=label_name)


class SkipCreateMaskingRoiTable(BaseModel):
    """Skip Creating Masking ROI Table Configuration.

    Attributes:
        mode (Literal["Skip Creating Masking ROI Table"]): Mode to skip creating
            masking ROI table.
    """

    mode: Literal["Skip Creating Masking ROI Table"] = "Skip Creating Masking ROI Table"


AnyCreateRoiTableModel = Annotated[
    CreateMaskingRoiTable | SkipCreateMaskingRoiTable,
    Field(discriminator="mode"),
]

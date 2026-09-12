"""axolotl.integrations.stablemax.args

Pydantic model definitions for StableMax integration arguments.

This module provides the StableMaxArgs model used to configure the StableMax
integration within Axolotl.
"""

from pydantic import BaseModel, Field


class StableMaxArgs(BaseModel):
    """
    Arguments for enabling the StableMax integration.
    """

    stablemax: bool = Field(
        default=False,
        description="Enable StableMax as a numerically stable alternative to softmax cross-entropy loss.",
    )

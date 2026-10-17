from pydantic import BaseModel, Field

"""
StableMax integration configuration arguments.
"""


class StableMaxArgs(BaseModel):
    """
    Arguments for enabling the StableMax integration.
    """

    stablemax: bool = Field(
        default=False,
        description="Enable StableMax as a numerically stable alternative to softmax cross-entropy loss.",
    )

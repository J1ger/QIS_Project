"""Standalone exceptions for the data_processing module."""

from __future__ import annotations


class QuantModuleError(Exception):
    """Base exception for standalone quantitative modules."""


class ProviderError(QuantModuleError):
    """Raised when data acquisition or provider normalization fails."""


class DataValidationError(QuantModuleError):
    """Raised when input data does not satisfy required schema or quality rules."""

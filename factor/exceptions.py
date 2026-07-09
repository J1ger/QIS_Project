"""Standalone exceptions for the factor module."""

from __future__ import annotations


class QuantModuleError(Exception):
    """Base exception for standalone quantitative modules."""


class DataValidationError(QuantModuleError):
    """Raised when factor computation input data is incomplete or invalid."""

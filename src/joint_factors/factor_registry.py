"""
Factor registry for managing multiple factors

This module provides a central registry for all factors in the framework,
making it easy to add factor3 and beyond.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ..utils.config_loader import load_factors_config
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class FactorSpec:
    """Specification for a single factor"""
    name: str
    description: str
    source_project: str
    enabled: bool
    columns: Dict[str, str]
    properties: Dict[str, Any]
    normalization: Dict[str, Any]


class FactorRegistry:
    """
    Central registry for all factors
    
    Usage:
        registry = get_factor_registry()
        enabled_factors = registry.get_enabled_factors()
        manip_spec = registry.get_factor('manip')
    """
    
    def __init__(self):
        self._factors: Dict[str, FactorSpec] = {}
        self._load_from_config()
    
    def _load_from_config(self):
        """Load factor specifications from config"""
        config = load_factors_config()
        factors_config = config.get('factors', {})
        
        for factor_name, factor_cfg in factors_config.items():
            spec = FactorSpec(
                name=factor_cfg.get('name', factor_name),
                description=factor_cfg.get('description', ''),
                source_project=factor_cfg.get('source_project', ''),
                enabled=factor_cfg.get('enabled', False),
                columns=factor_cfg.get('columns', {}),
                properties=factor_cfg.get('properties', {}),
                normalization=factor_cfg.get('normalization', {}),
            )
            
            self._factors[factor_name] = spec
            
            if spec.enabled:
                logger.info(f"Registered factor: {factor_name} ({spec.name})")
            else:
                logger.debug(f"Factor {factor_name} registered but disabled")
    
    def get_factor(self, factor_name: str) -> Optional[FactorSpec]:
        """Get factor specification by name"""
        return self._factors.get(factor_name)
    
    def get_enabled_factors(self) -> List[str]:
        """Get list of enabled factor names"""
        return [
            name for name, spec in self._factors.items()
            if spec.enabled
        ]
    
    def get_all_factors(self) -> List[str]:
        """Get list of all factor names (enabled or not)"""
        return list(self._factors.keys())
    
    def is_enabled(self, factor_name: str) -> bool:
        """Check if a factor is enabled"""
        spec = self.get_factor(factor_name)
        return spec.enabled if spec else False
    
    def get_column_name(self, factor_name: str, column_type: str) -> Optional[str]:
        """
        Get column name for a factor
        
        Args:
            factor_name: Factor name (e.g., 'manip', 'ofi')
            column_type: Column type (e.g., 'raw', 'z_score', 'abs_z_score')
            
        Returns:
            Column name or None if not found
        """
        spec = self.get_factor(factor_name)
        if spec:
            return spec.columns.get(column_type)
        return None
    
    def get_factor_direction(self, factor_name: str) -> Optional[str]:
        """
        Get expected direction for a factor
        
        Returns:
            'reversal', 'momentum', or None
        """
        spec = self.get_factor(factor_name)
        if spec:
            return spec.properties.get('direction')
        return None


# Singleton instance
_registry: Optional[FactorRegistry] = None


def get_factor_registry() -> FactorRegistry:
    """Get the global factor registry instance"""
    global _registry
    if _registry is None:
        _registry = FactorRegistry()
    return _registry


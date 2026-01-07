"""
Hedonic valuation model for property price estimation.

This module implements a simplified hedonic pricing model that estimates
fair market value based on property characteristics and local market statistics.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass

from data.schemas import Property
from analytics.market_insights import MarketInsights


@dataclass
class ValuationResult:
    """Result of a property valuation."""
    estimated_price: float
    price_delta: float  # Actual - Estimated
    delta_percent: float  # (Actual - Estimated) / Estimated
    confidence: float  # 0.0 to 1.0
    valuation_status: str  # "undervalued", "fair", "overvalued"
    factors: Dict[str, float]  # Contribution of each factor to price


class HedonicValuationModel:
    """
    Estimates property value based on characteristics and market data.
    
    Uses a component-based approach:
    Price = (Base_Sqm_Price * Area * Loc_Factor) + 
            (Room_Value * Rooms) + 
            Amenity_Adjustments + 
            Condition_Adjustments
    """
    
    def __init__(self, market_insights: MarketInsights):
        self.market_insights = market_insights
        
        # Coefficients (could be learned, but using heuristics for now)
        self.amenity_premium = {
            "has_parking": 0.05,  # +5%
            "has_garden": 0.08,   # +8%
            "has_elevator": 0.03, # +3%
            "furnished": 0.10     # +10%
        }
        
        self.energy_premium = {
            "A": 0.10, "B": 0.07, "C": 0.04,
            "D": 0.00, "E": -0.03, "F": -0.06, "G": -0.10
        }
        
        self.year_premium_per_year = 0.005 # +0.5% per year newer than baseline (e.g., 1980)

    def predict_fair_price(self, property_data: Property) -> ValuationResult:
        """
        Predict the fair market price for a property.
        """
        # Get local market stats
        city = property_data.city
        
        # Default to global stats if city not found
        # In a real app, we'd handle this more gracefully
        base_price_sqm = 0.0
        
        # Try to get specific city stats
        trend = self.market_insights.get_price_trend(city)
        if trend and trend.average_price > 0:
            # Estimate sqm price from average price (rough approximation if sqm avg missing)
            # Ideally MarketInsights should provide avg_price_per_sqm per city
            # Let's check if we can get better data. 
            # For now, assume a standard size of 60sqm if not available to derive base
            base_price_sqm = trend.average_price / 60.0 
        
        # If we have direct access to the dataframe in market_insights, we could compute it better.
        # But let's assume we use the public API of MarketInsights.
        # For this implementation, let's look at `get_location_insights` if it exists or similar.
        
        # Let's fallback to a simpler logic:
        # If we can't get reliable local data, we can't value it reliably.
        if base_price_sqm == 0:
             return ValuationResult(0, 0, 0, 0, "unknown", {})

        # 1. Base Value (Area based)
        area = property_data.area_sqm if property_data.area_sqm else 50.0 # Fallback
        base_value = base_price_sqm * area
        
        # 2. Adjustments
        adjustments = 0.0
        factors = {"base_value": base_value}
        
        # Amenities
        multiplier = 1.0
        if property_data.has_parking:
            multiplier += self.amenity_premium["has_parking"]
        if property_data.has_garden:
            multiplier += self.amenity_premium["has_garden"]
        if property_data.has_elevator:
            multiplier += self.amenity_premium["has_elevator"]
            
        # Energy
        if property_data.energy_rating in self.energy_premium:
            multiplier += self.energy_premium[property_data.energy_rating]
            
        # Year Built (Simple linear decay/appreciation from 2000)
        if property_data.year_built:
            age_diff = property_data.year_built - 2000
            # Cap at +/- 20%
            age_factor = max(min(age_diff * 0.002, 0.2), -0.2)
            multiplier += age_factor
            
        final_price = base_value * multiplier
        
        # Calculate Delta
        actual_price = property_data.price if property_data.price else final_price
        delta = actual_price - final_price
        delta_percent = delta / final_price if final_price > 0 else 0
        
        # Determine Status
        if delta_percent < -0.10:
            status = "highly_undervalued"
        elif delta_percent < -0.05:
            status = "undervalued"
        elif delta_percent > 0.10:
            status = "highly_overvalued"
        elif delta_percent > 0.05:
            status = "overvalued"
        else:
            status = "fair"
            
        return ValuationResult(
            estimated_price=final_price,
            price_delta=delta,
            delta_percent=delta_percent,
            confidence=0.7, # Placeholder confidence
            valuation_status=status,
            factors=factors
        )

    def bulk_value(self, properties: List[Property]) -> List[ValuationResult]:
        """Value a list of properties."""
        return [self.predict_fair_price(p) for p in properties]

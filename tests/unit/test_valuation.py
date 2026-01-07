import pytest
from unittest.mock import MagicMock
from analytics.valuation_model import HedonicValuationModel, ValuationResult
from data.schemas import Property
from analytics.market_insights import MarketInsights, PriceTrend

@pytest.fixture
def mock_market_insights():
    insights = MagicMock(spec=MarketInsights)
    return insights

@pytest.fixture
def valuation_model(mock_market_insights):
    return HedonicValuationModel(mock_market_insights)

def test_predict_fair_price_basic(valuation_model, mock_market_insights):
    # Setup mock trend
    mock_trend = MagicMock(spec=PriceTrend)
    mock_trend.average_price = 300000  # For 60sqm -> 5000/sqm
    mock_market_insights.get_price_trend.return_value = mock_trend
    
    prop = Property(
        id="p1", 
        title="Test Prop",
        city="Warsaw",
        price=250000,
        area_sqm=50,
        rooms=2,
        url="http://example.com"
    )
    
    result = valuation_model.predict_fair_price(prop)
    
    assert isinstance(result, ValuationResult)
    # Base: 300000/60 = 5000/sqm. 50sqm * 5000 = 250,000 base.
    # No amenities -> 250,000.
    # Price is 250,000. Delta should be 0.
    
    assert result.estimated_price == 250000
    assert result.price_delta == 0
    assert result.valuation_status == "fair"

def test_predict_fair_price_with_amenities(valuation_model, mock_market_insights):
    # Setup mock trend
    mock_trend = MagicMock(spec=PriceTrend)
    mock_trend.average_price = 300000  # 5000/sqm
    mock_market_insights.get_price_trend.return_value = mock_trend
    
    prop = Property(
        id="p2",
        title="Luxury Prop",
        city="Warsaw",
        price=300000,
        area_sqm=50,
        rooms=2,
        has_parking=True,  # +5%
        has_garden=True,   # +8%
        energy_rating="A",  # +10%
        url="http://example.com"
    )
    
    # Base: 250,000
    # Multiplier: 1 + 0.05 + 0.08 + 0.10 = 1.23
    # Estimated: 250,000 * 1.23 = 307,500
    
    result = valuation_model.predict_fair_price(prop)
    
    assert result.estimated_price == pytest.approx(307500)
    # Actual 300,000 vs Est 307,500 -> Undervalued by 7,500 (~2.4%)
    # Delta % = -2.4%. Threshold for "undervalued" is -5%. So "fair".
    
    assert result.valuation_status == "fair"
    assert result.price_delta == pytest.approx(-7500)

def test_predict_fair_price_undervalued(valuation_model, mock_market_insights):
    # Setup mock trend
    mock_trend = MagicMock(spec=PriceTrend)
    mock_trend.average_price = 300000  # 5000/sqm
    mock_market_insights.get_price_trend.return_value = mock_trend
    
    prop = Property(
        id="p3",
        title="Cheap Prop",
        city="Warsaw",
        price=200000, # Actual
        area_sqm=50,
        rooms=2,
        url="http://example.com"
    )
    
    # Est: 250,000
    # Delta: 200k - 250k = -50k (-20%)
    
    result = valuation_model.predict_fair_price(prop)
    
    assert result.valuation_status == "highly_undervalued"

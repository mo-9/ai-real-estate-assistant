from datetime import datetime, timedelta
from data.schemas import Property, PropertyCollection
from ui.geo_viz import create_historical_trends_map

def test_create_historical_trends_map_returns_map():
    prop = Property(
        city="Warsaw",
        price=5000,
        rooms=2,
        scraped_at=datetime.now()
    )
    coll = PropertyCollection(properties=[prop], total_count=1)
    m = create_historical_trends_map(coll)
    assert m is not None
    # Folium map objects are complex, but we can check if it has children
    assert m.get_root() is not None

def test_create_historical_trends_map_with_old_property():
    old_date = datetime.now() - timedelta(days=400)
    prop = Property(
        city="Krakow",
        price=4000,
        scraped_at=old_date
    )
    coll = PropertyCollection(properties=[prop], total_count=1)
    m = create_historical_trends_map(coll)
    assert m is not None

def test_create_historical_trends_map_handles_missing_date():
    prop = Property(
        city="Wroclaw",
        price=3000,
        scraped_at=None 
    )
    coll = PropertyCollection(properties=[prop], total_count=1)
    m = create_historical_trends_map(coll)
    assert m is not None

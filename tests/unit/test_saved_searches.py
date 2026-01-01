from utils.saved_searches import SavedSearch


def test_matches_basic_filters():
    ss = SavedSearch(id="s1", name="Budget Krakow", city="Krakow", max_price=1000, min_rooms=2)
    prop = {
        "city": "Krakow",
        "price": 950,
        "rooms": 2,
        "has_parking": True,
        "is_furnished": True,
        "property_type": "apartment",
    }
    assert ss.matches(prop) is True

    prop_bad = {**prop, "price": 1200}
    assert ss.matches(prop_bad) is False


def test_to_query_string_human_readable():
    ss = SavedSearch(id="s1", name="Family", city="Warsaw", min_rooms=2, max_rooms=3, min_price=800, max_price=1500, must_have_parking=True, must_be_furnished=True, property_types=["apartment", "house"])
    q = ss.to_query_string()
    assert "in Warsaw" in q
    assert "with 2-3 rooms" in q
    assert "priced between $800-$1500" in q
    assert "parking" in q and "furnished" in q
    assert "(apartment, house)" in q


def test_matches_year_built_and_energy_cert_filters():
    ss = SavedSearch(
        id="s1",
        name="Modern Efficient",
        year_built_min=2000,
        year_built_max=2020,
        energy_certs=["a", "B"],
    )
    prop = {"year_built": "2010", "energy_cert": "b"}
    assert ss.matches(prop) is True

    assert ss.matches({**prop, "year_built": 1990}) is False
    assert ss.matches({**prop, "energy_cert": "C"}) is False


# API Reference

This document provides a reference for the core Python APIs of the AI Real Estate Assistant.

## Analytics

### HedonicValuationModel

**Module**: `analytics.valuation_model`

Estimates the fair market value of a property using a component-based hedonic pricing model.

#### `HedonicValuationModel(market_insights)`

*   **Args**:
    *   `market_insights` (MarketInsights): Instance of market insights engine to retrieve local price data.

#### Methods

*   `predict_fair_price(property: Property) -> ValuationResult`
    *   Calculates estimated price, price delta, and valuation status.
    *   **Returns**: `ValuationResult` object containing:
        *   `estimated_price`: Predicted fair price.
        *   `price_delta`: Difference between listing price and estimated price.
        *   `delta_percent`: Percentage difference.
        *   `valuation_status`: "undervalued", "fair", "overvalued", etc.
        *   `confidence`: Confidence score (0.0 - 1.0).

---

## Vector Store

### StrategicReranker

**Module**: `vector_store.reranker`

Reranks search results based on semantic relevance, metadata, and high-level user strategies.

#### `StrategicReranker(valuation_model=None, ...)`

*   **Args**:
    *   `valuation_model` (Optional[HedonicValuationModel]): Model to identify undervalued properties for "investor" strategy.
    *   `boost_exact_matches` (float): Boost for keyword matches (default: 1.5).
    *   `boost_metadata_match` (float): Boost for user preference matches (default: 1.3).

#### Methods

*   `rerank_with_strategy(query, documents, strategy="balanced", initial_scores=None, k=None)`
    *   Applies strategy-specific boosting logic.
    *   **Strategies**:
        *   `"investor"`: Prioritizes ROI, yield, and undervalued properties.
        *   `"family"`: Prioritizes size (rooms, area), amenities (garden, parking), and safety.
        *   `"bargain"`: Prioritizes lowest absolute price.
        *   `"balanced"`: Standard relevance ranking.
    *   **Returns**: List of `(Document, score)` tuples.

---

## Data

### Property

**Module**: `data.schemas`

Pydantic model representing a real estate listing.

*   **Fields**:
    *   `id` (str): Unique ID.
    *   `title` (str): Title (min 5 chars).
    *   `price` (float): Listing price.
    *   `area_sqm` (float): Area in square meters.
    *   `city` (str): City name.
    *   `rooms` (float): Number of rooms.
    *   `year_built` (int): Year of construction.
    *   `points_of_interest` (List[PointOfInterest]): Nearby POIs.
    *   ... (see source for full list)

### PointOfInterest

**Module**: `data.schemas`

Represents a nearby location of interest.

*   **Fields**:
    *   `name` (str): Name of the place.
    *   `category` (str): Type (school, park, transport, etc.).
    *   `distance_meters` (float): Distance from property.
    *   `latitude` (float): Geo-coordinate.
    *   `longitude` (float): Geo-coordinate.
    *   `tags` (Dict[str, str]): Additional OSM tags.

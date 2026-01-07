# Developer Notes

## Notifications System - Digest Generator

### Overview
The digest system is built around `DigestGenerator`, which bridges the gap between raw property data (`VectorStore`) and user-facing notifications (`AlertManager`, `EmailService`).

### Key Components

1.  **DigestGenerator (`notifications/digest_generator.py`)**
    *   **Purpose**: Gathers and structures data for email digests.
    *   **Inputs**: `UserPreferences`, `SavedSearch` list, `MarketInsights` engine.
    *   **Outputs**: Dictionary payload compatible with `DigestTemplate`.
    *   **Logic**:
        *   Iterates through `SavedSearch` objects.
        *   Queries `ChromaPropertyStore` for new matches.
        *   Deduplicates properties across searches.
        *   Fetches market trends from `MarketInsights` (Expert mode).
        *   Aggregates data into `top_picks`, `saved_searches`, and `expert` sections.

2.  **AlertManager (`notifications/alert_manager.py`)**
    *   **Role**: Orchestrator.
    *   **Method**: `process_digest(...)`
    *   **Responsibility**:
        *   Instantiates `DigestGenerator`.
        *   Calls `generate_digest()`.
        *   Passes result to `send_digest()`.
        *   Handles errors and logging.

3.  **Email Templates (`notifications/email_templates.py`)**
    *   **Class**: `DigestTemplate`
    *   **Role**: Rendering.
    *   **Features**:
        *   Responsive HTML layout.
        *   Conditional rendering for "Expert" sections (tables, indices).
        *   Property card components.

### Extension Points

*   **New Data Sources**: To add more data (e.g., mortgage rates), inject a new service into `DigestGenerator.__init__` and update `generate_digest` to populate the payload.
*   **Custom Filtering**: Modify `DigestGenerator._build_filters` to support more complex `SavedSearch` criteria (e.g., polygon search).
*   **Performance**: Currently, `generate_digest` runs sequentially. for large user bases, consider parallelizing search execution or pre-computing matches.

### Testing
*   **Unit Tests**: `tests/unit/test_digest_generator.py` covers basic data gathering and logic.
*   **Integration Tests**: `tests/integration/test_digest_generator_integration.py` verifies the flow from `AlertManager` to `DigestGenerator`.

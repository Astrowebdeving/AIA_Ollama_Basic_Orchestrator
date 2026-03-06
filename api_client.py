"""
API Client Skeleton
===================
Provides a structured foundation for fetching and parsing JSON
data from external REST APIs.

** All methods are stubs with TODOs — implement as needed. **
"""

from typing import Any, Optional


class ApiClient:
    """
    Async HTTP client for JSON APIs.

    Usage (once implemented):
        client = ApiClient(base_url="https://api.example.com/v1")
        data = await client.fetch_json("/users", params={"page": 1})
    """

    def __init__(
        self,
        base_url: str = "",
        headers: Optional[dict[str, str]] = None,
        timeout: float = 30.0,
    ):
        """
        Configure the API client.

        Parameters
        ----------
        base_url : str
            Root URL for the API (e.g. "https://api.example.com/v1").
            All endpoint paths are appended to this.
        headers : dict, optional
            Default headers sent with every request (e.g. auth tokens).
        timeout : float
            Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.timeout = timeout
        self._client = None  # TODO: initialise httpx.AsyncClient here

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------

    async def connect(self):
        """
        Initialise the underlying HTTP client.

        TODO:
          - import httpx
          - self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.headers,
                timeout=self.timeout,
            )
        """
        raise NotImplementedError("TODO: initialise httpx.AsyncClient")

    async def close(self):
        """
        Gracefully close the HTTP client.

        TODO:
          - await self._client.aclose()
        """
        raise NotImplementedError("TODO: close httpx.AsyncClient")

    # ----------------------------------------------------------------
    # GET requests
    # ----------------------------------------------------------------

    async def fetch_json(
        self,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
    ) -> dict:
        """
        Make a GET request and return parsed JSON.

        Parameters
        ----------
        endpoint : str
            API path (e.g. "/users/123"). Appended to base_url.
        params : dict, optional
            URL query parameters.

        Returns
        -------
        dict
            Parsed JSON response.

        TODO:
          - response = await self._client.get(endpoint, params=params)
          - return self._handle_response(response)
        """
        raise NotImplementedError("TODO: implement GET request")

    # ----------------------------------------------------------------
    # POST requests
    # ----------------------------------------------------------------

    async def post_json(
        self,
        endpoint: str,
        payload: Optional[dict[str, Any]] = None,
    ) -> dict:
        """
        Make a POST request with a JSON body and return parsed JSON.

        Parameters
        ----------
        endpoint : str
            API path (e.g. "/users").
        payload : dict, optional
            JSON body to send.

        Returns
        -------
        dict
            Parsed JSON response.

        TODO:
          - response = await self._client.post(endpoint, json=payload)
          - return self._handle_response(response)
        """
        raise NotImplementedError("TODO: implement POST request")

    # ----------------------------------------------------------------
    # Response handling
    # ----------------------------------------------------------------

    def _handle_response(self, response) -> dict:
        """
        Validate an HTTP response and extract the JSON body.

        TODO:
          - Check response.status_code (raise on 4xx/5xx)
          - Handle rate-limiting (429) with backoff
          - Return self._parse_json(response.text)
        """
        raise NotImplementedError("TODO: implement response handling")

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """
        Parse a raw JSON string into a dict with error handling.

        TODO:
          - import json
          - try: return json.loads(raw)
          - except json.JSONDecodeError as exc:
          -     raise ValueError(f"Invalid JSON: {exc}") from exc
        """
        raise NotImplementedError("TODO: implement JSON parsing")

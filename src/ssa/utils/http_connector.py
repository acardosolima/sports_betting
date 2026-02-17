# Copyright (C) 2026 Adriano Lima
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from src.ssa.utils.logger import Logger as custom_logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class HTTPConnector:
    """
    A class to handle HTTP requests, including parallel requests.

    Attributes:
        base_url (str): The base URL for the API.
        auth_token (Optional[str]): The authentication token for the API.
        retry_strategy (Retry): The retry strategy for failed requests.
        session (requests.Session): The session object for making requests.
    """

    def __init__(
        self,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        auth_token: Optional[str] = None,
        max_retries: int = 3,
        backoff_factor: float = 0.3,
        status_forcelist: Optional[List[int]] = None,
        log_level: int = "WARNING",
    ):
        """
        Initializes the HttpClient.

        Args:
            base_url (str): The base URL for the API.
            headers (Optional[Dict[str, str]]): Default headers to be sent with every request.
            auth_token (Optional[str]): The authentication token for the API.
            max_retries (int): Maximum number of retries for failed requests.
            backoff_factor (float): Factor to apply between attempts.
                {backoff factor} * (2 ** ({number of total retries} - 1))
            status_forcelist (Optional[List[int]]): List of status codes that
                should trigger a retry.
            log_level (int): Logging level for the client (default: logging.INFO)
            custom_handler (Optional[logging.Handler]): Custom logging handler to add
                (e.g., NewRelic handler)
        """
        self.logger = custom_logger.get_logger(log_level=log_level, caller=self)
        self.logger.info("Initializing HttpClient with base_url: %s", base_url)

        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.default_headers = headers or {}

        # Configure retry strategy
        if status_forcelist is None:
            status_forcelist = [
                408,  # Request Timeout
                429,  # Too Many Requests
                500,  # Internal Server Error
                502,  # Bad Gateway
                503,  # Service Unavailable
                504,  # Gateway Timeout
            ]

        self.retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            allowed_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        )

        # Create session with retry strategy
        self.session = requests.Session()
        adapter = HTTPAdapter(max_retries=self.retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.logger.debug(
            "Retry configuration: max_retries=%d, backoff_factor=%f, "
            "status_forcelist=%s",
            max_retries,
            backoff_factor,
            status_forcelist,
        )

    def _get_headers(self, headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Generates the headers for the requests.

        Args:
            headers (Optional[Dict[str, str]]): Additional headers to include.

        Returns:
            Dict[str, str]: The headers dictionary.
        """
        headers = {
            "Content-Type": "application/json",  # Default if not provided
            "Accept": "application/json",  # Default value if not provided
            **self.default_headers,
            **(headers or {}),
        }
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        self.logger.debug("Generated headers: %s", headers)
        return headers

    def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> requests.Response:
        """
        Makes an HTTP request to the specified endpoint, with automatic retry functionality.

        Retries are handled automatically for transient errors (such as timeouts, server errors, etc.)
        according to the retry strategy configured in the session (see __init__).

        Args:
            method (str): The HTTP method to use (GET, POST, PUT, PATCH, DELETE).
            endpoint (str): The endpoint to send the request to.
            params (Optional[Dict[str, Any]]): Query parameters for the request.
            data (Optional[Dict[str, Any]]): The JSON payload for the request.
            headers (Optional[Dict[str, str]]): Additional headers to include.

        Returns:
            requests.Response: The response from the server.
        Raises:
            requests.exceptions.RequestException: For network-related errors or exceeded retries.
            requests.exceptions.HTTPError: For HTTP error responses (4xx, 5xx).
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        self.logger.info("Making %s request to: %s", method, url)
        if params:
            self.logger.debug("Request parameters: %s", params)
        if data:
            self.logger.debug("Request payload: %s", data)

        try:
            response = self.session.request(
                method=method,
                url=url,
                headers=self._get_headers(headers),
                params=params,
                data=data,
            )
            self.logger.info(
                "Request to %s/%s returned response status code: %d",
                self.base_url,
                endpoint,
                response.status_code,
            )
            self.logger.debug("Response headers: %s", dict(response.headers))

            if response.status_code >= 400:
                if response.status_code in self.retry_strategy.status_forcelist:
                    self.logger.warning(
                        "Request to %s/%s failed with retryable status code %d: %s",
                        self.base_url,
                        endpoint,
                        response.status_code,
                        response.text,
                    )
                    # Let the retry mechanism handle it
                    return response
                else:
                    self.logger.error(
                        "Request to %s/%s failed with non-retryable status code %d: %s",
                        self.base_url,
                        endpoint,
                        response.status_code,
                        response.text,
                    )
                    response.raise_for_status()

            return response

        except requests.exceptions.RequestException as e:
            self.logger.error(
                "Request to %s/%s failed: %s", self.base_url, endpoint, str(e)
            )
            raise

    def get(
        self,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> requests.Response:
        """
        Makes a GET request to the specified endpoint.

        Args:
            endpoint (str): The endpoint to send the GET request to.
            params (Optional[Dict[str, Any]]): Query parameters for the request.

        Returns:
            requests.Response: The response from the server.
        """
        return self.request("GET", endpoint, headers=headers, params=params)

    def post(
        self,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> requests.Response:
        """
        Makes a POST request to the specified endpoint.

        Args:
            endpoint (str): The endpoint to send the POST request to.
            data (Optional[Dict[str, Any]]): The JSON payload for the POST request.

        Returns:
            requests.Response: The response from the server.
        """
        return self.request("POST", endpoint, headers=headers, data=data)

    def put(
        self,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> requests.Response:
        """
        Makes a PUT request to the specified endpoint.

        Args:
            endpoint (str): The endpoint to send the PUT request to.
            data (Optional[Dict[str, Any]]): The JSON payload for the PUT request.

        Returns:
            requests.Response: The response from the server.
        """
        return self.request("PUT", endpoint, headers=headers, data=data)

    def patch(
        self,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> requests.Response:
        """
        Makes a PATCH request to the specified endpoint.

        Args:
            endpoint (str): The endpoint to send the PATCH request to.
            data (Optional[Dict[str, Any]]): The JSON payload for the PATCH request.

        Returns:
            requests.Response: The response from the server.
        """
        return self.request("PATCH", endpoint, headers=headers, data=data)

    def delete(
        self,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> requests.Response:
        """
        Makes a DELETE request to the specified endpoint.

        Args:
            endpoint (str): The endpoint to send the DELETE request to.
            data (Optional[Dict[str, Any]]): The JSON payload for the DELETE request.

        Returns:
            requests.Response: The response from the server.
        """
        return self.request("DELETE", endpoint, headers=headers, data=data)

    def _request_multiple(
        self,
        method: str,
        endpoints: List[str],
        headers_list: Optional[List[Dict[str, str]]] = None,
        params_list: Optional[List[Dict[str, Any]]] = None,
        data_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[requests.Response]:
        """
        Makes multiple HTTP requests in parallel.

        Args:
            method (str): The HTTP method to use.
            endpoints (List[str]): A list of endpoints to send the requests to.
            headers_list (Optional[List[Dict[str, str]]]): A list of headers
                dictionaries for each request.
            params_list (Optional[List[Dict[str, Any]]]): A list of query
                parameter dictionaries for each request.
            data_list (Optional[List[Dict[str, Any]]]): A list of JSON payload
                dictionaries for each request.

        Returns:
            List[requests.Response]: A list of responses from the server.
        """
        self.logger.info("Making %d parallel %s requests", len(endpoints), method)
        if headers_list is None:
            headers_list = [None] * len(endpoints)
        if params_list is None:
            params_list = [None] * len(endpoints)
        if data_list is None:
            data_list = [None] * len(endpoints)

        responses = []
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.request, method, endpoint, data=data): endpoint
                for endpoint, data in zip(endpoints, data_list)
            }
            for future in as_completed(futures):
                endpoint = futures[future]
                try:
                    response = future.result()
                    responses.append(response)
                    self.logger.debug(
                        "Successfully completed %s request to: %s", method, endpoint
                    )
                except Exception as e:
                    self.logger.error(
                        "Failed to complete %s request to %s: %s",
                        method,
                        endpoint,
                        str(e),
                    )
                    raise
        return responses

    def get_multiple(
        self,
        endpoints: List[str],
        headers_list: Optional[List[Dict[str, str]]] = None,
        params_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[requests.Response]:
        """
        Makes multiple GET requests in parallel.

        Args:
            endpoints (List[str]): A list of endpoints to send the GET requests to.
            headers_list (Optional[List[Dict[str, str]]]): A list of headers dictionaries for each request.
            params_list (Optional[List[Dict[str, Any]]]): A list of query parameter dictionaries for each request.

        Returns:
            List[requests.Response]: A list of responses from the server.
        """
        return self._request_multiple("GET", endpoints, headers_list, params_list)

    def post_multiple(
        self,
        endpoints: List[str],
        headers_list: Optional[List[Dict[str, str]]] = None,
        data_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[requests.Response]:
        """
        Makes multiple POST requests in parallel.

        Args:
            endpoints (List[str]): A list of endpoints to send the POST requests to.
            headers_list (Optional[List[Dict[str, str]]]): A list of headers dictionaries for each request.
            data_list (Optional[List[Dict[str, Any]]]): A list of JSON payload dictionaries for each request.

        Returns:
            List[requests.Response]: A list of responses from the server.
        """
        return self._request_multiple("POST", endpoints, headers_list, data_list)

    def put_multiple(
        self,
        endpoints: List[str],
        headers_list: Optional[List[Dict[str, str]]] = None,
        data_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[requests.Response]:
        """
        Makes multiple PUT requests in parallel.

        Args:
            endpoints (List[str]): A list of endpoints to send the PUT requests to.
            headers_list (Optional[List[Dict[str, str]]]): A list of headers dictionaries for each request.
            data_list (Optional[List[Dict[str, Any]]]): A list of JSON payload dictionaries for each request.

        Returns:
            List[requests.Response]: A list of responses from the server.
        """
        return self._request_multiple("PUT", endpoints, headers_list, data_list)

    def patch_multiple(
        self,
        endpoints: List[str],
        headers_list: Optional[List[Dict[str, str]]] = None,
        data_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[requests.Response]:
        """
        Makes multiple PATCH requests in parallel.

        Args:
            endpoints (List[str]): A list of endpoints to send the PATCH requests to.
            headers_list (Optional[List[Dict[str, str]]]): A list of headers dictionaries for each request.
            data_list (Optional[List[Dict[str, Any]]]): A list of JSON payload dictionaries for each request.

        Returns:
            List[requests.Response]: A list of responses from the server.
        """
        return self._request_multiple("PATCH", endpoints, headers_list, data_list)

    def delete_multiple(
        self,
        endpoints: List[str],
        headers_list: Optional[List[Dict[str, str]]] = None,
        data_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[requests.Response]:
        """
        Makes multiple DELETE requests in parallel.

        Args:
            endpoints (List[str]): A list of endpoints to send the DELETE requests to.
            headers_list (Optional[List[Dict[str, str]]]): A list of headers dictionaries for each request.
            data_list (Optional[List[Dict[str, Any]]]): A list of JSON payload dictionaries for each request.

        Returns:
            List[requests.Response]: A list of responses from the server.
        """
        return self._request_multiple("DELETE", endpoints, headers_list, data_list)

    def close(self):
        self.session.close()

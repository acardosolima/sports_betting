from unittest.mock import MagicMock, patch

import pytest
import requests

from src.ssa.utils.http_connector import HTTPConnector


class TestHTTPConnector:
    """Test suite for HTTPConnector class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.base_url = "http://api.example.com"
        self.connector = HTTPConnector(
            base_url=self.base_url,
            headers={"X-Custom-Header": "test"},
            auth_token="test-token",
            max_retries=3,
            backoff_factor=0.1,
            status_forcelist=[429, 503],
        )

    def _create_mock_response(self, status_code=200, text="Success", headers=None, raise_for_status=None):
        """Helper method to create a mock response."""
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.text = text
        mock_response.headers = headers or {"Content-Type": "application/json"}
        if raise_for_status:
            mock_response.raise_for_status.side_effect = raise_for_status
        return mock_response

    def _setup_mock_session(self, mock_session, response):
        """Helper method to set up a mock session with a response."""
        mock_session_instance = MagicMock()
        mock_session_instance.request.return_value = response
        mock_session.return_value = mock_session_instance
        self.connector.session = mock_session_instance
        return mock_session_instance

    def test_init(self):
        """Test HTTPConnector initialization with default values."""
        connector = HTTPConnector(base_url=self.base_url)

        assert connector.base_url == self.base_url
        assert connector.auth_token is None
        assert connector.default_headers == {}
        assert connector.retry_strategy.total == 3
        assert connector.retry_strategy.backoff_factor == 0.3
        assert connector.retry_strategy.status_forcelist == [408, 429, 500, 502, 503, 504]
        assert connector.retry_strategy.allowed_methods == ["GET", "POST", "PUT", "PATCH", "DELETE"]

    def test_init_with_custom_values(self):
        """Test HTTPConnector initialization with custom values."""
        connector = HTTPConnector(
            base_url=self.base_url,
            headers={"Custom-Header": "value"},
            auth_token="test-token",
            max_retries=5,
            backoff_factor=0.5,
            status_forcelist=[500, 503],
        )

        assert connector.base_url == self.base_url
        assert connector.auth_token == "test-token"
        assert connector.default_headers == {"Custom-Header": "value"}
        assert connector.retry_strategy.total == 5
        assert connector.retry_strategy.backoff_factor == 0.5
        assert connector.retry_strategy.status_forcelist == [500, 503]
        assert connector.retry_strategy.allowed_methods == ["GET", "POST", "PUT", "PATCH", "DELETE"]

    def test_get_headers(self):
        """Test header generation with different configurations."""
        test_cases = [
            # (connector_config, additional_headers, expected_headers)
            (
                {"base_url": self.base_url},
                None,
                {"Content-Type": "application/json", "Accept": "application/json"},
            ),
            (
                {"base_url": self.base_url, "auth_token": "test-token"},
                None,
                {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": "Bearer test-token",
                },
            ),
            (
                {"base_url": self.base_url, "headers": {"X-Custom-Header": "default-value"}},
                None,
                {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "X-Custom-Header": "default-value",
                },
            ),
            (
                {"base_url": self.base_url},
                {"X-Additional-Header": "additional-value"},
                {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "X-Additional-Header": "additional-value",
                },
            ),
            (
                {"base_url": self.base_url, "headers": {"X-Custom-Header": "default-value"}},
                {"X-Custom-Header": "override-value"},
                {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "X-Custom-Header": "override-value",
                },
            ),
        ]

        for config, additional_headers, expected_headers in test_cases:
            connector = HTTPConnector(**config)
            headers = connector._get_headers(additional_headers)
            assert headers == expected_headers

    @patch("requests.Session")
    def test_request_non_retryable_error(self, mock_session):
        """Test that non-retryable errors are not retried."""
        mock_response = self._create_mock_response(
            status_code=400,
            text="Bad Request",
            raise_for_status=requests.exceptions.HTTPError("400 Bad Request"),
        )
        mock_session_instance = self._setup_mock_session(mock_session, mock_response)

        with pytest.raises(requests.exceptions.HTTPError):
            self.connector.request("GET", "test")

        mock_session_instance.request.assert_called_once()

    @patch("requests.Session")
    def test_request_success(self, mock_session):
        """Test successful request."""
        mock_response = self._create_mock_response()
        mock_session_instance = self._setup_mock_session(mock_session, mock_response)

        response = self.connector.request("GET", "test")

        assert response.status_code == 200
        assert response.text == "Success"
        mock_session_instance.request.assert_called_once()

    @patch("requests.Session.request")
    def test_get_request(self, mock_request):
        """Test GET request"""
        mock_response = self._create_mock_response()
        mock_request.return_value = mock_response

        response = self.connector.get("test-endpoint", params={"key": "value"})
        assert response.status_code == 200
        mock_request.assert_called_once_with(
            method="GET",
            url=f"{self.connector.base_url}/test-endpoint",
            headers=self.connector._get_headers(),
            params={"key": "value"},
            data=None,
        )

    @patch("requests.Session.request")
    def test_post_request(self, mock_request):
        """Test POST request"""
        mock_response = self._create_mock_response(status_code=201)
        mock_request.return_value = mock_response

        data = {"key": "value"}
        response = self.connector.post("test-endpoint", data=data)
        assert response.status_code == 201
        mock_request.assert_called_once_with(
            method="POST",
            url=f"{self.connector.base_url}/test-endpoint",
            headers=self.connector._get_headers(),
            params=None,
            data=data,
        )

    @patch("requests.Session.request")
    def test_put_request(self, mock_request):
        """Test PUT request"""
        mock_response = self._create_mock_response(status_code=200)
        mock_request.return_value = mock_response

        data = {"key": "value"}
        response = self.connector.put("test-endpoint", data=data)
        assert response.status_code == 200
        mock_request.assert_called_once_with(
            method="PUT",
            url=f"{self.connector.base_url}/test-endpoint",
            headers=self.connector._get_headers(),
            params=None,
            data=data,
        )

    @patch("requests.Session.request")
    def test_patch_request(self, mock_request):
        """Test PATCH request"""
        mock_response = self._create_mock_response(status_code=200)
        mock_request.return_value = mock_response

        data = {"key": "value"}
        response = self.connector.patch("test-endpoint", data=data)
        assert response.status_code == 200
        mock_request.assert_called_once_with(
            method="PATCH",
            url=f"{self.connector.base_url}/test-endpoint",
            headers=self.connector._get_headers(),
            params=None,
            data=data,
        )

    @patch("requests.Session.request")
    def test_delete_request(self, mock_request):
        """Test DELETE request"""
        mock_response = self._create_mock_response(status_code=204)
        mock_request.return_value = mock_response

        response = self.connector.delete("test-endpoint")
        assert response.status_code == 204
        mock_request.assert_called_once_with(
            method="DELETE",
            url=f"{self.connector.base_url}/test-endpoint",
            headers=self.connector._get_headers(),
            params=None,
            data=None,
        )

    @patch("requests.Session.request")
    def test_parallel_requests(self, mock_request):
        """Test parallel requests"""
        mock_response = self._create_mock_response()
        mock_request.return_value = mock_response

        endpoints = ["endpoint1", "endpoint2", "endpoint3"]
        responses = self.connector.get_multiple(endpoints)

        assert len(responses) == 3
        assert all(r.status_code == 200 for r in responses)
        assert mock_request.call_count == 3

    @patch("requests.Session.request")
    def test_post_multiple(self, mock_request):
        """Test multiple POST requests"""
        mock_response = self._create_mock_response(status_code=201)
        mock_request.return_value = mock_response

        endpoints = ["endpoint1", "endpoint2"]
        data_list = [{"key1": "value1"}, {"key2": "value2"}]
        responses = self.connector.post_multiple(endpoints, data_list=data_list)

        assert len(responses) == 2
        assert all(r.status_code == 201 for r in responses)
        assert mock_request.call_count == 2

    @patch("requests.Session.request")
    def test_put_multiple(self, mock_request):
        """Test multiple PUT requests"""
        mock_response = self._create_mock_response()
        mock_request.return_value = mock_response

        endpoints = ["endpoint1", "endpoint2"]
        data_list = [{"key1": "value1"}, {"key2": "value2"}]
        responses = self.connector.put_multiple(endpoints, data_list=data_list)

        assert len(responses) == 2
        assert all(r.status_code == 200 for r in responses)
        assert mock_request.call_count == 2

    @patch("requests.Session.request")
    def test_patch_multiple(self, mock_request):
        """Test multiple PATCH requests"""
        mock_response = self._create_mock_response()
        mock_request.return_value = mock_response

        endpoints = ["endpoint1", "endpoint2"]
        data_list = [{"key1": "value1"}, {"key2": "value2"}]
        responses = self.connector.patch_multiple(endpoints, data_list=data_list)

        assert len(responses) == 2
        assert all(r.status_code == 200 for r in responses)
        assert mock_request.call_count == 2

    @patch("requests.Session.request")
    def test_delete_multiple(self, mock_request):
        """Test multiple DELETE requests"""
        mock_response = self._create_mock_response()
        mock_request.return_value = mock_response

        endpoints = ["endpoint1", "endpoint2"]
        responses = self.connector.delete_multiple(endpoints)

        assert len(responses) == 2
        assert all(r.status_code == 200 for r in responses)
        assert mock_request.call_count == 2

    @patch("requests.Session.request")
    def test_request_multiple_with_headers_and_params(self, mock_request):
        """Test _request_multiple with headers and params"""
        mock_response = self._create_mock_response()
        mock_request.return_value = mock_response

        endpoints = ["endpoint1", "endpoint2"]
        headers_list = [{"Header1": "value1"}, {"Header2": "value2"}]
        params_list = [{"param1": "value1"}, {"param2": "value2"}]

        responses = self.connector._request_multiple(
            "GET", endpoints, headers_list=headers_list, params_list=params_list
        )

        assert len(responses) == 2
        assert all(r.status_code == 200 for r in responses)
        assert mock_request.call_count == 2

    def test_close(self):
        """Test session close"""
        self.connector.close()
        assert self.connector.session.close

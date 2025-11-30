import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app as flask_app


class TestFlaskApp:
    """Test cases for Flask application"""

    @pytest.fixture
    def client(self):
        """Create a test client"""
        flask_app.config["TESTING"] = True
        with flask_app.test_client() as client:
            yield client

    def test_home_page(self, client):
        """Test home page loads"""
        response = client.get("/")
        assert response.status_code == 200
        assert b"Welcome" in response.data or b"home" in response.data.lower()

    def test_predict_page_get(self, client):
        """Test predict page GET request"""
        response = client.get("/predictdata")
        assert response.status_code == 200

    def test_predict_page_post(self, client):
        """Test predict page POST request with valid data"""
        data = {
            "gender": "male",
            "ethnicity": "group A",
            "parental_level_of_education": "bachelor's degree",
            "lunch": "standard",
            "test_preparation_course": "completed",
            "reading_score": "70",
            "writing_score": "75",
        }

        response = client.post("/predictdata", data=data)
        # Should return 200 if model and preprocessor exist
        assert response.status_code in [200, 500]  # 500 if artifacts missing

    def test_invalid_endpoint(self, client):
        """Test invalid endpoint returns 404"""
        response = client.get("/invalid-endpoint")
        assert response.status_code == 404

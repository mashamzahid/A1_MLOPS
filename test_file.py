import unittest
from flask import Flask
from flask.testing import FlaskClient

# Import your Flask app and ensure it's in the same directory
from app import app

class FlaskAppTestCase(unittest.TestCase):

    def setUp(self):
        # Create a test client for the app
        self.app = app.test_client()
        self.app.testing = True

    def test_index_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_predict_route(self):
        data = {
            'sepal_length': 5.1,
            'sepal_width': 3.5,
            'petal_length': 1.4,
            'petal_width': 0.2
        }

        response = self.app.post('/predict', data=data, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Predicted Flower Species: ', response.data)

if __name__ == '__main__':
    unittest.main()

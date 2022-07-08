from django.shortcuts import reverse
from django.http import response
from django.test import SimpleTestCase, TestCase

class TestUrls(SimpleTestCase):
    def test_homepage(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

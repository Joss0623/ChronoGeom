import unittest
from unittest.mock import patch
from scripts.chronogeom.fetch_official_member import read_range


class FakeResponse:
    def __init__(self,status,headers,data=b''):
        self.status=status;self.headers=headers;self.data=data;self.read_called=False
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def read(self,n):self.read_called=True;return self.data[:n]


class TestBoundedAssetAcquisition(unittest.TestCase):
    def test_full_archive_response_refused_without_read(self):
        response=FakeResponse(200,{'Content-Length':'13176647884'})
        with patch('urllib.request.urlopen',return_value=response):
            with self.assertRaises(RuntimeError):read_range('https://example.org/cache.zip',50,10)
        self.assertFalse(response.read_called)

    def test_incorrect_cached_range_refused(self):
        response=FakeResponse(206,{'Content-Range':'bytes 0-9/100'},b'0123456789')
        with patch('urllib.request.urlopen',return_value=response):
            with self.assertRaises(RuntimeError):read_range('https://example.org/cache.zip',50,10)
        self.assertFalse(response.read_called)

    def test_expected_range_read_exactly(self):
        response=FakeResponse(206,{'Content-Range':'bytes 50-59/100'},b'0123456789')
        with patch('urllib.request.urlopen',return_value=response):
            self.assertEqual(read_range('https://example.org/cache.zip',50,10),b'0123456789')


if __name__=='__main__':unittest.main()

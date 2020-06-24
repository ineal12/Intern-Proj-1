# first-party
from geofred_interface import pull_data_from_geodb


def test_json_decode_error():
    series_ids = [
        "A794RX0Q048SBEA"
    ]
    
    df_a, df_m = pull_data_from_geodb(series_ids)

    return None


if __name__ == "__main__":
    test_json_decode_error()

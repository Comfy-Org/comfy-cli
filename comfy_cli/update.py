import requests


def check_for_update(package_name, current_version):
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises stored HTTPError, if one occurred
        latest_version = response.json()["info"]["version"]
        return latest_version != current_version, latest_version
    except requests.RequestException as e:
        print(f"Error checking latest version: {e}")
        return False, current_version

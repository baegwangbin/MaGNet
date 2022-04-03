# Download network weights
# Source: https://stackoverflow.com/a/39225039
import requests


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id,
                  'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


if __name__ == "__main__":

    # AdaBins encoder ... used when training on KITTI eigen split
    download_file_from_google_drive('1wNMVvZmaLVUflIM_yFLj9vQBD7jBmT0N', './ckpts/AdaBins_kitti_encoder.pt')

    # DNET weights
    download_file_from_google_drive('1eRQtf9MJNPXmn1UDr2RjEqbQfY4NQ7jT', './ckpts/DNET_kitti_eigen.pt')
    download_file_from_google_drive('1z_3zz-hPxSfiUKsN1TIBeZv6YRvZGtfP', './ckpts/DNET_kitti_official.pt')
    download_file_from_google_drive('1bbzfboj6XkfFhoJ54Iiqc5Ylj95A015M', './ckpts/DNET_scannet.pt')

    # FNET weights
    download_file_from_google_drive('1_mcielHqddp9p9ua7by77JG55h_5S9tT', './ckpts/FNET_kitti_eigen.pt')
    download_file_from_google_drive('1raQGaE5HrciulIZmNn5TNGp87AgyYp4Y', './ckpts/FNET_kitti_official.pt')
    download_file_from_google_drive('1ugDr67UOanpQZMlPopiM8OihUexhPql4', './ckpts/FNET_scannet.pt')

    # MAGNET weights
    download_file_from_google_drive('1MmqunqAr1mGqYUGBNUUmaJHAO7fYgiYn', './ckpts/MAGNET_kitti_eigen.pt')
    download_file_from_google_drive('1mKspc_p3yXp-zd1sZDeau9qrl82pJyGG', './ckpts/MAGNET_kitti_official.pt')
    download_file_from_google_drive('1Zuy_8P97OT9Of5PtyNc22DzhXQlD2OE-', './ckpts/MAGNET_scannet.pt')


from bs4 import BeautifulSoup
import requests
def sent_message(message):

    headers = {
        "Authorization": "Bearer " + "zRWlFhIBpu22NDLAkVy6jHb1e02CtKJDr7OkYvyTyAu",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    params = {"message": message}

    r = requests.post("https://notify-api.line.me/api/notify",
                      headers=headers, params=params)
    print(r.status_code)  # 200
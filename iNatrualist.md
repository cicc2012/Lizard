Downloading a **bulk of images** from multiple observations isn’t directly supported through the web interface. One indirect solution:

---

## Overview: Use the iNaturalist API + Scripting**

iNaturalist offers a public **API** that you can use to:

1. Search for observations (filtered by taxon, place, user, etc.).
2. Extract image URLs from the observation data.
3. Download the images with a script (Python, R, etc.).

### Basic Steps with a Python example:

1. **Install dependencies:**

   ```bash
   python -m pip install requests
   ```

2. **Python script example:**

   ```python
   import requests
   import os

   # Customize your search query here
   params = {
       'taxon_name': 'Danaus plexippus',  # Monarch butterfly
       'per_page': 50,
       'page': 1
   }

   os.makedirs("images", exist_ok=True)

   while True:
       response = requests.get('https://api.inaturalist.org/v1/observations', params=params).json()
       results = response['results']
       if not results:   # if response.status_code != 200:
           break

       for obs in results:
           for photo in obs.get('photos', []):
               url = photo['url'].replace('square', 'original')  # high-res image
               img_data = requests.get(url).content
               img_filename = os.path.join("images", f"{photo['id']}.jpg")
               with open(img_filename, 'wb') as f:
                   f.write(img_data)
               print(f"Downloaded {img_filename}")

       params['page'] += 1
   ```

> A more complete example can be found at [fetch.py](fetch.py).
> You can filter by user, location, date, or taxon using parameters listed in [iNaturalist API docs](https://api.inaturalist.org/v1/docs).
> Here we used Get observations. The response and parameters can be found at the [iNaturalist API docs for "get observations"](https://api.inaturalist.org/v1/docs/#!/Observations/get_observations).

**!Attention!**: "Please note that we throttle API usage to a max of 100 requests per minute, though we ask that you try to keep it to 60 requests per minute or lower, and to keep under 10,000 requests per day. If we notice usage that has serious impact on our performance we may institute blocks without notification."



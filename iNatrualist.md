Downloading a **bulk of images** from multiple observations isn’t directly supported through the web interface. Here are two indirect methods to do it:

---

## 1: Use the iNaturalist API + Scripting**

iNaturalist offers a public **API** that you can use to:

1. Search for observations (filtered by taxon, place, user, etc.).
2. Extract image URLs from the observation data.
3. Download the images with a script (Python, R, etc.).

### 🔧 Step-by-step (Python example):

1. **Install dependencies:**

   ```bash
   pip install requests
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
       if not results:
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

> You can filter by user, location, date, or taxon using parameters listed in [iNaturalist API docs](https://api.inaturalist.org/v1/docs).

---

## 2: Download from Exported Observation Data**

1. Go to [iNaturalist Observation Export](https://www.inaturalist.org/observations/export)
2. Use filters (taxa, place, dates, etc.)
3. Export the CSV (it includes image URLs)
4. Use a script (or a bulk downloader extension) to fetch the images from those URLs.

Note: The URLs in the CSV point to the **square thumbnail**; modify them to get the original:

* Example URL:

  ```
  https://static.inaturalist.org/photos/12345/square.jpg
  ```

  → Change `square` to `original`:

  ```
  https://static.inaturalist.org/photos/12345/original.jpg
  ```

---

## Terms of Use Reminder

Always respect:

* **iNaturalist's Terms of Service**
* **Image licensing** (many photos are under Creative Commons, but not all)

You can check licensing info in the observation metadata or use API fields like `license_code`.



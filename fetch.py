import requests
import os
import time
from openpyxl import Workbook, load_workbook

# Customize your search query here
# Follow the iNaturalist API documentation for more options: https://api.inaturalist.org/v1/docs/
# To get observations, you can follow the API documentation: https://api.inaturalist.org/v1/docs/#observations
# https://api.inaturalist.org/v1/docs/#!/Observations/get_observations
# TODO: Change this to your desired taxon name
# For example, to search for "Prairie Lizard", you can use the following parameters
params = {
    'taxon_name': 'Prairie Lizard',
    'per_page': 10, # Number of results to return in a page. Max is 200.
    'page': 1
}

# Specify the directory to save images
# TODO: Change this to your desired directory
save_directory = "E:/Lizard/images"
os.makedirs(save_directory, exist_ok=True)

# File to log image filenames, URLs, and times taken
# log_file = os.path.join(save_directory, "image_log.txt")
log_file = os.path.join(save_directory, "image_log.xlsx")
# Initialize Excel workbook and sheet
# Check if the log file exists
if os.path.exists(log_file):
    wb = load_workbook(log_file)
    ws = wb.active
else:
    wb = Workbook()
    ws = wb.active
    ws.title = "Image Log"
    ws.append(["Image Filename", "Image URL", "Observation ID", "Time Used (ms)"])  # Add headers
    # Adjust column widths
    ws.column_dimensions['A'].width = 40  # Image Filename
    ws.column_dimensions['B'].width = 50  # Image URL
    ws.column_dimensions['C'].width = 12  # Observation ID
    ws.column_dimensions['D'].width = 10  # Time Used (ms)

# Initialize counters
image_count = 0
start_time_global = time.time()

while params['page'] <= 4 and image_count <= 60:  # Limit to 4 pages for demonstration
    # try to keep it to 60 requests per minute or lower, and to keep under 10,000 requests per day

    response = requests.get('https://api.inaturalist.org/v1/observations', params=params)
    results = response.json()['results']
    if not results or response.status_code != 200:   # if response.status_code != 200:
        break
    for obs in results:
        observation_id = obs.get('id')  # Get the observation ID
        for photo in obs.get('photos', []):
            url = photo['url'].replace('square', 'original')  # high-res image

            # Start timing the download
            start_time = time.time()

            img_data = requests.get(url).content
            img_filename = os.path.join(save_directory, f"{observation_id}_{photo['id']}.jpg")
            if os.path.exists(img_filename):
                print(f"File {img_filename} already exists. Skipping download.")
                time.sleep(1)  # Wait for 1 second before checking the next image
                continue  # Skip downloading if the file already exists
            with open(img_filename, 'wb') as f:
                f.write(img_data)

            # End timing the download
            end_time = time.time()
            time_used_ms = (end_time - start_time) * 1000  # Convert to milliseconds

            # Append the filename and URL to the log file
            # with open(log_file, 'a') as log:
            #     log.write(f"{img_filename}, {url}\n")
            ws.append([img_filename, url, observation_id, f"{time_used_ms:.2f}"])

            # Print the download time to study the performance
            print(f"Downloaded {img_filename}   in {time_used_ms:.2f} ms")

            # Increment the image count
            image_count += 1

            # Wait for 1 second before downloading the next image
            # Ensure you don't hit the API too hard: less than 60 requests per minute
            sleep_time = max(0, 1 - time_used_ms / 1000)  # Ensure non-negative sleep time
            time.sleep(sleep_time)

    params['page'] += 1

# Save the Excel file
# By pressing Ctrl+C, you can stop the program, but the update won't be saved in the log file
wb.save(log_file)

# Print summary
end_time_global = time.time()
total_time_global = (end_time_global - start_time_global) 
print(f"Downloaded {image_count} images within {total_time_global} seconds. Log saved to {log_file}.")
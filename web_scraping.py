from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import csv

# 1. Setup headless Chrome
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
driver = webdriver.Chrome(options=chrome_options)

all_products = []

# 2. Loop through multiple Flipkart search result pages
for page in range(1, 30): 
    url = f"https://www.flipkart.com/search?q=laptop&page={page}"
    driver.get(url)
    time.sleep(3)  # Allow page to load

    soup = BeautifulSoup(driver.page_source, "html.parser")
    products = soup.find_all("div", {"data-id": True})

    for item in products:
        # Basic Details
        name_tag = item.find("div", class_="KzDlHZ")
        specs_tag = item.find("div", class_="_6NESgJ")
        rating_tag = item.find("div", class_="XQDdHH") or item.find("span", class_="Y1HW00")
        price_tag = item.find("div", class_="hl05eU") or item.find("span", class_="cN1yY0")
        link_tag = item.find("a", href=True)

        name = name_tag.get_text().strip() if name_tag else "Not found"
        specs = specs_tag.get_text().strip() if specs_tag else "Not found"
        rating = rating_tag.get_text().strip() if rating_tag else "Not found"
        price = price_tag.get_text().strip() if price_tag else "Not found"
        product_url = "https://www.flipkart.com" + link_tag["href"] if link_tag else "Not found"

        # Extra Details from Product Page
        battery = weight = webcam = display = "Not found"

        if product_url != "Not found":
            try:
                driver.get(product_url)
                time.sleep(2)
                detail_soup = BeautifulSoup(driver.page_source, "html.parser")
                table_cells = detail_soup.find_all("td")

                for i in range(len(table_cells)):
                    label = table_cells[i].get_text().lower()
                    value = table_cells[i + 1].get_text() if i + 1 < len(table_cells) else ""

                    if "battery" in label or "battery backup" in label:
                        battery = value
                    elif "weight" in label:
                        weight = value
                    elif "web camera" in label or "webcam" in label:
                        webcam = value
                    elif "display size" in label or "screen size" in label:
                        display = value

            except Exception as e:
                print(f"⚠️ Failed to extract details from {product_url}")

        # Save product data
        all_products.append([
            name, specs, rating, price, product_url,
            battery, weight, webcam, display
        ])

    print(f"✅ Page {page} scraped — Total products so far: {len(all_products)}")

# 3. Close browser
driver.quit()

# 4. Save data to CSV
with open("flipkart_laptop_final.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Product Name", "Specifications", "Rating", "Price", "Product URL",
        "Battery Life", "Weight", "Webcam", "Display Size"
    ])
    writer.writerows(all_products)

print("✅ Saved data to 'flipkart_laptop_final.csv'")

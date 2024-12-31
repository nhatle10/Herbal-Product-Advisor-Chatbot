import argparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import json
import time

def scrape_product_links(driver, base_url, productpage_url, num_pages, output_file):
    """Scrapes product links from multiple pages and saves them to a file."""
    product_links = []
    with open(output_file, 'w', encoding='utf-8') as file:
        for page in range(1, num_pages + 1):
            url = f'{productpage_url}{page}'
            print(f'Đang xử lý trang sản phẩm: {url}')
            driver.get(url)
            time.sleep(2)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            links = soup.find_all('a', class_='btn-readmore')

            for link in links:
                product_url = link['href']
                full_product_url = base_url + product_url
                product_links.append(full_product_url)
                file.write(full_product_url + '\n')
    return product_links

def extract_product_data(driver, base_url, product_links, fields_to_extract, output_json):
    """Extracts product details from individual product pages and saves them to a JSON file."""
    data = []
    for i, link in enumerate(product_links):
        print(f'Đang xử lý sản phẩm: {link}')
        driver.get(link)
        time.sleep(2)

        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')

        product_data = {"url": link}

        for field, spec in fields_to_extract.items():
            element = None
            if spec["type"] == "class":
                element = soup.find(class_=spec["value"])
                if "attribute" in spec and element:
                    element = element.find("a")
                    if element:
                        element = element.get(spec["attribute"])
                    else:
                        element = None
            elif spec["type"] == "id":
                element = soup.find(id=spec["value"])

            if field == "description" and element:
                product_data[field] = element.get_text(strip=True)
            elif field == "image_url" and element:
                product_data[field] = base_url + element
            else:
                product_data[field] = element.text.strip() if element else None

        content = soup.find('div', id='pills-tabContent')
        product_data["details"] = content.get_text("\n", strip=True) if content else None

        product_cart_options = soup.find("div", class_="product-cart-options")
        options = []
        if product_cart_options:
            for option_div in product_cart_options.find_all("div", class_="option-item"):
                option_text = option_div.get_text(strip=True)
                if option_text:
                    options.append(option_text)
        product_data["options"] = options

        tags_div = soup.find("div", class_="d-flex gap-1 flex-wrap")
        tags = []
        if tags_div:
            for tag_a in tags_div.find_all("a"):
                tag_text = tag_a.get_text(strip=True)
                tag_link = tag_a.get("href")
                if tag_text and tag_link:
                    tags.append(tag_text)
        product_data["tags"] = tags

        data.append(product_data)
        print(f"----------Đã xử lý {i + 1}/{len(product_links)}")

    with open(output_json, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
    print(f'Dữ liệu sản phẩm đã được lưu vào: {output_json}')

def main():
    """Main function to execute the web scraping."""
    parser = argparse.ArgumentParser(description='Thu thập dữ liệu sản phẩm từ tratoanthang.com')
    parser.add_argument('--output_links', default='product_links.txt', help='Tên file để lưu đường dẫn sản phẩm')
    parser.add_argument('--output_json', default='product_data.json', help='Tên file JSON để lưu dữ liệu sản phẩm')
    parser.add_argument('--headless', action='store_true', help='Chạy trình duyệt ở chế độ headless')

    args = parser.parse_args()

    base_url = 'https://tratoanthang.com/'
    productpage_url = 'https://tratoanthang.com/san-pham?page='
    num_pages = 7

    chrome_options = Options()
    if args.headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    fields_to_extract = {
        "title": {"type": "class", "value": "title-head"},
        "info": {"type": "class", "value": "info-products"},
        "price": {"type": "id", "value": "product-detail-price-sale"},
        "sale-price": {"type": "id", "value": "product-detail-price"},
        "description": {"type": "class", "value": "description"},
        "image_url": {"type": "class", "value": "box-image-featured", "attribute": "href"},
    }

    try:
        service = ChromeService(executable_path=ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)

        product_links = scrape_product_links(driver, base_url, productpage_url, num_pages, args.output_links)
        extract_product_data(driver, base_url, product_links, fields_to_extract, args.output_json)

    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
    finally:
        if 'driver' in locals():
            driver.quit()

if __name__ == "__main__":
    main()
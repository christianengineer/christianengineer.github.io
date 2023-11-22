import os
import yaml
from datetime import datetime
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom


def parse_front_matter(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        front_matter = "\n".join(lines[1 : lines.index("---\n", 1)])
        return yaml.safe_load(front_matter)


def create_sitemap_element(url, lastmod):
    url_element = Element("url")
    loc = SubElement(url_element, "loc")
    loc.text = url
    last_mod = SubElement(url_element, "lastmod")
    last_mod.text = (
        lastmod.strftime("%Y-%m-%d")
        if isinstance(lastmod, datetime)
        else lastmod.isoformat()
    )
    changefreq = SubElement(url_element, "changefreq")
    changefreq.text = "yearly"
    priority = SubElement(url_element, "priority")
    priority.text = "0.8"

    return url_element


def generate_sitemap(posts_folder, domain):
    urlset = Element("urlset", xmlns="http://www.sitemaps.org/schemas/sitemap/0.9")

    for filename in os.listdir(posts_folder):
        if filename.endswith(".md"):
            metadata = parse_front_matter(os.path.join(posts_folder, filename))
            if "permalink" in metadata and "date" in metadata:
                date = metadata["date"]
                if isinstance(date, str):
                    date = datetime.strptime(date, "%Y-%m-%d").date()

                url_element = create_sitemap_element(
                    domain + metadata["permalink"], date
                )
                urlset.append(url_element)

    xml_str = minidom.parseString(tostring(urlset)).toprettyxml(indent="   ")
    with open("sitemap.xml", "w", encoding="utf-8") as f:
        f.write(xml_str)


# Example usage
posts_folder = "_posts"
domain = "https://christianipanaque.com/"
generate_sitemap(posts_folder, domain)

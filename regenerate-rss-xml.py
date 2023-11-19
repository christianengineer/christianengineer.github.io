import os
import markdown
from datetime import datetime
import xml.etree.ElementTree as ET


def update_rss_xml(posts_dir, rss_file):
    tree = ET.parse(rss_file)
    root = tree.getroot()
    channel = root.find("channel")

    first_item_index = next(
        (i for i, element in enumerate(list(channel)) if element.tag == "item"),
        len(channel),
    )

    last_build_date = channel.find("lastBuildDate")
    current_utc_time = datetime.utcnow()

    formatted_date = current_utc_time.strftime("%a, %d %b %Y %H:%M:%S GMT")

    if last_build_date is not None:
        last_build_date.text = formatted_date
    else:
        new_last_build_date = ET.SubElement(channel, "lastBuildDate")
        new_last_build_date.text = formatted_date

    existing_links = {item.find("link").text for item in channel.findall("item")}
    print(existing_links)

    new_items = []

    for filename in os.listdir(posts_dir):
        if filename.endswith(".md"):
            with open(os.path.join(posts_dir, filename), "r") as file:
                content = file.read()
                md = markdown.Markdown(extensions=["meta"])
                md.convert(content)

                permalink = f"https://christianipanaque.com/{md.Meta['permalink'][0]}"

                if permalink not in existing_links:
                    item = ET.Element("item")

                    title = ET.SubElement(item, "title")
                    title.text = md.Meta["title"][0]

                    link = ET.SubElement(item, "link")
                    link.text = permalink

                    description = ET.SubElement(item, "description")
                    description.text = f"Read about {md.Meta['title'][0]}"

                    author = ET.SubElement(item, "author")
                    author.text = "christianipanaque@outlook.com (Christian Ipanaque)"

                    category = ET.SubElement(item, "category")
                    category.text = "Software Engineering"

                    guid = ET.SubElement(item, "guid", isPermaLink="true")
                    guid.text = permalink

                    # Format date for pubDate
                    post_date = datetime.strptime(md.Meta["date"][0], "%Y-%m-%d")
                    formatted_date = post_date.strftime("%a, %d %b %Y %H:%M:%S GMT")

                    pubDate = ET.SubElement(item, "pubDate")
                    pubDate.text = formatted_date

                    source = ET.SubElement(
                        item, "source", url="https://christianipanaque.com/rss.xml"
                    )
                    source.text = "Senior Full Stack Software Engineer"

                    new_items.append(item)

    for item in reversed(new_items):
        channel.insert(first_item_index, item)

    tree.write(rss_file)


update_rss_xml("_posts", "rss.xml")

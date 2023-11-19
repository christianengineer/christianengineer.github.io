import os
import markdown
from datetime import datetime
from xml.etree.ElementTree import Element, SubElement, ElementTree
import xml.etree.ElementTree as ET


def create_rss_xml(posts_dir):
    rss = Element(
        "rss",
        version="2.0",
        attrib={
            "xmlns:atom": "http://www.w3.org/2005/Atom",
            "xmlns:media": "http://search.yahoo.com/mrss/",
        },
    )
    channel = SubElement(rss, "channel")

    for filename in os.listdir(posts_dir):
        if filename.endswith(".md"):
            with open(os.path.join(posts_dir, filename), "r") as file:
                content = file.read()
                md = markdown.Markdown(extensions=["meta"])
                md.convert(content)

                item = SubElement(channel, "item")

                title = SubElement(item, "title")
                title.text = md.Meta["title"][0]

                link = SubElement(item, "link")
                link.text = md.Meta["permalink"][0]

                description = SubElement(item, "description")
                description.text = content

                category = SubElement(item, "category")
                category.text = "Software Engineering"

                author = SubElement(item, "author")
                author.text = "christianipanaque@outlook.com (Christian Ipanaque)"

                guid = SubElement(item, "guid")
                guid.text = md.Meta["permalink"][0]

                source = SubElement(item, "source")
                source.text = "https://christianipanaque.com/rss.xml"

                # Additional fields like author, category, guid, pubDate, source can be added here

    tree = ElementTree(rss)
    tree.write("rss.xml", encoding="UTF-8", xml_declaration=True)


create_rss_xml("_posts")

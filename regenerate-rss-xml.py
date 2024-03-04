import os
from datetime import datetime
import frontmatter
from lxml import etree
from datetime import datetime, date, time, timedelta
import random


def convert_to_rfc3339(date_input):
    # Handle None or empty input
    if not date_input:
        # Generate a random default datetime for today with a random time
        today_with_random_time = datetime.combine(
            date.today(),
            time(random.randint(0, 23), random.randint(0, 59), random.randint(0, 59)),
        )
        return today_with_random_time.isoformat() + "Z"

    # If input is already a datetime object, ensure it's in UTC ('Z') format
    if isinstance(date_input, datetime):
        return date_input.isoformat() + "Z"

    # If input is a date object, combine it with a random time
    if isinstance(date_input, date):
        random_time = datetime.combine(
            date_input,
            time(random.randint(0, 23), random.randint(0, 59), random.randint(0, 59)),
        )
        return random_time.isoformat() + "Z"

    # Assume input is a string and try parsing it
    try:
        date_obj = datetime.strptime(date_input, "%Y-%m-%d")
        # Combine parsed date with a random time
        date_with_random_time = datetime.combine(
            date_obj.date(),
            time(random.randint(0, 23), random.randint(0, 59), random.randint(0, 59)),
        )
        return date_with_random_time.isoformat() + "Z"
    except ValueError as e:
        # Log or handle the parsing error
        print(f"Error parsing date: {date_input}. Error: {e}")
        return ""  # Return an empty string or a default value


def generate_atom_feed(posts_folder):
    ns_map = {
        None: "http://www.w3.org/2005/Atom",  # Default namespace
        "media": "http://search.yahoo.com/mrss/",  # Additional namespace
    }
    feed = etree.Element("{http://www.w3.org/2005/Atom}feed", nsmap=ns_map)
    etree.SubElement(feed, "{http://www.w3.org/2005/Atom}title").text = (
        "Full Stack Software Engineer - Christian Ipanaque"
    )
    etree.SubElement(
        feed, "{http://www.w3.org/2005/Atom}subtitle", type="html"
    ).text = "Diving deep into the world of Full Stack Development with a focus on building scalable enterprise AI applications and platforms. Discover cutting-edge design patterns, best practices, and insights tailored for the next generation of AI innovations. Whether you're bootstrapping your AI startup or scaling an AI Corporation to new heights, this blog provides the technical knowledge and engineering-focused advice to build robust, scalable, and intelligent platforms."
    etree.SubElement(feed, "{http://www.w3.org/2005/Atom}updated").text = (
        convert_to_rfc3339(datetime.now())
    )
    etree.SubElement(feed, "{http://www.w3.org/2005/Atom}id").text = (
        "tag:christianipanaque.com,2024:3"
    )
    etree.SubElement(
        feed,
        "{http://www.w3.org/2005/Atom}link",
        rel="alternate",
        type="text/html",
        hreflang="en",
        href="https://christianipanaque.com",
    )
    etree.SubElement(
        feed,
        "{http://www.w3.org/2005/Atom}link",
        rel="self",
        type="application/atom+xml",
        href="https://christianipanaque.com/feed.atom",
    )
    etree.SubElement(feed, "{http://www.w3.org/2005/Atom}rights").text = (
        "Copyright (c) 2024, Christian Ipanaque"
    )
    etree.SubElement(
        feed,
        "{http://www.w3.org/2005/Atom}generator",
        uri="https://github.com/christianipanque",
        version="1.0",
    ).text = "Christian Ipanaque RSS / Atom Feed Generator"

    for filename in os.listdir(posts_folder):
        if filename.endswith(".md"):
            path = os.path.join(posts_folder, filename)
            with open(path, "r", encoding="utf-8") as file:
                post = frontmatter.load(file)
                entry = etree.SubElement(feed, "{http://www.w3.org/2005/Atom}entry")
                etree.SubElement(entry, "{http://www.w3.org/2005/Atom}title").text = (
                    post.metadata.get(
                        "title", "Full Stack Software Engineer - Christian Ipanaque"
                    )
                )
                etree.SubElement(
                    entry,
                    "{http://www.w3.org/2005/Atom}link",
                    rel="alternate",
                    type="text/html",
                    href=f"https://christianipanaque.com/{post.metadata.get('permalink')}",
                )
                etree.SubElement(entry, "{http://www.w3.org/2005/Atom}id").text = (
                    f"tag:christianipanaque.com,2024:{post.metadata.get('permalink')}"
                )
                etree.SubElement(entry, "{http://www.w3.org/2005/Atom}updated").text = (
                    convert_to_rfc3339(post.metadata.get("date"))
                )
                etree.SubElement(
                    entry, "{http://www.w3.org/2005/Atom}published"
                ).text = convert_to_rfc3339(post.metadata.get("date"))
                author = etree.SubElement(entry, "{http://www.w3.org/2005/Atom}author")
                etree.SubElement(author, "{http://www.w3.org/2005/Atom}name").text = (
                    post.metadata.get("author", "Christian Ipanaque")
                )
                etree.SubElement(author, "{http://www.w3.org/2005/Atom}uri").text = (
                    "https://linkedin.com/in/christianipanaque-ai"
                )
                etree.SubElement(author, "{http://www.w3.org/2005/Atom}email").text = (
                    "christian.ipanaque@outlook.com"
                )
                content = etree.SubElement(
                    entry,
                    "{http://www.w3.org/2005/Atom}content",
                    type="text",
                )
                # Wrap markdown-converted HTML in CDATA
                content.text = f"Read about {post.metadata.get('title', 'Full Stack Software Engineer - Christian Ipanaque')}."

    # Write the XML to a file
    tree = etree.ElementTree(feed)
    with open("feed.atom", "wb") as file:
        tree.write(file, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    with open("rss.xml", "wb") as file:
        tree.write(file, pretty_print=True, xml_declaration=True, encoding="UTF-8")


# Usage
generate_atom_feed("_posts")

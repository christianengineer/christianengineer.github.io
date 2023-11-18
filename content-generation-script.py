import os
from openai import OpenAI
from datetime import datetime, timedelta
import random

# Initialize the OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")  # Get the API key from environment variable
)


def get_random_date(start_date, end_date):
    time_between_dates = end_date - start_date
    random_number_of_days = random.randrange(time_between_dates.days)
    return start_date + timedelta(days=random_number_of_days)


def generate_title(topic):
    print("generate_title start")
    print("generate_title chat_completion start")
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Generate a concise, SEO-optimized file name suitable for a URL, based on an article about {topic}. Include relevant SEO keywords, ensure the file name is catchy, and relevant to the topic. Use only letters and hyphens (no other characters) instead of spaces. Exclude the '.md' extension from the file name.",
            }
        ],
        model="gpt-4",
    )
    print("generate_title end")
    print("generate_title chat_completion end")
    return response.choices[0].message.content.strip()


def generate_article(topic):
    print("generate_article start")
    print("generate_article chat_completion start")
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Create a detailed, informative article in Markdown format about {topic}. The article should have a professional engineering tone, be well-structured with appropriate headings, bullet points, and code snippets (if applicable).",
            }
        ],
        model="gpt-4",
    )
    article = response.choices[0].message.content.strip()
    print("generate_article end")
    print("generate_article chat_completion end")
    return article


def main():
    print("script start")

    with open("topics.txt", "r") as file:
        print(f"opened topics")
        topics = [line.strip() for line in file if line.strip()]
        print(f"topics stripped")

    if topics:
        print("retrieve first topic from topics.txt")
        topic = topics.pop(0)

        print("generate title start")
        title = generate_title(topic)
        print("generate title end")

        # Define start and end dates
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 11, 1)

        random_date = get_random_date(start_date, end_date).strftime("%Y-%m-%d")

        # today_date = datetime.now().strftime("%Y-%m-%d")
        markdown_filename = f"_posts/{random_date}-{title}.md"

        os.makedirs(os.path.dirname(markdown_filename), exist_ok=True)

        if not os.path.exists(markdown_filename):
            with open(markdown_filename, "w") as md_file:
                # # Writing empty front matter
                # md_file.write("---\n---\n")
                md_file.write(f"---\npermalink: /{title}/\n---\n\n")
                print("Markdown file created with empty front matter")

        article = generate_article(topic)

        if article:
            # Write the article content to the already created Markdown file
            with open(markdown_filename, "a") as md_file:
                print("Markdown writing start")
                md_file.write(article)
                print("Markdown writing end")

            print(f"Article for '{topic}' generated and saved as {markdown_filename}.")

            print(f"Write the updated topics back to the file")
            with open("topics.txt", "w") as file:
                for remaining_topic in topics:
                    file.write(f"{remaining_topic}\n")
            print(f"topics.txt updated")
        else:
            print(f"Failed to generate article for '{topic}'.")
    else:
        print("No more topics to generate articles for.")
    print("script end")


if __name__ == "__main__":
    main()

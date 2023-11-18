import os
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")  # Get the API key from environment variable
)


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

        markdown_filename = f"_posts/{title}.md"

        os.makedirs(os.path.dirname(markdown_filename), exist_ok=True)

        if not os.path.exists(markdown_filename):
            with open(markdown_filename, "w") as md_file:
                md_file.write("---\n---\n")  # Writing empty front matter
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

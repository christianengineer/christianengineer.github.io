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


def generate_title_non_formated(topic):
    print("generate_title_non_formated start")
    print("generate_title_non_formated chat_completion start")
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Generate an SEO-optimized title for an article aimed at Senior Software Engineers about {topic}. The title should include advanced technical terms and relevant SEO keywords to attract experienced professionals. Ensure the title is engaging and reflects the in-depth nature of the article, catering specifically to a senior-level audience in the software engineering field.",
            }
        ],
        model="gpt-4",
    )
    print("generate_title_non_formated end")
    print("generate_title_non_formated chat_completion end")
    return response.choices[0].message.content.strip()


def generate_title(topic):
    print("generate_title start")
    print("generate_title chat_completion start")
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Develop a concise, SEO-focused file name tailored for web URLs, derived from an in-depth article about {topic}. The file name should integrate targeted SEO keywords to maximize search engine visibility, appealing to a technically adept audience. Ensure it's both engaging and directly related to the article's subject. Use a simple format with only letters and hyphens to replace spaces, maintaining web URL standards. Do not include the '.md' extension. The file name should effectively represent the article's core theme and be optimized for high-ranking search results.",
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
                "content": f"Create a comprehensive, technically rich article in Markdown format about {topic}, tailored for experienced software engineers. The article should delve into complex engineering concepts, demonstrating a high level of expertise. Ensure it's well-structured with clear, concise headings, bullet points for key takeaways, and include in-depth code snippets or examples where relevant. The tone should be professional yet engaging, providing deep insights and advanced knowledge that resonates with a senior engineering audience.",
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

        non_formated_title = generate_title_non_formated(topic)

        print("generate title start")
        title = generate_title(non_formated_title)
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
                # md_file.write(f"---\npermalink: posts/{title}\n---\n\n")
                md_file.write(
                    f"---\ntitle: `{non_formated_title}`\ndate: {random_date}\npermalink: posts/{title}\n---\n\n"
                )
                print("Markdown file created with empty front matter")

        article = generate_article(non_formated_title)

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
